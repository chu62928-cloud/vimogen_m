#!/usr/bin/env python3
"""Run v3.1 contact windows or v3.2 full-sequence compensation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import sys
from typing import Any

import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import PROTOCOL_NAME
from motion_rep.smplx_utils import default_smpl_model_path
from sampling.pelvis_contact_compensation_v3 import PelvisCompensationConfig, PelvisContactCompensationSolver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _next_attempt(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while (parent / f"attempt_{index:02d}").exists():
        index += 1
    result = parent / f"attempt_{index:02d}"
    result.mkdir()
    return result


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    patches = json.loads((root / "foot_patches.json").read_text(encoding="utf-8"))
    m0 = torch.load(root / "m0_physical.pt", map_location="cpu", weights_only=True).float()
    valid = torch.load(root / "valid_mask.pt", map_location="cpu", weights_only=True).bool()
    if m0.ndim != 3 or m0.shape[-1] != 276 or valid.shape != m0.shape[:2]:
        raise ValueError("frozen M0 and valid mask have incompatible shapes")
    return protocol, patches, m0, valid


def _model_vertices(model: SMPLX, motion: torch.Tensor, device: torch.device) -> torch.Tensor:
    from evaluation.relative_root_trunk_v2_1 import direct_smpl_parameters
    with torch.inference_mode():
        params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
        params = {key: value[0] for key, value in params.items()}
        return model(**params, return_verts=True).vertices.detach().cpu()


def _run_case(
    *,
    phase: str,
    case: dict[str, Any],
    protocol_root: Path,
    patches: dict[str, Any],
    m0: torch.Tensor,
    valid: torch.Tensor,
    model: SMPLX,
    device: torch.device,
    dose: float,
    output_root: Path,
) -> dict[str, Any]:
    case_id = str(case["sample_id"])
    source_index = int(case["source_index"])
    length = int(valid[source_index].sum().item())
    base = m0[source_index, :length]
    base_mask = valid[source_index, :length]
    config = PelvisCompensationConfig()
    if phase == "v3_1_window_feasibility":
        sides = {}
        for side in ("left", "right"):
            side_root = output_root / side
            side_root.mkdir(parents=True, exist_ok=True)
            window = case["sides"][side]["stable_window"]
            if window.get("status") != "PASS" or not window.get("frames"):
                sides[side] = {"status": "NOT_EVALUABLE", "window": window}
                continue
            start = int(window["window_start"])
            end = int(window["window_end_exclusive"])
            stable_masks = {
                name: torch.as_tensor(
                    case["sides"][name]["evidence"]["valid_masks"]["flat_contact"],
                    dtype=torch.bool,
                )[start:end]
                for name in ("left", "right")
            }
            current: torch.Tensor | None = None
            dose_records = []
            for continuation_dose in (2.0, 5.0, float(dose)):
                solver = PelvisContactCompensationSolver(
                    base[start:end], model, patches,
                    valid_mask=torch.ones(end - start, dtype=torch.bool),
                    stable_masks=stable_masks,
                    config=config, device=device,
                )
                result = solver.solve(continuation_dose, initial_motion=current)
                best_candidate = result["motion"]
                if bool(result.get("feasible", False)):
                    current = best_candidate
                dose_records.append({key: value for key, value in result.items() if key != "motion"})
                torch.save(best_candidate, side_root / f"dose_{continuation_dose:+g}deg_best.pt")
            selected = current if current is not None and bool(dose_records[-1].get("feasible", False)) else base[start:end].cpu()
            sides[side] = {"status": dose_records[-1]["status"], "feasible": bool(dose_records[-1].get("feasible", False)), "window": window, "dose_records": dose_records, "motion_path": str(side_root / "candidate.pt")}
            torch.save(selected, side_root / "candidate.pt")
        return {"phase": phase, "sample_id": case_id, "status": "COMPLETED", "sides": sides}
    stable_masks = {
        name: torch.as_tensor(
            case["sides"][name]["evidence"]["valid_masks"]["flat_contact"],
            dtype=torch.bool,
        )[:length]
        for name in ("left", "right")
    }
    solver = PelvisContactCompensationSolver(
        base, model, patches, valid_mask=base_mask, stable_masks=stable_masks,
        config=config, device=device,
    )
    result = solver.solve(float(dose))
    best_candidate = result["motion"]
    feasible = bool(result.get("feasible", False))
    candidate = best_candidate if feasible else base.cpu()
    torch.save(base.cpu(), output_root / "m0_physical.pt")
    torch.save(candidate, output_root / "selected_motion.pt")
    if not feasible:
        torch.save(best_candidate, output_root / "best_infeasible_motion.pt")
    return {
        "phase": phase,
        "sample_id": case_id,
        "status": result["status"],
        "feasible": feasible,
        "target_delta_deg": float(dose),
        "frames": length,
        "solver": {key: value for key, value in result.items() if key != "motion"},
        "selected_motion": "selected_motion.pt",
        "fallback_is_m0": not feasible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("v3_1_window_feasibility", "v3_2_full_sequence"), required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol, patches, m0, valid = _load_protocol(args.protocol_root)
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise ValueError("protocol root is not a v3 pelvis contact protocol")
    cases = protocol["cases"]
    if args.sample_id is not None:
        cases = [case for case in cases if str(case["sample_id"]) == str(args.sample_id)]
    if not cases:
        raise ValueError("no requested case in frozen protocol")
    model_path = Path(args.model_path) if args.model_path is not None else Path(default_smpl_model_path("smplx", ROOT))
    device = torch.device(args.device)
    max_frames = int(valid.sum(dim=1).max().item())
    model = SMPLX(model_path=str(model_path), gender="neutral", num_betas=10, batch_size=max_frames, use_pca=False).to(device)
    output_base = args.output_root or (ROOT / "results/phase8/pelvis_contact_compensation_v3")
    output_base = Path(output_base) / args.phase
    records = []
    for case in cases:
        parent = output_base / f"sample_{case['sample_id']}" / f"dose_{args.target_delta_deg:+g}deg"
        run_root = _next_attempt(parent)
        record: dict[str, Any] = {
            "status": "RUNNING",
            "protocol": protocol["protocol"],
            "phase": args.phase,
            "sample_id": str(case["sample_id"]),
            "seed": int(case.get("seed", 0)),
            "target_delta_deg": float(args.target_delta_deg),
            "run_root": str(run_root),
            "protocol_root": str(args.protocol_root),
            "protocol_sha256": _sha256(args.protocol_root / "protocol.json"),
            "patch_sha256": protocol["foot_patches"]["sha256"],
            "git_revision": _git_revision(),
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path) if model_path.is_file() else None,
        }
        record_path = run_root / "run_record.json"
        _write_json(record_path, record)
        started = time.perf_counter()
        try:
            result = _run_case(phase=args.phase, case=case, protocol_root=args.protocol_root, patches=patches, m0=m0, valid=valid, model=model, device=device, dose=args.target_delta_deg, output_root=run_root)
            record.update(result)
            record["status"] = "COMPLETED"
        except Exception as exc:
            record.update({"status": "FAILED", "error": repr(exc)})
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            _write_json(record_path, record)
        records.append(record)
    print(json.dumps({"phase": args.phase, "runs": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
