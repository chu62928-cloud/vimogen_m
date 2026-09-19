#!/usr/bin/env python3
"""Audit current-environment M0 replays for the v0.3 paired protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.phase1 import MOTION_LAYOUT  # noqa: E402
from motion_rep.pose_authority import authority_project  # noqa: E402
from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    CURRENT_ENV_PAIRED_PROTOCOL,
    write_strict_json,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor_row(path: Path, row_index: int) -> tuple[str, list[int], str]:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
        raise ValueError(f"invalid tensor artifact in {path}")
    if not 0 <= row_index < tensor.shape[0]:
        raise ValueError(f"row {row_index} is outside {path}")
    row = tensor[row_index].detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(row.dtype).encode("ascii"))
    digest.update(json.dumps(list(row.shape), separators=(",", ":")).encode("ascii"))
    digest.update(row.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest(), list(row.shape), str(row.dtype)


def direct_max(candidate: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor) -> float:
    values = []
    for span in (MOTION_LAYOUT.body_pose, MOTION_LAYOUT.root_rotation, MOTION_LAYOUT.root_translation):
        values.append((candidate[..., span] - reference[..., span])[valid].abs().max())
    return float(torch.stack(values).max().item())


def load_stage(path: Path, mean: torch.Tensor, std: torch.Tensor, valid: torch.Tensor, row_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = torch.load(path, map_location="cpu", weights_only=True).float()
    if normalized.ndim != 3 or normalized.shape[-1] != 276:
        raise ValueError(f"invalid M0 tensor shape in {path}: {tuple(normalized.shape)}")
    row = normalized[row_index : row_index + 1]
    physical = row * std.view(1, 1, -1) + mean.view(1, 1, -1)
    authoritative = authority_project(physical, valid_mask=valid, output_dtype=torch.float32).physical_motion
    return physical, authoritative


def find_valid_mask(run_root: Path, row_index: int, frozen_valid: torch.Tensor) -> tuple[bool | None, str | None]:
    for archive_path in sorted(run_root.rglob("mbench_raw_norm_batch.pt")):
        archive = torch.load(archive_path, map_location="cpu", weights_only=True)
        archived = archive.get("motion_mask") if isinstance(archive, dict) else None
        if isinstance(archived, torch.Tensor) and archived.ndim == 2 and 0 <= row_index < archived.shape[0]:
            return bool(torch.equal(archived[row_index].bool(), frozen_valid)), str(archive_path)
    return None, None


def audit(
    frozen_protocol: Path,
    runs: list[str],
    output: Path,
    *,
    sample_index: int = 1,
    threshold: float = 2.0e-3,
    legacy_protocol: Path | None = None,
) -> dict[str, Any]:
    protocol = json.loads((frozen_protocol / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol") != CURRENT_ENV_PAIRED_PROTOCOL:
        raise ValueError("frozen protocol is not the v0.3 current-environment protocol")
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    frozen_all = torch.load(frozen_protocol / "m0_physical.pt", map_location="cpu", weights_only=True).float()
    valid_all = torch.load(frozen_protocol / "valid_mask.pt", map_location="cpu", weights_only=True).bool()
    frozen = authority_project(
        frozen_all[sample_index : sample_index + 1],
        valid_mask=valid_all[sample_index : sample_index + 1],
        output_dtype=torch.float32,
    ).physical_motion
    reconstruction = direct_max(frozen, frozen, valid_all[sample_index : sample_index + 1])
    records: list[dict[str, Any]] = []
    for item in runs:
        parts = item.split("=", 2)
        if len(parts) != 3:
            raise ValueError("--run must be label=run_root=row_index")
        label, raw_path, row_text = parts
        run_root = Path(raw_path)
        row_index = int(row_text)
        valid = valid_all[sample_index : sample_index + 1]
        stages: dict[str, Any] = {}
        for stage, filename in (("raw", "m0_raw_norm_batch.pt"), ("official_pre_cast", "m0_official_norm_batch.pt"), ("official", "m0_official_post_cast_norm_batch.pt")):
            path = run_root / "m0_artifacts" / "batch_000" / filename
            if not path.is_file():
                continue
            physical, authoritative = load_stage(path, mean, std, valid, row_index)
            row_hash, row_shape, row_dtype = sha256_tensor_row(path, row_index)
            stages[stage] = {
                "path": str(path),
                "sha256": sha256_path(path),
                "row_sha256": row_hash,
                "row_shape": row_shape,
                "row_dtype": row_dtype,
                "authority_direct_max_abs": direct_max(authoritative, frozen, valid),
                "authority_full_max_abs": float((authoritative - frozen).abs()[valid].max().item()),
                "raw_full_max_abs": float((physical - frozen).abs()[valid].max().item()),
            }
        snapshot_path = run_root / "input_snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else {}
        valid_equal, valid_path = find_valid_mask(run_root, row_index, valid_all[sample_index])
        records.append({
            "label": label,
            "run_root": str(run_root),
            "sample_index": row_index,
            "z0_row_sha256": sha256_tensor_row(run_root / "m0_artifacts" / "batch_000" / "z0_replayed.pt", row_index)[0] if (run_root / "m0_artifacts" / "batch_000" / "z0_replayed.pt").is_file() else None,
            "valid_mask_equal": valid_equal,
            "valid_mask_artifact": valid_path,
            "input_snapshot": str(snapshot_path) if snapshot else None,
            "input_snapshot_fingerprint": {
                key: snapshot.get(key)
                for key in ("vimogen_checkpoint", "motion_mean", "motion_std", "smplx_model", "reference_protocol", "manifest", "sampling_schedule", "guidance_step_mask", "runtime_fingerprint")
                if key in snapshot
            },
            "stages": stages,
        })
    snapshots = [r["input_snapshot_fingerprint"] for r in records]
    fingerprint_keys = ("vimogen_checkpoint", "motion_mean", "motion_std", "smplx_model", "reference_protocol", "sampling_schedule", "guidance_step_mask", "runtime_fingerprint")
    fingerprints_complete = bool(snapshots) and all(all(key in value for key in fingerprint_keys) for value in snapshots)
    metadata_consistent = fingerprints_complete and all(all(value.get(key) == snapshots[0].get(key) for key in fingerprint_keys) for value in snapshots)
    manifests_equal = fingerprints_complete and all(value.get("manifest") == snapshots[0].get("manifest") for value in snapshots)
    valid_present = bool(records) and all(value["valid_mask_equal"] is not None for value in records)
    valid_consistent = valid_present and all(value["valid_mask_equal"] for value in records)
    row_hashes = {stage: [r["stages"].get(stage, {}).get("row_sha256") for r in records] for stage in ("raw", "official_pre_cast", "official")}
    stage_replay_equal = {stage: bool(values) and None not in values and len(set(values)) == 1 for stage, values in row_hashes.items()}
    official_pass = all(r["stages"].get("official_pre_cast", {}).get("authority_direct_max_abs", float("inf")) <= threshold for r in records)
    windows = {str(case["sample_id"]): {side: case["sides"][side]["stable_window"].get("status") for side in ("left", "right")} for case in protocol.get("cases", [])}
    legacy_diag = None
    if legacy_protocol is not None:
        legacy = torch.load(legacy_protocol / "m0_physical.pt", map_location="cpu", weights_only=True).float()
        legacy_case = authority_project(legacy[1:2], valid_mask=valid_all[1:2], output_dtype=torch.float32).physical_motion
        legacy_diag = {"protocol": str(legacy_protocol), "direct_max_abs": direct_max(frozen, legacy_case, valid_all[1:2]), "reference_only": True}
    result = {
        "protocol": CURRENT_ENV_PAIRED_PROTOCOL,
        "frozen_protocol": str(frozen_protocol),
        "frozen_sample_index": sample_index,
        "threshold_direct_max_abs": threshold,
        "freeze_reconstruction_direct_max_abs": reconstruction,
        "windows": windows,
        "legacy_v3_diagnostic": legacy_diag,
        "input_consistency": {
            "sample_noise_row_hash_equal": bool(records) and len({r.get("z0_row_sha256") for r in records}) == 1 and None not in {r.get("z0_row_sha256") for r in records},
            "valid_mask_equal": valid_consistent,
            "valid_mask_artifacts_present": valid_present,
            "input_fingerprints_complete": fingerprints_complete,
            "checkpoint_mean_std_schedule_environment_equal": metadata_consistent,
            "manifest_equal": manifests_equal,
            "stage_row_hash_equal": stage_replay_equal,
        },
        "records": records,
        "status": "M0_PAIRING_PASS" if records and reconstruction <= threshold and metadata_consistent and valid_consistent and official_pass and all(stage_replay_equal.get(stage, False) for stage in ("raw", "official_pre_cast")) and all(value == "PASS" for item in windows.values() for value in item.values()) else "M0_PAIRING_FAIL",
    }
    write_strict_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="label=run_root=row_index")
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--legacy-protocol", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.frozen_protocol, args.run, args.output, sample_index=args.sample_index, legacy_protocol=args.legacy_protocol), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
