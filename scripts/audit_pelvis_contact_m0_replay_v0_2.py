"""Audit M0 replay variants against the frozen v3.0.1 physical endpoint."""

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

from motion_rep.phase1 import MOTION_LAYOUT
from motion_rep.pose_authority import authority_project
from sampling.pelvis_contact_flow_projection_v0_1 import write_strict_json


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor_row(path: Path, row_index: int) -> tuple[str, list[int], str]:
    """Hash one sample row, independent of the replay batch size."""

    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
        raise ValueError(f"invalid tensor artifact in {path}")
    if not 0 <= row_index < tensor.shape[0]:
        raise ValueError(f"row {row_index} is outside {path}")
    row = tensor[row_index].detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(row.dtype).encode("ascii"))
    digest.update(json.dumps(list(row.shape), separators=(",", ":")).encode("ascii"))
    digest.update(row.numpy().tobytes(order="C"))
    return digest.hexdigest(), list(row.shape), str(row.dtype)


def _direct_max(candidate: torch.Tensor, frozen: torch.Tensor, valid: torch.Tensor) -> float:
    values = []
    for span in (
        MOTION_LAYOUT.body_pose,
        MOTION_LAYOUT.root_rotation,
        MOTION_LAYOUT.root_translation,
    ):
        values.append((candidate[..., span] - frozen[..., span])[valid].abs().max())
    return float(torch.stack(values).max().item())


def _load_stage(
    path: Path,
    mean: torch.Tensor,
    std: torch.Tensor,
    valid: torch.Tensor,
    row_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = torch.load(path, map_location="cpu", weights_only=True).float()
    if normalized.ndim != 3 or normalized.shape[-1] != 276:
        raise ValueError(f"invalid M0 tensor shape in {path}: {tuple(normalized.shape)}")
    if not 0 <= row_index < normalized.shape[0]:
        raise ValueError(f"row {row_index} is outside {path}")
    row = normalized[row_index : row_index + 1]
    physical = row * std.view(1, 1, -1) + mean.view(1, 1, -1)
    authoritative = authority_project(
        physical, valid_mask=valid, output_dtype=torch.float32
    ).physical_motion
    return physical, authoritative


def audit(
    frozen_protocol: Path,
    runs: list[str],
    output: Path,
    *,
    sample_index: int,
    threshold: float = 2.0e-3,
) -> dict[str, Any]:
    protocol = json.loads((frozen_protocol / "protocol.json").read_text(encoding="utf-8"))
    mean = torch.from_numpy(np.load(protocol["inputs"]["mean"]["path"])).float()
    std = torch.from_numpy(np.load(protocol["inputs"]["std"]["path"])).float()
    frozen_all = torch.load(
        frozen_protocol / "m0_physical.pt", map_location="cpu", weights_only=True
    ).float()
    valid_all = torch.load(
        frozen_protocol / "valid_mask.pt", map_location="cpu", weights_only=True
    ).bool()
    frozen = authority_project(
        frozen_all[sample_index : sample_index + 1],
        valid_mask=valid_all[sample_index : sample_index + 1],
        output_dtype=torch.float32,
    ).physical_motion
    records: list[dict[str, Any]] = []
    for item in runs:
        label, raw_path, raw_index = item.split("=", 2)
        run_root = Path(raw_path)
        row_index = sample_index if raw_index == "" else int(raw_index)
        valid = valid_all[sample_index : sample_index + 1]
        stage_rows: dict[str, Any] = {}
        for stage, filename in (
            ("raw", "m0_raw_norm_batch.pt"),
            ("official_pre_cast", "m0_official_norm_batch.pt"),
            ("official_post_cast", "m0_official_post_cast_norm_batch.pt"),
        ):
            path = run_root / "m0_artifacts" / "batch_000" / filename
            if not path.is_file():
                continue
            physical, authoritative = _load_stage(
                path, mean, std, valid, row_index
            )
            stage_rows[stage] = {
                "path": str(path),
                "sha256": sha256_path(path),
                "raw_full_max_abs": float((physical - frozen).abs()[valid].max().item()),
                "authority_direct_max_abs": _direct_max(authoritative, frozen, valid),
                "authority_full_max_abs": float((authoritative - frozen).abs()[valid].max().item()),
            }
        z0 = run_root / "m0_artifacts" / "batch_000" / "z0_replayed.pt"
        snapshot_path = run_root / "input_snapshot.json"
        snapshot = (
            json.loads(snapshot_path.read_text(encoding="utf-8"))
            if snapshot_path.is_file()
            else {}
        )
        z0_row_hash = z0_shape = z0_dtype = None
        if z0.is_file():
            z0_row_hash, z0_shape, z0_dtype = sha256_tensor_row(z0, row_index)
        record = {
            "label": label,
            "run_root": str(run_root),
            "sample_index": row_index,
            "z0_sha256": sha256_path(z0) if z0.is_file() else None,
            "z0_row_sha256": z0_row_hash,
            "z0_row_shape": z0_shape,
            "z0_row_dtype": z0_dtype,
            "input_snapshot": str(snapshot_path) if snapshot else None,
            "input_snapshot_fingerprint": {
                key: snapshot.get(key)
                for key in (
                    "vimogen_checkpoint",
                    "smplx_model",
                    "frozen_v3_protocol",
                    "manifest",
                    "sampling_schedule",
                    "guidance_step_mask",
                )
                if key in snapshot
            },
            "stages": stage_rows,
        }
        official = stage_rows.get("official_pre_cast")
        record["status"] = (
            "PASS"
            if official is not None
            and official["authority_direct_max_abs"] <= threshold
            else "FAIL"
        )
        records.append(record)
    fingerprints = [
        record["input_snapshot_fingerprint"]
        for record in records
        if record["input_snapshot_fingerprint"]
    ]
    fingerprint_keys = (
        "vimogen_checkpoint",
        "smplx_model",
        "frozen_v3_protocol",
        "manifest",
        "sampling_schedule",
        "guidance_step_mask",
    )
    metadata_consistent = bool(fingerprints) and all(
        all(item.get(key) == fingerprints[0].get(key) for item in fingerprints)
        for key in fingerprint_keys
    )
    row_hashes = [record["z0_row_sha256"] for record in records]
    sample_noise_consistent = bool(row_hashes) and len(set(row_hashes)) == 1
    result = {
        "protocol": "vimogen_pelvis_contact_flow_projection_v0_2_temporal_contact",
        "frozen_protocol": str(frozen_protocol),
        "frozen_sample_index": sample_index,
        "threshold_direct_max_abs": threshold,
        "input_consistency": {
            "sample_noise_row_hash_equal": sample_noise_consistent,
            "checkpoint_mean_std_schedule_equal": metadata_consistent,
        },
        "records": records,
        "status": (
            "PASS"
            if records
            and metadata_consistent
            and sample_noise_consistent
            and all(r["status"] == "PASS" for r in records)
            else "FAIL"
        ),
    }
    write_strict_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label=run_root=row_index; repeat for controlled replay variants",
    )
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.frozen_protocol, args.run, args.output, sample_index=args.sample_index), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
