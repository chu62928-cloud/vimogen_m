"""Auditable three-way recovery conversion for ViMoGen MBench outputs.

The module deliberately separates generation from representation recovery.
Each input 276D tensor is loaded once and converted by all three frozen
strategies.  The resulting manifest is the unit used by later paired
statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


PROTOCOL = "vimogen_mbench_threeway_recovery_v1"
RECONCILIATION_PROTOCOL = "vimogen_276d_control_aware_reconciliation_v1"
METHODS = ("absolute_position", "velocity_integral", "reconciled")

BASE_CONVERSION = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)
FRONT_ROTATION = torch.tensor(
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=torch.float32,
)
COORD_CONVERSION = FRONT_ROTATION @ BASE_CONVERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_motion_tensor(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, Mapping):
        if "motion" not in data:
            raise ValueError(f"{path} contains a mapping without motion")
        data = data["motion"]
    tensor = torch.as_tensor(data).float()
    if tensor.ndim != 2 or tensor.shape[-1] != 276:
        raise ValueError(f"expected [T,276] at {path}, got {tuple(tensor.shape)}")
    if tensor.shape[0] < 2 or not torch.isfinite(tensor).all():
        raise ValueError(f"invalid motion tensor at {path}")
    return tensor


def find_motion_file(folder: Path) -> Path | None:
    for name in (
        "motion_gen_condition_on_motion.pt",
        "motion_gen_condition_on_text.pt",
    ):
        candidate = folder / name
        if candidate.exists():
            return candidate
    candidates = sorted(folder.glob("*.pt"))
    return candidates[0] if candidates else None


def recover_motion_variants(motion: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the three frozen finalizers on one physical 276D tensor."""

    from motion_rep.consistent_finalizer import finalize_consistent_motion_tensor
    from motion_rep.reconciliation import (
        ReconciliationConfig,
        reconcile_motion_tensor,
    )
    from motion_rep.unified_finalizer import finalize_motion_tensor

    config = ReconciliationConfig(
        correction_window=9,
        anchor_weight=1.0,
        root_rotation_anchor_weight=1.0,
    )
    outputs = {
        "absolute_position": finalize_consistent_motion_tensor(motion).motion.float(),
        "velocity_integral": finalize_motion_tensor(motion).motion.float(),
        "reconciled": reconcile_motion_tensor(motion, config=config).motion.float(),
    }
    shape = tuple(outputs["absolute_position"].shape)
    for method, output in outputs.items():
        if tuple(output.shape) != shape:
            raise ValueError(f"{method} changed shape to {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise ValueError(f"{method} produced non-finite values")
    return outputs


def motion_to_joints(motion: torch.Tensor) -> np.ndarray:
    from motion_rep.retarget_motion import motion_rep_to_SMPL

    _, recovered_joints = motion_rep_to_SMPL(
        motion,
        recover_from_velocity=True,
        equal_length=False,
    )
    joints = torch.einsum(
        "ij,tvj->tvi",
        COORD_CONVERSION.to(device=recovered_joints.device, dtype=recovered_joints.dtype),
        recovered_joints,
    )
    result = joints.detach().cpu().numpy().astype("float32")
    if result.ndim != 3 or result.shape[1:] != (22, 3):
        raise ValueError(f"expected [T,22,3] joints, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("joint output contains non-finite values")
    return result


def _auc(values: np.ndarray) -> float:
    if len(values) <= 1:
        return float(values[0]) if len(values) else 0.0
    time = np.linspace(0.0, 1.0, len(values))
    return float(np.trapezoid(values, time) if hasattr(np, "trapezoid") else np.trapz(values, time))


def _slope(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    x = x - x.mean()
    y = values.astype(np.float64) - values.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom else 0.0


def _frame_mean_displacement(joints: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return np.linalg.norm(joints - anchor, axis=-1).mean(axis=-1)


def trajectory_diagnostics(joints: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    frame_values = _frame_mean_displacement(joints, anchor)
    root_values = np.linalg.norm(joints[:, 0] - anchor[:, 0], axis=-1)
    return {
        "anchor_deviation_mean_m": float(frame_values.mean()),
        "anchor_deviation_endpoint_m": float(frame_values[-1]),
        "root_endpoint_deviation_m": float(root_values[-1]),
        "anchor_drift_auc_m": _auc(frame_values),
        "anchor_drift_slope_m_per_frame": _slope(frame_values),
        "sequence_length_frames": int(len(joints)),
    }


def raw_channel_diagnostics(motion: torch.Tensor) -> dict[str, float]:
    position = motion[:, 126:192].reshape(-1, 22, 3).numpy()
    velocity = motion[:, 192:258].reshape(-1, 22, 3).numpy()
    differences = np.diff(position, axis=0)
    residual = np.linalg.norm(differences - velocity[:-1], axis=-1).mean(axis=-1)
    integrated = position[:1] + np.cumsum(velocity[:-1], axis=0)
    separation = np.linalg.norm(position[1:] - integrated, axis=-1).mean(axis=-1)
    return {
        "raw_position_velocity_residual_mean_m_per_frame": float(residual.mean()),
        "raw_position_velocity_residual_median_m_per_frame": float(np.median(residual)),
        "raw_position_velocity_residual_max_m_per_frame": float(residual.max()),
        "raw_integrated_separation_endpoint_m": float(separation[-1]),
        "raw_integrated_separation_auc_m": _auc(separation),
        "raw_integrated_separation_slope_m_per_frame": _slope(separation),
    }


def final_internal_residual(motion: torch.Tensor) -> dict[str, float]:
    position = motion[:, 126:192].reshape(-1, 22, 3).numpy()
    velocity = motion[:, 192:258].reshape(-1, 22, 3).numpy()
    residual = np.linalg.norm(np.diff(position, axis=0) - velocity[:-1], axis=-1)
    return {
        "final_position_velocity_residual_m_per_frame_median": float(np.median(residual)),
        "final_position_velocity_residual_m_per_frame_max": float(residual.max()),
    }


def build_sample_record(
    *,
    sample_id: str,
    source_path: Path,
    source_file_sha256: str,
    source_tensor_sha256: str,
    condition: str,
    seed: int | None,
    motion: torch.Tensor,
    variants: dict[str, torch.Tensor],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    joints = {method: motion_to_joints(value) for method, value in variants.items()}
    anchor = joints["absolute_position"]
    methods = {}
    for method, value in variants.items():
        metrics = trajectory_diagnostics(joints[method], anchor)
        metrics.update(final_internal_residual(value))
        metrics["output_tensor_sha256"] = sha256_tensor(value)
        metrics["output_shape"] = list(value.shape)
        metrics["output_path"] = str(output_paths[method])
        methods[method] = metrics
    return {
        "sample_id": sample_id,
        "source_path": str(source_path),
        "source_file_sha256": source_file_sha256,
        "source_tensor_sha256": source_tensor_sha256,
        "condition": condition,
        "seed": seed,
        "frame_count": int(motion.shape[0]),
        "raw_channel_diagnostics": raw_channel_diagnostics(motion),
        "methods": methods,
    }


def _sorted_folders(input_dir: Path) -> list[Path]:
    folders = [path for path in input_dir.iterdir() if path.is_dir()]
    try:
        return sorted(folders, key=lambda path: int(path.name))
    except ValueError:
        return sorted(folders, key=lambda path: path.name)


def organize_directory(
    input_dir: Path,
    output_root: Path,
    *,
    condition: str = "unknown",
    seed: int | None = None,
    expected_count: int | None = None,
    verify_only: bool = False,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_root = output_root.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    folders = _sorted_folders(input_dir)
    if expected_count is not None and len(folders) != expected_count:
        raise ValueError(f"expected {expected_count} folders, found {len(folders)}")
    if not verify_only:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty output directory: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            (output_root / method).mkdir(parents=True, exist_ok=True)

    records = []
    errors = []
    for folder in folders:
        motion_path = find_motion_file(folder)
        if motion_path is None:
            errors.append({"sample_id": folder.name, "error": "missing motion file"})
            continue
        try:
            motion = load_motion_tensor(motion_path)
            variants = recover_motion_variants(motion)
            output_paths = {
                method: output_root / method / f"{folder.name}.npy"
                for method in METHODS
            }
            if not verify_only:
                for method, value in variants.items():
                    np.save(output_paths[method], motion_to_joints(value))
            record = build_sample_record(
                sample_id=folder.name,
                source_path=motion_path,
                source_file_sha256=sha256_file(motion_path),
                source_tensor_sha256=sha256_tensor(motion),
                condition=condition,
                seed=seed,
                motion=motion,
                variants=variants,
                output_paths=output_paths,
            )
            records.append(record)
        except Exception as exc:
            errors.append({"sample_id": folder.name, "error": repr(exc)})

    if expected_count is not None and len(records) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} successful records, found {len(records)}; "
            f"errors={errors[:3]}"
        )
    payload = {
        "status": "VALID" if not errors else "INVALID",
        "protocol": PROTOCOL,
        "reconciliation_protocol": RECONCILIATION_PROTOCOL,
        "condition": condition,
        "seed": seed,
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "expected_count": expected_count,
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors,
        "records": records,
    }
    if not verify_only:
        (output_root / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


__all__ = [
    "METHODS",
    "PROTOCOL",
    "RECONCILIATION_PROTOCOL",
    "build_sample_record",
    "find_motion_file",
    "load_motion_tensor",
    "organize_directory",
    "raw_channel_diagnostics",
    "recover_motion_variants",
    "trajectory_diagnostics",
]
