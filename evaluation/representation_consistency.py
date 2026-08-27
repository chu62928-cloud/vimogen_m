"""Read-only diagnostics for ViMoGen's redundant 276-D motion channels.

The public functions in this module consume physical-space ``[T, 276]``
motions.  They do not alter the motion tensor.  The representation contains
direct joint samples and forward differences side by side, so the first two
diagnostics deliberately exclude the final row, whose forward difference
does not have a corresponding direct ``T+1`` pose in the packed tensor.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.rotation_transform import rot6d_to_axis_angle


JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)

# SMPL-X parent indices for the first 22 body joints used by ViMoGen.  The
# model's own parent tensor is preferred when available; this fallback keeps
# scale diagnostics usable without constructing a body model.
JOINT_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 15, 16, 17, 18, 19, 20)


def _as_physical_motion(motion: torch.Tensor) -> torch.Tensor:
    if not isinstance(motion, torch.Tensor):
        raise TypeError(f"motion must be a torch.Tensor, got {type(motion)!r}")
    if motion.ndim != 2 or motion.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(f"expected physical [T,276], got {tuple(motion.shape)}")
    if motion.shape[0] < 1:
        raise ValueError("motion must contain at least one frame")
    if not torch.isfinite(motion).all():
        raise ValueError("motion contains non-finite values")
    return motion.float()


def _stats(values: torch.Tensor | np.ndarray | Sequence[float]) -> dict[str, float]:
    value = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if value.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(value.mean()),
        "median": float(value.median()),
        "p95": float(torch.quantile(value, 0.95)),
        "max": float(value.max()),
    }


def _linear_slope(values: torch.Tensor) -> float:
    if values.numel() < 2:
        return 0.0
    x = torch.arange(values.numel(), dtype=torch.float64, device=values.device)
    y = values.double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = (x * x).sum()
    return float((x * y).sum() / denominator) if denominator else 0.0


def _rotation_angle_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = first @ second.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def _parent_indices(model: Any | None) -> tuple[int, ...]:
    if model is None or not hasattr(model, "parents"):
        return JOINT_PARENTS
    parents = model.parents.detach().cpu().reshape(-1).tolist()[:22]
    if len(parents) != 22:
        return JOINT_PARENTS
    return tuple(int(item) for item in parents)


def _scale_from_joints(joints: torch.Tensor, model: Any | None) -> float:
    parents = _parent_indices(model)
    edges = [(index, parent) for index, parent in enumerate(parents) if parent >= 0 and parent < 22]
    if not edges:
        return 1.0
    lengths = torch.stack([
        torch.linalg.vector_norm(joints[:, child] - joints[:, parent], dim=-1)
        for child, parent in edges
    ], dim=-1)
    scale = lengths.mean(dim=-1).median()
    return max(float(scale), 1e-8)


def _quartile_delta(values: torch.Tensor) -> tuple[float, float, float]:
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    count = max(1, int(np.ceil(values.numel() / 4)))
    early = float(values[:count].mean())
    late = float(values[-count:].mean())
    return early, late, late - early


def _fk_joints(motion: torch.Tensor, model: Any) -> torch.Tensor:
    frames = motion.shape[0]
    body6d = motion[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6)
    root6d = motion[:, MOTION_LAYOUT.root_rotation].reshape(frames, 6)
    body_aa = rot6d_to_axis_angle(body6d.reshape(-1, 6)).reshape(frames, 63)
    root_aa = rot6d_to_axis_angle(root6d)
    transl = motion[:, MOTION_LAYOUT.root_translation]
    # Supplying beta explicitly makes the zero-shape convention independent of
    # the constructor's batch_size and supports variable-length references.
    betas = torch.zeros((frames, 10), dtype=motion.dtype, device=motion.device)
    with torch.no_grad():
        output = model(
            global_orient=root_aa,
            body_pose=body_aa,
            left_hand_pose=torch.zeros((frames, 45), dtype=motion.dtype, device=motion.device),
            right_hand_pose=torch.zeros((frames, 45), dtype=motion.dtype, device=motion.device),
            jaw_pose=torch.zeros((frames, 3), dtype=motion.dtype, device=motion.device),
            leye_pose=torch.zeros((frames, 3), dtype=motion.dtype, device=motion.device),
            reye_pose=torch.zeros((frames, 3), dtype=motion.dtype, device=motion.device),
            transl=transl,
            betas=betas,
            expression=torch.zeros((frames, 10), dtype=motion.dtype, device=motion.device),
        )
    return output.joints[:, :22].float()


def compute_sequence_metrics(
    motion: torch.Tensor,
    *,
    model: Any | None = None,
    sample_id: str | None = None,
    method: str | None = None,
    output_stage: str | None = None,
    source_kind: str | None = None,
) -> dict[str, Any]:
    """Compute all representation diagnostics for one physical motion."""

    motion = _as_physical_motion(motion)
    frames = int(motion.shape[0])
    joints = motion[:, MOTION_LAYOUT.joints].reshape(frames, 22, 3)
    joints_velocity = motion[:, MOTION_LAYOUT.joints_velocity].reshape(frames, 22, 3)

    if frames > 1:
        direct_step = joints[1:] - joints[:-1]
        speed_residual = torch.linalg.vector_norm(direct_step - joints_velocity[:-1], dim=(1, 2))
        direct_step_norm = torch.linalg.vector_norm(direct_step, dim=(1, 2))
        integrated = torch.cat(
            (joints[:1], joints[:1] + torch.cumsum(joints_velocity[:-1], dim=0)), dim=0
        )
        drift = torch.linalg.vector_norm(joints - integrated, dim=(1, 2))
        speed_joint_residual = torch.linalg.vector_norm(direct_step - joints_velocity[:-1], dim=-1)
        drift_joint = torch.linalg.vector_norm(joints - integrated, dim=-1)
    else:
        speed_residual = torch.empty(0)
        direct_step_norm = torch.empty(0)
        integrated = joints.clone()
        drift = torch.zeros(1)
        speed_joint_residual = torch.empty((0, 22))
        drift_joint = torch.zeros((1, 22))

    body_scale = _scale_from_joints(joints, model)
    speed_mean = float(speed_residual.mean()) if speed_residual.numel() else 0.0
    direct_step_mean = float(direct_step_norm.mean()) if direct_step_norm.numel() else 0.0
    speed_ratio = speed_mean / max(direct_step_mean, 1e-8)
    drift_early, drift_late, drift_delta = _quartile_delta(drift)

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "method": method,
        "output_stage": output_stage,
        "source_kind": source_kind,
        "frame_count": frames,
        "body_scale_mean_bone_m": body_scale,
        "speed_residual": _stats(speed_residual),
        "speed_residual_mean_m_per_frame": speed_mean,
        "speed_residual_relative_to_direct_step": speed_ratio,
        "trajectory_drift": _stats(drift),
        "trajectory_drift_final_m": float(drift[-1]),
        "trajectory_drift_auc_m": float(drift.mean()),
        "trajectory_drift_slope_m_per_frame": _linear_slope(drift),
        "trajectory_drift_early_mean_m": drift_early,
        "trajectory_drift_late_mean_m": drift_late,
        "trajectory_drift_late_minus_early_m": drift_delta,
        "trajectory_drift_final_over_body_scale": float(drift[-1]) / body_scale,
        "trajectory_drift_auc_over_body_scale": float(drift.mean()) / body_scale,
        "curves": {
            "speed_residual_m_per_frame": speed_residual.tolist(),
            "trajectory_drift_m": drift.tolist(),
            "speed_joint_residual_m_per_frame": speed_joint_residual.tolist(),
            "trajectory_drift_joint_m": drift_joint.tolist(),
        },
        "integrated_positions": integrated.tolist(),
    }

    translation = motion[:, MOTION_LAYOUT.root_translation]
    translation_velocity = motion[:, MOTION_LAYOUT.root_translation_velocity]
    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    root_velocity = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation_velocity])
    if frames > 1:
        translation_residual = torch.linalg.vector_norm(
            (translation[1:] - translation[:-1]) - translation_velocity[:-1], dim=-1
        )
        root_step = root_velocity[:-1] @ root[:-1]
        root_step_target = root[1:]
        root_step_residual = _rotation_angle_degrees(root_step_target, root_step)
        integrated_root = [root[:1]]
        for index in range(frames - 1):
            integrated_root.append(root_velocity[index:index + 1] @ integrated_root[-1])
        integrated_root = torch.cat(integrated_root, dim=0)
        root_drift = _rotation_angle_degrees(root, integrated_root)
    else:
        translation_residual = torch.empty(0)
        root_step_residual = torch.empty(0)
        root_drift = torch.zeros(1)
    result["root_translation_speed_residual"] = _stats(translation_residual)
    result["root_rotation_speed_residual_degrees"] = _stats(root_step_residual)
    result["root_rotation_integrated_drift_degrees"] = _stats(root_drift)
    result["root_translation_speed_residual_mean"] = float(translation_residual.mean()) if translation_residual.numel() else 0.0
    result["root_rotation_speed_residual_degrees_mean"] = float(root_step_residual.mean()) if root_step_residual.numel() else 0.0
    result["root_rotation_integrated_drift_degrees_mean"] = float(root_drift.mean()) if root_drift.numel() else 0.0

    if model is not None:
        fk = _fk_joints(motion, model)
        fk_delta = fk - joints
        fk_joint_error = torch.linalg.vector_norm(fk_delta, dim=-1)
        direct_rel = joints[:, 1:] - joints[:, :1]
        fk_rel = fk[:, 1:] - fk[:, :1]
        fk_relative_joint_error = torch.linalg.vector_norm(fk_rel - direct_rel, dim=-1)
        fk_absolute = torch.linalg.vector_norm(fk_delta, dim=(1, 2))
        fk_relative = torch.linalg.vector_norm(fk_rel - direct_rel, dim=(1, 2))
        fk_mean_joint = fk_joint_error.mean(dim=-1)
        fk_relative_mean_joint = fk_relative_joint_error.mean(dim=-1)
        result["fk_absolute"] = _stats(fk_absolute)
        result["fk_absolute_mean_m"] = float(fk_mean_joint.mean())
        result["fk_absolute_over_body_scale"] = float(fk_mean_joint.mean()) / body_scale
        result["fk_relative_pelvis"] = _stats(fk_relative)
        result["fk_relative_pelvis_mean_m"] = float(fk_relative_mean_joint.mean())
        result["fk_relative_pelvis_over_body_scale"] = float(fk_relative_mean_joint.mean()) / body_scale
        result["curves"].update({
            "fk_absolute_m": fk_absolute.tolist(),
            "fk_relative_pelvis_m": fk_relative.tolist(),
            "fk_joint_error_m": fk_joint_error.tolist(),
            "fk_relative_pelvis_joint_error_m": fk_relative_joint_error.tolist(),
        })
    else:
        result["fk_absolute"] = None
        result["fk_relative_pelvis"] = None

    return result


def interpolate_curve(values: Sequence[float], length: int = 100) -> np.ndarray:
    """Resample a curve to normalized progress for variable-length references."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(length, dtype=np.float64)
    if values.size == 1:
        return np.full(length, values[0], dtype=np.float64)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, values)


def summarize_records(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(record[key]) for record in records]
    return {"count": len(values), **_stats(values)}


def bootstrap_cluster_stat(
    records: Sequence[dict[str, Any]],
    value_key: str,
    *,
    cluster_key: str = "sample_id",
    repetitions: int = 2000,
    seed: int = 20260823,
) -> dict[str, float]:
    """Cluster bootstrap for a scalar record metric."""

    if not records:
        return {"median": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    clusters: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        clusters.setdefault(str(record.get(cluster_key)), []).append(record)
    names = sorted(clusters)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(names), size=len(names))
        sample = [row for index in chosen for row in clusters[names[index]]]
        values.append(float(np.median([float(row[value_key]) for row in sample])))
    return {
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def bootstrap_cluster_curve(
    records: Sequence[dict[str, Any]],
    curve_key: str,
    *,
    cluster_key: str = "sample_id",
    length: int = 100,
    repetitions: int = 2000,
    seed: int = 20260823,
) -> dict[str, list[float]]:
    """Cluster bootstrap median curve, with normalized-time interpolation."""

    if not records:
        zeros = [0.0] * length
        return {"median": zeros, "ci95_low": zeros, "ci95_high": zeros}
    curves = {
        id(record): interpolate_curve(record["curves"][curve_key], length)
        for record in records
    }
    clusters: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        clusters.setdefault(str(record.get(cluster_key)), []).append(record)
    names = sorted(clusters)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(names), size=len(names))
        rows = [row for index in chosen for row in clusters[names[index]]]
        samples.append(np.median(np.stack([curves[id(row)] for row in rows]), axis=0))
    samples = np.stack(samples)
    return {
        "median": np.median(samples, axis=0).tolist(),
        "ci95_low": np.quantile(samples, 0.025, axis=0).tolist(),
        "ci95_high": np.quantile(samples, 0.975, axis=0).tolist(),
    }


__all__ = [
    "JOINT_NAMES",
    "compute_sequence_metrics",
    "interpolate_curve",
    "summarize_records",
    "bootstrap_cluster_stat",
    "bootstrap_cluster_curve",
]
