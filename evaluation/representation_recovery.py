"""Controlled corruption and paired recovery metrics for the 276-D protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from motion_rep.consistent_finalizer import finalize_consistent_motion_tensor
from motion_rep.phase1 import MOTION_LAYOUT, validate_motion_tensor
from motion_rep.reconciliation import ReconciliationConfig, reconcile_motion_tensor
from motion_rep.unified_finalizer import finalize_motion_tensor


@dataclass(frozen=True)
class CorruptionConfig:
    """Fixed corruption scales calibrated on the development split only."""

    position_noise_m: float = 0.005
    position_drift_m: float = 0.01
    velocity_noise_m: float = 0.005
    velocity_bias_m: float = 0.002

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _seed_for(sample_key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{sample_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _normalised_random_walk(shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    values = torch.randn(shape, generator=generator, device=device)
    values = torch.cumsum(values, dim=0)
    scale = values.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
    return values / scale


def corrupt_motion(
    clean: torch.Tensor,
    *,
    sample_key: str,
    config: CorruptionConfig,
    seed: int = 20260824,
) -> torch.Tensor:
    """Inject deterministic position drift/noise and velocity errors."""

    validate_motion_tensor(clean)
    if clean.ndim != 2:
        raise ValueError("corrupt_motion expects one [T,276] motion")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_for(sample_key, seed))
    result = clean.float().clone()
    frames = clean.shape[0]
    joints = result[:, MOTION_LAYOUT.joints].reshape(frames, 22, 3)
    joints_velocity = result[:, MOTION_LAYOUT.joints_velocity].reshape(frames, 22, 3)
    translation = result[:, MOTION_LAYOUT.root_translation]
    translation_velocity = result[:, MOTION_LAYOUT.root_translation_velocity]
    for direct, velocity in ((joints, joints_velocity), (translation, translation_velocity)):
        direct_noise = torch.randn(direct.shape, generator=generator, device="cpu")
        drift = _normalised_random_walk(tuple(direct.shape), generator, torch.device("cpu"))
        velocity_noise = torch.randn(velocity.shape, generator=generator, device="cpu")
        velocity_bias = _normalised_random_walk(tuple(velocity.shape), generator, torch.device("cpu"))
        direct.add_(config.position_noise_m * direct_noise + config.position_drift_m * drift)
        velocity.add_(config.velocity_noise_m * velocity_noise + config.velocity_bias_m * velocity_bias)
    result[:, MOTION_LAYOUT.joints] = joints.reshape(frames, 66)
    result[:, MOTION_LAYOUT.joints_velocity] = joints_velocity.reshape(frames, 66)
    result[:, MOTION_LAYOUT.root_translation] = translation
    result[:, MOTION_LAYOUT.root_translation_velocity] = translation_velocity
    return result


def calibrate_corruption(
    motions: Iterable[torch.Tensor], *,
    sample_count: int = 512,
    fraction_of_typical_step: float = 0.05,
) -> CorruptionConfig:
    """Estimate fixed perturbation scales from development motions only."""

    direct_steps: list[float] = []
    velocity_steps: list[float] = []
    for index, motion in enumerate(motions):
        if index >= sample_count:
            break
        validate_motion_tensor(motion)
        joints = motion[:, MOTION_LAYOUT.joints].reshape(motion.shape[0], 22, 3).float()
        velocity = motion[:, MOTION_LAYOUT.joints_velocity].reshape(motion.shape[0], 22, 3).float()
        if joints.shape[0] > 1:
            direct_steps.extend(torch.linalg.vector_norm(joints[1:] - joints[:-1], dim=-1).flatten().tolist())
        velocity_steps.extend(torch.linalg.vector_norm(velocity, dim=-1).flatten().tolist())
    if not direct_steps or not velocity_steps:
        raise ValueError("development calibration requires non-empty motions with at least two frames")
    typical_direct = float(np.median(direct_steps))
    typical_velocity = float(np.median(velocity_steps))
    base = max(typical_direct, typical_velocity, 1e-4)
    noise = max(1e-4, fraction_of_typical_step * base)
    return CorruptionConfig(
        position_noise_m=noise,
        position_drift_m=2.0 * noise,
        velocity_noise_m=noise,
        velocity_bias_m=0.5 * noise,
    )


def _direct_positions(motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    joints = motion[:, MOTION_LAYOUT.joints].reshape(motion.shape[0], 22, 3).float()
    translation = motion[:, MOTION_LAYOUT.root_translation].float()
    return joints, translation


def _smoothness(joints: torch.Tensor, translation: torch.Tensor) -> float:
    values = [joints, translation[:, None, :]]
    terms = []
    for value in values:
        if value.shape[0] >= 3:
            terms.append(torch.linalg.vector_norm(value[2:] - 2 * value[1:-1] + value[:-2], dim=-1).mean())
    return float(torch.stack(terms).mean()) if terms else 0.0


def _position_error(reference: tuple[torch.Tensor, torch.Tensor], candidate: torch.Tensor) -> dict[str, float]:
    ref_joints, ref_translation = reference
    joints, translation = _direct_positions(candidate)
    joint_error = torch.linalg.vector_norm(joints - ref_joints, dim=-1)
    translation_error = torch.linalg.vector_norm(translation - ref_translation, dim=-1)
    return {
        "joint_position_rmse_m": float(torch.sqrt((joint_error.square()).mean())),
        "joint_position_median_m": float(joint_error.median()),
        "root_translation_rmse_m": float(torch.sqrt((translation_error.square()).mean())),
        "root_translation_median_m": float(translation_error.median()),
        "smoothness_second_difference": _smoothness(joints, translation),
    }


def evaluate_one(
    clean: torch.Tensor,
    *,
    sample_key: str,
    corruption: CorruptionConfig,
    reconciliation: ReconciliationConfig | None = None,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Evaluate direct, velocity and reconciled outputs for one motion."""

    validate_motion_tensor(clean)
    corrupted = corrupt_motion(clean, sample_key=sample_key, config=corruption, seed=seed)
    reference = _direct_positions(clean.float())
    direct = finalize_consistent_motion_tensor(corrupted).motion.float()
    velocity = finalize_motion_tensor(corrupted).motion.float()
    fused = reconcile_motion_tensor(
        corrupted,
        config=reconciliation or ReconciliationConfig(),
    ).motion.float()
    return {
        "sample_key": sample_key,
        "corruption": corruption.as_dict(),
        "methods": {
            "absolute_position": _position_error(reference, direct),
            "velocity_integral": _position_error(reference, velocity),
            "reconciled": _position_error(reference, fused),
        },
    }


def _bootstrap_median(values: np.ndarray, *, repetitions: int, seed: int) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty value array")
    rng = np.random.default_rng(seed)
    medians = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = values[rng.integers(0, values.size, size=values.size)]
        medians[index] = np.median(sample)
    return {
        "median": float(np.median(medians)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
    }


def _bootstrap_paired_effect(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(
        [
            float(row["methods"][candidate][metric]) - float(row["methods"][baseline][metric])
            for row in records
        ],
        dtype=np.float64,
    )
    return _bootstrap_median(values, repetitions=repetitions, seed=seed)


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260824,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty record list")
    methods = ("absolute_position", "velocity_integral", "reconciled")
    metrics = ("joint_position_rmse_m", "root_translation_rmse_m", "smoothness_second_difference")
    summary: dict[str, Any] = {"record_count": len(records), "methods": {}}

    def stable_offset(method: str, metric: str) -> int:
        digest = hashlib.sha256(f"{method}\0{metric}".encode("utf-8")).digest()
        return int.from_bytes(digest[:2], "little") % 997

    for method in methods:
        summary["methods"][method] = {}
        for metric in metrics:
            values = np.asarray([float(row["methods"][method][metric]) for row in records], dtype=np.float64)
            summary["methods"][method][metric] = {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.quantile(values, 0.95)),
                "bootstrap_median_ci95": _bootstrap_median(
                    values, repetitions=bootstrap_repetitions, seed=bootstrap_seed + stable_offset(method, metric)
                ),
            }
    summary["paired_bootstrap_effects"] = {}
    for metric in ("joint_position_rmse_m", "root_translation_rmse_m"):
        summary["paired_bootstrap_effects"][metric] = {
            "reconciled_minus_absolute_position": _bootstrap_paired_effect(
                records,
                candidate="reconciled",
                baseline="absolute_position",
                metric=metric,
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed + 11,
            ),
            "reconciled_minus_velocity_integral": _bootstrap_paired_effect(
                records,
                candidate="reconciled",
                baseline="velocity_integral",
                metric=metric,
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed + 29,
            ),
        }
    summary["reconciled_beats_absolute_position_rmse"] = summary["methods"]["reconciled"]["joint_position_rmse_m"]["median"] < summary["methods"]["absolute_position"]["joint_position_rmse_m"]["median"]
    summary["reconciled_beats_velocity_integral_rmse"] = summary["methods"]["reconciled"]["joint_position_rmse_m"]["median"] < summary["methods"]["velocity_integral"]["joint_position_rmse_m"]["median"]
    return summary


__all__ = [
    "CorruptionConfig",
    "calibrate_corruption",
    "corrupt_motion",
    "evaluate_one",
    "summarize",
]
