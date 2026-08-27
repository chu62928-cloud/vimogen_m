"""Paired M1-versus-M0 residual metrics.

The M0 representation already contains non-zero position/velocity residuals.
This module therefore evaluates *extra* residual introduced by a method on the
same text, seed, and initial-noise trajectory instead of treating zero as the
baseline target.
"""

from __future__ import annotations

from typing import Mapping, Sequence


CHANNEL_KEYS = (
    "joint_position_velocity_median_m",
    "root_translation_velocity_median_m",
    "root_rotation_velocity_median_degrees",
)

# Frozen engineering tolerances for the first paired M1 gate.  They are not
# fitted from a holdout result.
DEFAULT_EXCESS_LIMITS = {
    "joint_position_velocity_median_m": 0.001,
    "root_translation_velocity_median_m": 0.001,
    "root_rotation_velocity_median_degrees": 0.1,
}


def paired_excess(
    m0: Mapping[str, float],
    method: Mapping[str, float],
    *,
    limits: Mapping[str, float] = DEFAULT_EXCESS_LIMITS,
) -> dict[str, object]:
    """Return per-channel M1-minus-M0 residual and a fixed non-degradation flag."""

    missing = [key for key in CHANNEL_KEYS if key not in m0 or key not in method]
    if missing:
        raise KeyError(f"missing paired residual channels: {missing}")
    missing_limits = [key for key in CHANNEL_KEYS if key not in limits]
    if missing_limits:
        raise KeyError(f"missing residual limits: {missing_limits}")
    excess = {key: float(method[key]) - float(m0[key]) for key in CHANNEL_KEYS}
    channel_pass = {
        key: excess[key] <= float(limits[key]) for key in CHANNEL_KEYS
    }
    return {
        "m0": {key: float(m0[key]) for key in CHANNEL_KEYS},
        "method": {key: float(method[key]) for key in CHANNEL_KEYS},
        "excess": excess,
        "limits": {key: float(limits[key]) for key in CHANNEL_KEYS},
        "channel_pass": channel_pass,
        "non_degradation_pass": bool(all(channel_pass.values())),
    }


def summarize_paired(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize already-paired per-sample records without re-pairing them."""

    if not records:
        raise ValueError("at least one paired record is required")
    excess_records = [
        paired_excess(record["m0"], record["method"], limits=record.get("limits", DEFAULT_EXCESS_LIMITS))
        for record in records
    ]
    return {
        "count": len(excess_records),
        "all_samples_pass": bool(all(item["non_degradation_pass"] for item in excess_records)),
        "records": excess_records,
    }
