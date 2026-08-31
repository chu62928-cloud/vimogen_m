"""Unified, provenance preserving evaluation for root-forward v2.

The evaluator deliberately sits outside the optimizer.  It owns the common
M0/candidate decoding path, contact evidence policy, and three-valued gate
semantics so that a missing contact sample cannot silently become a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.pose_authority import authority_project, consistency_report
from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics


PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"
GATE_STATES = frozenset({PASS, FAIL, NOT_EVALUABLE})


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    threshold: Any
    observed: Any
    valid_count: int
    reason: str

    def __post_init__(self) -> None:
        if self.status not in GATE_STATES:
            raise ValueError(f"unknown gate status: {self.status}")
        if self.valid_count < 0:
            raise ValueError("valid_count must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "threshold": self.threshold,
            "observed": self.observed,
            "valid_count": self.valid_count,
            "reason": self.reason,
        }


def combine_gate_statuses(gates: Iterable[Gate | Mapping[str, Any]]) -> str:
    statuses = [gate.status if isinstance(gate, Gate) else str(gate["status"]) for gate in gates]
    if any(status == FAIL for status in statuses):
        return FAIL
    if any(status == NOT_EVALUABLE for status in statuses):
        return NOT_EVALUABLE
    return PASS


def threshold_gate(
    name: str,
    observed: float | None,
    threshold: float,
    *,
    valid_count: int,
    direction: str = "le",
    reason: str | None = None,
) -> Gate:
    """Build a scalar gate without treating absent data as success."""

    if observed is None or valid_count <= 0:
        return Gate(name, NOT_EVALUABLE, threshold, observed, max(valid_count, 0), reason or "no finite evidence")
    finite = bool(torch.isfinite(torch.tensor(float(observed))).item())
    if not finite:
        return Gate(name, FAIL, threshold, observed, valid_count, reason or "non-finite observed value")
    if direction == "le":
        passed = float(observed) <= float(threshold)
    elif direction == "ge":
        passed = float(observed) >= float(threshold)
    else:
        raise ValueError("direction must be 'le' or 'ge'")
    return Gate(name, PASS if passed else FAIL, threshold, float(observed), valid_count, reason or "threshold evaluated")


def contact_evidence(
    heel: torch.Tensor,
    toe: torch.Tensor,
    *,
    contact_height_m: float = 0.025,
    contact_speed_m_per_frame: float = 0.030,
    flat_gap_m: float = 0.020,
    min_height_frames: int = 3,
    min_sliding_pairs: int = 3,
) -> dict[str, Any]:
    """Return fixed-M0 contact evidence with explicit first-frame masking."""

    if heel.shape != toe.shape or heel.ndim != 2 or heel.shape[-1] != 3:
        raise ValueError("heel and toe must both have shape [T,3]")
    if heel.shape[0] < 1 or not torch.isfinite(heel).all() or not torch.isfinite(toe).all():
        raise ValueError("heel and toe must be non-empty and finite")
    sole = torch.minimum(heel[:, 2], toe[:, 2])
    floor = torch.quantile(sole, 0.05)
    centre = 0.5 * (heel + toe)
    speed = torch.full((heel.shape[0],), float("nan"), dtype=heel.dtype, device=heel.device)
    if heel.shape[0] > 1:
        speed[1:] = torch.linalg.vector_norm(centre[1:, :2] - centre[:-1, :2], dim=-1)
    contact = (sole <= floor + contact_height_m) & (speed <= contact_speed_m_per_frame)
    flat = contact & ((heel[:, 2] - toe[:, 2]).abs() <= flat_gap_m)
    pair_mask = contact[1:] & contact[:-1] if heel.shape[0] > 1 else contact[:0]
    flat_pair_mask = flat[1:] & flat[:-1] if heel.shape[0] > 1 else flat[:0]
    contact_count = int(contact.sum().item())
    flat_count = int(flat.sum().item())
    pair_count = int(pair_mask.sum().item())
    flat_pair_count = int(flat_pair_mask.sum().item())

    def stat(values: torch.Tensor, minimum: int) -> dict[str, float | None | int]:
        values = values[torch.isfinite(values)]
        if values.numel() < minimum:
            return {"mean": None, "p95": None, "count": int(values.numel()), "status": NOT_EVALUABLE}
        return {
            "mean": float(values.mean().item()),
            "p95": float(torch.quantile(values, 0.95).item()),
            "count": int(values.numel()),
            "status": PASS,
        }

    return {
        "floor_height_m": float(floor.item()),
        "contact_frames": contact_count,
        "flat_contact_frames": flat_count,
        "continuous_contact_pairs": pair_count,
        "continuous_flat_pairs": flat_pair_count,
        "height_evidence": stat((sole - floor).clamp_min(0.0)[contact], min_height_frames),
        "sliding_evidence_m_per_frame": stat(
            torch.linalg.vector_norm(centre[1:, :2] - centre[:-1, :2], dim=-1)[pair_mask],
            min_sliding_pairs,
        ),
        "valid_masks": {
            "contact": contact.detach().cpu().tolist(),
            "flat": flat.detach().cpu().tolist(),
            "continuous_contact_pair": pair_mask.detach().cpu().tolist(),
        },
        "thresholds": {
            "contact_height_m": contact_height_m,
            "contact_speed_m_per_frame": contact_speed_m_per_frame,
            "flat_gap_m": flat_gap_m,
            "minimum_height_frames": min_height_frames,
            "minimum_sliding_pairs": min_sliding_pairs,
            "first_frame_speed_is_valid": False,
        },
    }


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode("utf-8")
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def _finite_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    return {
        "finite": bool(finite.all().item()),
        "finite_count": int(finite.sum().item()),
        "total_count": int(flat.numel()),
        "max_abs": float(flat[finite].abs().max().item()) if finite.any() else None,
    }


def causal_audit(
    *,
    official_raw: torch.Tensor,
    official_pre_cast: torch.Tensor,
    consistent_m0: torch.Tensor,
    candidate_uncanonical: torch.Tensor,
    final_output: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Separate representation effects from the source-noise candidate."""

    values = {
        "official_raw_m0": official_raw,
        "official_pre_cast_m0": official_pre_cast,
        "consistent_m0": consistent_m0,
        "candidate_uncanonical": candidate_uncanonical,
        "final_output": final_output,
    }
    shape = tuple(official_raw.shape)
    if any(tuple(value.shape) != shape for value in values.values()):
        raise ValueError("all causal-audit tensors must have equal shape")
    if valid_mask.shape != shape[:2]:
        raise ValueError("valid_mask must match motion batch/time dimensions")
    direct = {
        "body_pose": MOTION_LAYOUT.body_pose,
        "root_rotation": MOTION_LAYOUT.root_rotation,
        "root_translation": MOTION_LAYOUT.root_translation,
    }

    def diff(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
        delta = (left.float() - right.float())[valid_mask]
        return {
            "rms": float(torch.sqrt(delta.square().mean()).item()) if delta.numel() else None,
            "max_abs": float(delta.abs().max().item()) if delta.numel() else None,
            "finite": bool(torch.isfinite(delta).all().item()),
        }

    return {
        "role": "evaluation_only",
        "counts_as_v2_success": False,
        "objects": {
            name: {"shape": list(value.shape), "sha256": tensor_sha256(value), "finite": _finite_summary(value)}
            for name, value in values.items()
        },
        "pairwise_representation_deltas": {
            "official_raw_to_official_pre_cast": diff(official_raw, official_pre_cast),
            "official_pre_cast_to_consistent_m0": diff(official_pre_cast, consistent_m0),
            "candidate_uncanonical_to_final_output": diff(candidate_uncanonical, final_output),
            "consistent_m0_to_final_output": diff(consistent_m0, final_output),
        },
        "direct_channel_deltas": {
            name: diff(official_pre_cast[..., span], consistent_m0[..., span])
            for name, span in direct.items()
        },
        "idempotence": diff(
            consistent_m0,
            authority_project(consistent_m0, valid_mask=valid_mask, output_dtype=torch.float32).motion,
        ),
        "consistency": {
            "consistent_m0": consistency_report(consistent_m0.float(), valid_mask),
            "final_output": consistency_report(final_output.float(), valid_mask),
        },
    }


def build_v2_gates(metrics: Mapping[str, Any]) -> list[Gate]:
    rows = metrics.get("per_sample", [])
    if not rows:
        return [Gate("root_control", NOT_EVALUABLE, {"pitch_mae_deg": 1.0, "forward_p95_deg": 2.0}, None, 0, "no per-sample metrics")]
    row = rows[0]
    gates = [
        threshold_gate("root_pitch_mae", row.get("mean_absolute_error_deg"), 1.0, valid_count=1),
        threshold_gate("root_forward_p95", row.get("forward_vector_error_p95_deg"), 2.0, valid_count=1),
        threshold_gate("root_heading_p95", row.get("horizontal_heading_drift_p95_deg"), 2.0, valid_count=1),
        Gate("dose_sign", PASS if row.get("dose_sign_correct") is True else FAIL, True, row.get("dose_sign_correct"), 1, "signed dose check"),
    ]
    tail = metrics.get("tail_safety", {}).get("per_sample", [])
    if tail:
        item = tail[0]
        count = int(item.get("tail_pair_count", 0))
        gates.extend([
            threshold_gate("tail_extra_so3_jump", item.get("tail_extra_so3_jump_max_deg"), 2.0, valid_count=count),
            threshold_gate("tail_extra_pitch_step", item.get("tail_extra_pitch_step_max_deg"), 2.0, valid_count=count),
        ])
    else:
        gates.append(Gate("tail_safety", NOT_EVALUABLE, 2.0, None, 0, "no tail evidence"))
    whole = metrics.get("whole_body", {})
    trunk = whole.get("trunk_change_deg", {})
    gates.append(threshold_gate("trunk_direction_p95", trunk.get("p95"), 2.0, valid_count=1))
    q_rigid = whole.get("q_rigid")
    gates.append(threshold_gate("q_rigid", q_rigid, 0.2, valid_count=1))
    return gates


def write_json_strict(path: str | Path, value: Any) -> None:
    """Write JSON while rejecting NaN/Infinity at the artifact boundary."""

    def clean(item: Any) -> Any:
        if isinstance(item, float) and (item != item or item in (float("inf"), float("-inf"))):
            raise ValueError(f"non-finite value in JSON artifact {path}")
        if isinstance(item, Mapping):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item

    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


__all__ = [
    "PASS", "FAIL", "NOT_EVALUABLE", "Gate", "GATE_STATES", "combine_gate_statuses",
    "threshold_gate", "contact_evidence", "tensor_sha256", "causal_audit",
    "build_v2_gates", "write_json_strict",
]
