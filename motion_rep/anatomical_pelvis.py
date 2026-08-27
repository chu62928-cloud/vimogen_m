"""Unified anatomical pelvis geometry and anti-cheat audit metrics (v4).

The v3 angle used a root coordinate axis as a proxy for the anatomical
anterior direction.  This module makes the proxy explicit: a frozen,
project-specific calibration stores four virtual marker groups on the neutral
SMPL-X mesh (LASI/RASI/LPSI/RPSI).  Runtime code only consumes their calibrated
local coordinates, so rendering, guidance, and evaluation share exactly one
geometric definition.

The module intentionally does not claim that these markers are official
SMPL-X vertex landmarks.  ``PelvisCalibration`` records the template hash and
the reviewed vertex groups used by the project.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


V4_PROTOCOL = "vimogen_absolute_mean_pelvis_v4_anatomical_local"
MARKER_NAMES = ("LASI", "RASI", "LPSI", "RPSI")
JOINT_INDEX = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
               "left_knee": 4, "right_knee": 5, "spine2": 6,
               "spine3": 9, "neck": 12}


@dataclass(frozen=True)
class PelvisCalibration:
    """Frozen neutral-template calibration for the four virtual markers."""

    template_sha256: str
    model_path: str
    marker_vertex_groups: Mapping[str, tuple[int, ...]]
    marker_local_points: Mapping[str, tuple[float, float, float]]
    root_rest_joint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_axis: int = 2
    local_forward_description: str = "A-P (posterior to anterior)"
    reviewed_image: str | None = None
    calibration_version: str = "v4-anatomical-local-1"

    def __post_init__(self) -> None:
        if len(self.template_sha256) != 64:
            raise ValueError("template_sha256 must be a 64-character SHA256")
        missing = set(MARKER_NAMES) - set(self.marker_local_points)
        if missing:
            raise ValueError(f"missing calibrated marker(s): {sorted(missing)}")
        missing_groups = set(MARKER_NAMES) - set(self.marker_vertex_groups)
        if missing_groups:
            raise ValueError(f"missing marker vertex group(s): {sorted(missing_groups)}")
        for name in MARKER_NAMES:
            point = tuple(float(v) for v in self.marker_local_points[name])
            if len(point) != 3 or not all(math.isfinite(v) for v in point):
                raise ValueError(f"marker {name} local point must contain 3 finite values")
            group = tuple(int(v) for v in self.marker_vertex_groups[name])
            if not group or any(v < 0 for v in group):
                raise ValueError(f"marker {name} vertex group must be non-empty and non-negative")
        if len(self.root_rest_joint) != 3 or not all(math.isfinite(float(v)) for v in self.root_rest_joint):
            raise ValueError("root_rest_joint must contain 3 finite values")
        if self.up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be 0, 1, or 2")

    @property
    def local_points(self) -> torch.Tensor:
        """Return marker means in LASI, RASI, LPSI, RPSI order."""

        return torch.tensor(
            [self.marker_local_points[name] for name in MARKER_NAMES],
            dtype=torch.float32,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol": V4_PROTOCOL,
            "calibration_version": self.calibration_version,
            "template_sha256": self.template_sha256,
            "model_path": self.model_path,
            "marker_vertex_groups": {
                name: list(self.marker_vertex_groups[name]) for name in MARKER_NAMES
            },
            "marker_local_points": {
                name: list(self.marker_local_points[name]) for name in MARKER_NAMES
            },
            "root_rest_joint": list(self.root_rest_joint),
            "up_axis": self.up_axis,
            "local_forward_description": self.local_forward_description,
            "reviewed_image": self.reviewed_image,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PelvisCalibration":
        groups = value.get("marker_vertex_groups", {})
        points = value.get("marker_local_points", {})
        return cls(
            template_sha256=str(value["template_sha256"]),
            model_path=str(value.get("model_path", "")),
            marker_vertex_groups={name: tuple(int(v) for v in groups[name]) for name in MARKER_NAMES},
            marker_local_points={name: tuple(float(v) for v in points[name]) for name in MARKER_NAMES},
            root_rest_joint=tuple(float(v) for v in value.get("root_rest_joint", (0.0, 0.0, 0.0))),
            up_axis=int(value.get("up_axis", 2)),
            local_forward_description=str(value.get("local_forward_description", "A-P (posterior to anterior)")),
            reviewed_image=(None if value.get("reviewed_image") is None else str(value["reviewed_image"])),
            calibration_version=str(value.get("calibration_version", "v4-anatomical-local-1")),
        )


def load_pelvis_calibration(path: str | Path) -> PelvisCalibration:
    """Load and validate a frozen calibration JSON."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("protocol") not in (None, V4_PROTOCOL):
        raise ValueError(f"calibration protocol is not {V4_PROTOCOL}: {value.get('protocol')!r}")
    return PelvisCalibration.from_mapping(value)


def calibration_sha256(calibration: PelvisCalibration) -> str:
    """Hash the canonical JSON representation used in protocol records."""

    payload = json.dumps(calibration.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_rotation(root_rotation: torch.Tensor) -> None:
    if not isinstance(root_rotation, torch.Tensor) or root_rotation.ndim < 2:
        raise ValueError("root_rotation must be a floating tensor [...,3,3]")
    if root_rotation.shape[-2:] != (3, 3) or not torch.is_floating_point(root_rotation):
        raise ValueError("root_rotation must be a floating tensor [...,3,3]")
    if not torch.isfinite(root_rotation).all():
        raise ValueError("root_rotation contains non-finite values")


def _unit(value: torch.Tensor, fallback: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(value, dim=-1)
    normalized = value / norm.unsqueeze(-1).clamp_min(eps)
    fallback = fallback.to(device=value.device, dtype=value.dtype).expand_as(normalized)
    return torch.where((norm > eps).unsqueeze(-1), normalized, fallback), norm


@dataclass(frozen=True)
class PelvisGeometry:
    anterior_point: torch.Tensor
    posterior_point: torch.Tensor
    anterior_axis: torch.Tensor
    heading: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    angle_radians: torch.Tensor
    horizontal_norm: torch.Tensor
    valid: torch.Tensor

    @property
    def angle_degrees(self) -> torch.Tensor:
        return self.angle_radians * (180.0 / math.pi)


def anatomical_pelvis_geometry(
    root_rotation: torch.Tensor,
    calibration: PelvisCalibration,
    root_translation: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> PelvisGeometry:
    """Return calibrated P/A endpoints, local frame, and signed pelvic angle.

    ``theta = atan2(-v·up, v·heading)`` for ``v=A-P``.  Consequently a
    horizontal anterior side is zero and anterior-side-down is positive.
    """

    _validate_rotation(root_rotation)
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number")
    points = calibration.local_points.to(device=root_rotation.device, dtype=root_rotation.dtype)
    root_rest = torch.tensor(calibration.root_rest_joint, dtype=root_rotation.dtype, device=root_rotation.device)
    # Points are stored as offsets from the SMPL-X pelvis rest joint.  The
    # global orientation rotates about that joint, then root translation is
    # added, matching the authoritative FK convention.
    world = torch.matmul(points, root_rotation.transpose(-1, -2))
    world = world + root_rest
    # The row-vector form above is equivalent to R @ p for column vectors.
    anterior = 0.5 * (world[..., 0, :] + world[..., 1, :])
    posterior = 0.5 * (world[..., 2, :] + world[..., 3, :])
    if root_translation is not None:
        if root_translation.shape != root_rotation.shape[:-2] + (3,):
            raise ValueError("root_translation must have shape root_rotation.shape[:-2]+(3,)")
        anterior = anterior + root_translation
        posterior = posterior + root_translation
    vector = anterior - posterior
    up = torch.zeros_like(vector)
    up[..., calibration.up_axis] = 1.0
    horizontal = vector - (vector * up).sum(-1, keepdim=True) * up
    reference = torch.zeros_like(vector)
    # The project uses z-up and +y as the neutral heading.  For other up axes
    # choose a deterministic perpendicular reference for synthetic tests.
    reference_index = 1 if calibration.up_axis != 1 else 0
    reference[..., reference_index] = 1.0
    heading, horizontal_norm = _unit(horizontal, reference, eps)
    right, _ = _unit(torch.cross(heading, up, dim=-1), reference.new_tensor([1.0, 0.0, 0.0]), eps)
    vertical = (vector * up).sum(-1)
    along_heading = (vector * heading).sum(-1)
    angle = torch.atan2(-vertical, along_heading)
    valid = horizontal_norm > eps
    return PelvisGeometry(
        anterior_point=anterior,
        posterior_point=posterior,
        anterior_axis=vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps),
        heading=heading,
        right=right,
        up=up,
        angle_radians=angle,
        horizontal_norm=horizontal_norm,
        valid=valid,
    )


def anatomical_pelvis_angle_degrees(
    root_rotation: torch.Tensor,
    calibration: PelvisCalibration,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    return anatomical_pelvis_geometry(root_rotation, calibration, eps=eps).angle_degrees


def apply_anatomical_pelvis_delta(
    root_rotation: torch.Tensor,
    delta_degrees: float | torch.Tensor,
    calibration: PelvisCalibration,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply a left-multiplied correction whose positive sign is anterior-down."""

    geometry = anatomical_pelvis_geometry(root_rotation, calibration, eps=eps)
    delta = torch.as_tensor(delta_degrees, dtype=root_rotation.dtype, device=root_rotation.device)
    try:
        delta = torch.broadcast_to(delta, root_rotation.shape[:-2])
    except RuntimeError as error:
        raise ValueError("delta_degrees must be scalar or broadcastable to root_rotation leading shape") from error
    # Rodrigues about -right maps neutral +heading towards -up.
    axis = -geometry.right
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=root_rotation.dtype, device=root_rotation.device).expand_as(skew)
    radians = delta * (math.pi / 180.0)
    angle = radians.unsqueeze(-1).unsqueeze(-1)
    correction = eye + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)
    return correction @ root_rotation


def _segment_angle(
    segment: torch.Tensor,
    heading: torch.Tensor,
    up: torch.Tensor,
    *,
    downward: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(segment, dim=-1)
    if downward:
        vertical = -(segment * up).sum(-1)
    else:
        vertical = (segment * up).sum(-1)
    forward = (segment * heading).sum(-1)
    return torch.atan2(forward, vertical) * (180.0 / math.pi), norm > eps


def trunk_and_thigh_angles(
    joints: torch.Tensor,
    pelvis: PelvisGeometry,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Measure torso (spine1→neck) and bilateral thigh sagittal angles."""

    if joints.shape[-2:] != (22, 3):
        raise ValueError("joints must have shape [...,22,3]")
    heading, up = pelvis.heading, pelvis.up
    trunk, trunk_valid = _segment_angle(
        joints[..., JOINT_INDEX["neck"], :] - joints[..., JOINT_INDEX["spine1"], :],
        heading, up, downward=False, eps=eps,
    )
    left, left_valid = _segment_angle(
        joints[..., JOINT_INDEX["left_knee"], :] - joints[..., JOINT_INDEX["left_hip"], :],
        heading, up, downward=True, eps=eps,
    )
    right, right_valid = _segment_angle(
        joints[..., JOINT_INDEX["right_knee"], :] - joints[..., JOINT_INDEX["right_hip"], :],
        heading, up, downward=True, eps=eps,
    )
    return {
        "trunk_deg": trunk,
        "thigh_left_deg": left,
        "thigh_right_deg": right,
        "trunk_valid": trunk_valid,
        "thigh_left_valid": left_valid,
        "thigh_right_valid": right_valid,
    }


def _wrap_degrees(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + 180.0, 360.0) - 180.0


def anti_cheat_penalty(
    delta_trunk_deg: torch.Tensor,
    delta_thigh_left_deg: torch.Tensor,
    delta_thigh_right_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    soft_limit_deg: float = 2.0,
) -> torch.Tensor:
    """Return the fixed-coefficient soft anti-cheat loss in degree² units."""

    if valid_mask.shape != delta_trunk_deg.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool and match angle delta shape")
    limit = torch.as_tensor(float(soft_limit_deg), dtype=delta_trunk_deg.dtype, device=delta_trunk_deg.device)
    hinge = lambda value: torch.relu(value.abs() - limit).square()
    values = (hinge(delta_trunk_deg) + hinge(delta_thigh_left_deg) + hinge(delta_thigh_right_deg)) / 3.0
    mask = valid_mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def local_dominance_penalty(
    delta_pelvis_deg: torch.Tensor,
    delta_trunk_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    minimum_share: float = 0.5,
    low_signal_deg: float = 0.5,
) -> torch.Tensor:
    """Penalize pelvis changes that are not locally dominant over the trunk.

    This is a fixed, dimensionless safety term rather than a tunable gate.  It
    is zero for low-signal pelvis changes and otherwise requires the signed
    local change ``delta_pelvis - delta_trunk`` to explain at least half of the
    absolute pelvis change in the same direction.
    """

    if delta_pelvis_deg.shape != delta_trunk_deg.shape or valid_mask.shape != delta_pelvis_deg.shape:
        raise ValueError("pelvis/trunk deltas and valid_mask must have the same shape")
    signal = delta_pelvis_deg.abs() >= float(low_signal_deg)
    signed_local = torch.sign(delta_pelvis_deg.detach()) * (delta_pelvis_deg - delta_trunk_deg)
    required = float(minimum_share) * delta_pelvis_deg.abs()
    values = torch.relu(required - signed_local).square() * signal.to(delta_pelvis_deg.dtype)
    mask = valid_mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def anti_cheat_metrics(
    pelvis_m0_deg: torch.Tensor,
    pelvis_g_deg: torch.Tensor,
    trunk_m0_deg: torch.Tensor,
    trunk_g_deg: torch.Tensor,
    thigh_left_m0_deg: torch.Tensor,
    thigh_left_g_deg: torch.Tensor,
    thigh_right_m0_deg: torch.Tensor,
    thigh_right_g_deg: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    low_signal_deg: float = 0.5,
    ratio_floor_deg: float = 0.25,
) -> dict[str, torch.Tensor | float | bool]:
    """Compute direct local-change and ratio audits for one action."""

    if valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool")
    dp = _wrap_degrees(pelvis_g_deg - pelvis_m0_deg)
    dt = _wrap_degrees(trunk_g_deg - trunk_m0_deg)
    dl = _wrap_degrees(thigh_left_g_deg - thigh_left_m0_deg)
    dr = _wrap_degrees(thigh_right_g_deg - thigh_right_m0_deg)
    local = dp - dt
    signal = valid_mask & (dp.abs() >= float(low_signal_deg))
    ratio = dp.abs() / dt.abs().clamp_min(float(ratio_floor_deg))
    ratio_valid = valid_mask & (dp.abs() >= float(low_signal_deg))
    values = lambda x, mask=valid_mask: x[mask]
    local_mean = (local[valid_mask].mean() if bool(valid_mask.any()) else local.new_tensor(0.0))
    pelvis_mean = (dp[valid_mask].mean() if bool(valid_mask.any()) else dp.new_tensor(0.0))
    trunk_mean = (dt[valid_mask].mean() if bool(valid_mask.any()) else dt.new_tensor(0.0))
    # The share is a magnitude; direction is reported independently via
    # ``local_change_same_sign``.
    share = local_mean.abs() / pelvis_mean.abs().clamp_min(1e-8)
    share_sign = (local_mean * pelvis_mean >= 0) if abs(float(pelvis_mean.detach().cpu())) >= low_signal_deg else torch.tensor(False, device=local.device)
    def q(x: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
        selected = x[mask].abs()
        return torch.quantile(selected, p) if selected.numel() else x.new_tensor(0.0)
    return {
        "delta_pelvis_deg": dp,
        "delta_trunk_deg": dt,
        "delta_thigh_left_deg": dl,
        "delta_thigh_right_deg": dr,
        "delta_pelvis_trunk_deg": local,
        "ratio_t": ratio,
        "ratio_valid": ratio_valid,
        "low_signal_frame_rate": (~signal & valid_mask).to(dp.dtype).sum() / valid_mask.to(dp.dtype).sum().clamp_min(1.0),
        "delta_pelvis_mean_deg": pelvis_mean,
        "delta_trunk_mean_deg": trunk_mean,
        "delta_local_mean_deg": local_mean,
        "local_change_share": share,
        "local_change_same_sign": bool(share_sign.detach().cpu()),
        "local_change_low_signal": bool(abs(float(pelvis_mean.detach().cpu())) < low_signal_deg),
        "trunk_abs_p95_deg": q(dt, valid_mask, 0.95),
        "thigh_left_abs_p95_deg": q(dl, valid_mask, 0.95),
        "thigh_right_abs_p95_deg": q(dr, valid_mask, 0.95),
        "ratio_p05": q(ratio, ratio_valid, 0.05),
        "ratio_median": q(ratio, ratio_valid, 0.50),
        "ratio_p95": q(ratio, ratio_valid, 0.95),
        "ratio_action": dp[valid_mask].abs().mean() / dt[valid_mask].abs().mean().clamp_min(float(ratio_floor_deg)) if bool(valid_mask.any()) else dp.new_tensor(0.0),
    }


__all__ = [
    "JOINT_INDEX", "MARKER_NAMES", "V4_PROTOCOL", "PelvisCalibration", "PelvisGeometry",
    "anatomical_pelvis_angle_degrees", "anatomical_pelvis_geometry", "anti_cheat_metrics",
    "anti_cheat_penalty", "apply_anatomical_pelvis_delta", "calibration_sha256",
    "load_pelvis_calibration", "local_dominance_penalty", "trunk_and_thigh_angles",
]
