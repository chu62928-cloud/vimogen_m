"""Small, explicit motion-representation interfaces for phase 1.

The legacy ViMoGen path remains unchanged.  This module provides a tested
boundary for future editing/finalisation code so that the 276 channels,
forward-difference convention, masks, and SO(3) edits cannot be inferred from
magic offsets in an experiment script.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .rotation_transform import axis_angle_to_mat3x3


@dataclass(frozen=True)
class Motion276Layout:
    """The verified ViMoGen 276-dimensional channel layout."""

    joint_count: int = 22
    body_pose: slice = slice(0, 126)
    joints: slice = slice(126, 192)
    joints_velocity: slice = slice(192, 258)
    root_rotation: slice = slice(258, 264)
    root_rotation_velocity: slice = slice(264, 270)
    root_translation: slice = slice(270, 273)
    root_translation_velocity: slice = slice(273, 276)

    @property
    def total_dim(self) -> int:
        return 276

    def assert_valid(self) -> None:
        expected = (
            self.body_pose,
            self.joints,
            self.joints_velocity,
            self.root_rotation,
            self.root_rotation_velocity,
            self.root_translation,
            self.root_translation_velocity,
        )
        cursor = 0
        for section in expected:
            if section.start != cursor or section.stop <= section.start:
                raise AssertionError(f"invalid 276 layout at {section}")
            cursor = section.stop
        if cursor != self.total_dim:
            raise AssertionError(f"layout ends at {cursor}, expected {self.total_dim}")


MOTION_LAYOUT = Motion276Layout()
MOTION_LAYOUT.assert_valid()

# SMPL-X's first 22 joints are the body-only subset used by ViMoGen.  Keeping
# the order here prevents a later FK/editing module from silently swapping left
# and right hips or shoulders.
SMPLX_22_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
SMPLX_22_JOINT_INDEX = {name: index for index, name in enumerate(SMPLX_22_JOINT_NAMES)}
if len(SMPLX_22_JOINT_NAMES) != MOTION_LAYOUT.joint_count:
    raise AssertionError("SMPL-X body joint table must contain 22 entries")

# Channel slices for the phase-1 active sets.  These are Rot6D slices in the
# first 126 body-pose channels, not world-position slices.
ACTIVE_ROTATION_SLICES = {
    "root_pelvis": slice(258, 264),
    "left_hip": slice(0, 6),
    "right_hip": slice(6, 12),
    "spine1": slice(12, 18),
    "spine2": slice(30, 36),
    "spine3": slice(48, 54),
}


def validate_motion_tensor(
    motion: torch.Tensor, *, expected_dtype: torch.dtype | None = None
) -> None:
    """Validate the public ``[..., T, 276]`` motion boundary."""

    if not isinstance(motion, torch.Tensor):
        raise TypeError(f"motion must be a torch.Tensor, got {type(motion)!r}")
    if motion.ndim < 2 or motion.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(
            f"motion must have shape [..., T, 276], got {tuple(motion.shape)}"
        )
    if expected_dtype is not None and motion.dtype != expected_dtype:
        raise TypeError(f"motion dtype must be {expected_dtype}, got {motion.dtype}")
    if not torch.isfinite(motion).all():
        raise ValueError("motion contains non-finite values")


def encode_rot6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Encode the first two matrix columns in ViMoGen's interleaved layout."""

    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in [3, 3], got {tuple(rotation_matrix.shape)}")
    return rotation_matrix[..., :, :2].reshape(*rotation_matrix.shape[:-2], 6)


def decode_rot6d_safe(rot6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Decode Rot6D with a deterministic fallback for degenerate columns.

    The normal path is Gram--Schmidt.  If a column is zero or nearly parallel
    to the first column, the least-aligned Cartesian basis vector is used to
    complete a valid right-handed frame instead of emitting NaNs.
    """

    if rot6d.shape[-1] != 6:
        raise ValueError(f"Rot6D must have six channels, got {tuple(rot6d.shape)}")
    original_shape = rot6d.shape[:-1]
    x = rot6d.reshape(-1, 3, 2)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    finite = torch.isfinite(a1).all(-1) & torch.isfinite(a2).all(-1)
    a1 = torch.where(finite[:, None], a1, torch.zeros_like(a1))
    a2 = torch.where(finite[:, None], a2, torch.zeros_like(a2))

    norm1 = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    fallback_x = torch.zeros_like(a1)
    fallback_x[:, 0] = 1.0
    b1 = torch.where(norm1 > eps, a1 / norm1.clamp_min(eps), fallback_x)

    orthogonal = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    norm2 = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    valid_second = norm2 > eps

    basis = torch.eye(3, dtype=rot6d.dtype, device=rot6d.device)
    # Pick the Cartesian basis least aligned with b1, then orthogonalise it.
    basis_index = torch.argmin(torch.abs(b1 @ basis.T), dim=-1)
    basis_vector = basis[basis_index]
    fallback_second = basis_vector - (b1 * basis_vector).sum(-1, keepdim=True) * b1
    fallback_second = fallback_second / torch.linalg.vector_norm(
        fallback_second, dim=-1, keepdim=True
    ).clamp_min(eps)
    b2 = torch.where(valid_second, orthogonal / norm2.clamp_min(eps), fallback_second)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1).reshape(*original_shape, 3, 3)


def apply_rotation_delta(
    root_rot6d: torch.Tensor,
    *,
    delta_deg: float,
    axis: str = "z",
    side: str = "left",
) -> torch.Tensor:
    """Apply an explicit SO(3) edit and re-encode it as Rot6D.

    ``side='left'`` means a world/canonical-frame edit ``R_delta @ R``;
    ``side='right'`` means a local-frame edit ``R @ R_delta``.  The sign and
    multiplication side are deliberately explicit for the manual calibration
    required by phase 1.
    """

    if axis not in {"x", "y", "z"}:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    matrix = decode_rot6d_safe(root_rot6d)
    axis_vector = torch.zeros(3, dtype=root_rot6d.dtype, device=root_rot6d.device)
    axis_vector[{"x": 0, "y": 1, "z": 2}[axis]] = math.radians(float(delta_deg))
    delta = axis_angle_to_mat3x3(axis_vector)
    while delta.ndim < matrix.ndim:
        delta = delta.unsqueeze(0)
    edited = delta @ matrix if side == "left" else matrix @ delta
    return encode_rot6d(edited)


def forward_difference(values: torch.Tensor) -> torch.Tensor:
    """Compute the forward difference along the time axis (the penultimate axis)."""

    if values.ndim < 2 or values.shape[-2] < 2:
        raise ValueError("values must contain at least two time frames")
    return values[..., 1:, :] - values[..., :-1, :]


def recover_forward_difference(first_frame: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Recover ``T+1`` frames from the first frame and ``T`` differences."""

    if first_frame.ndim != velocity.ndim or first_frame.shape[-2] != 1:
        raise ValueError("first_frame must have the same rank as velocity and one time frame")
    if first_frame.shape[:-2] != velocity.shape[:-2] or first_frame.shape[-1] != velocity.shape[-1]:
        raise ValueError("first_frame and velocity batch/channel shapes differ")
    return torch.cat((first_frame, first_frame + torch.cumsum(velocity, dim=-2)), dim=-2)


def build_valid_frame_mask(lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Build a boolean ``[batch, frame_count]`` mask from valid lengths."""

    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional")
    if torch.any(lengths < 0) or torch.any(lengths > frame_count):
        raise ValueError("lengths must lie in [0, frame_count]")
    return torch.arange(frame_count, device=lengths.device)[None, :] < lengths[:, None]


def standardize_motion(
    motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Standardize motion without changing its shape or dtype."""

    validate_motion_tensor(motion)
    if mean.shape[-1] != MOTION_LAYOUT.total_dim or std.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError("mean and std must have 276 channels")
    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("std must be finite and strictly positive")
    mean = mean.to(device=motion.device, dtype=motion.dtype)
    std = std.to(device=motion.device, dtype=motion.dtype)
    return (motion - mean) / std


def destandardize_motion(
    motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`standardize_motion`."""

    validate_motion_tensor(motion)
    mean = mean.to(device=motion.device, dtype=motion.dtype)
    std = std.to(device=motion.device, dtype=motion.dtype)
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")
    return motion * std + mean
