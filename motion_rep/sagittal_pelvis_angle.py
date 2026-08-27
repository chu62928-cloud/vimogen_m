"""Root-rotation based local-sagittal pelvis tilt (v2).

The legacy :func:`motion_rep.pelvis_angle.pelvis_pitch_degrees` function is
intentionally not imported or modified here.  This module defines the v2
geometry boundary explicitly:

* ``root_rotation`` maps SMPL-X local axes to canonical/world axes;
* local ``+z`` is the pelvis forward axis;
* canonical ``+z`` is up;
* the horizontal projection of that forward axis defines the person's
  per-frame yaw, which is removed before measuring tilt in the sagittal
  plane; and
* the signed angle is positive when a positive rotation about the person's
  right ``+x`` axis sends forward ``+y`` towards up ``+z``.

The implementation avoids an ``atan2``-based yaw in the main angle path.
Using the projected forward direction directly is algebraically equivalent
to removing yaw, but has a better-conditioned derivative around yaw wrap.
Near a vertical forward projection, a canonical ``+y`` heading is used as a
finite differentiable fallback.  This is a numerical fallback only; callers
can inspect the returned horizontal-projection norm when such frames need to
be masked.
"""

from __future__ import annotations

import math

import torch


V2_PROTOCOL = "root_rotation_local_sagittal_pelvis_tilt_v2"


def _validate_root_rotation(root_rotation: torch.Tensor) -> None:
    if not isinstance(root_rotation, torch.Tensor):
        raise TypeError(f"root_rotation must be a torch.Tensor, got {type(root_rotation)!r}")
    if root_rotation.ndim < 2 or root_rotation.shape[-2:] != (3, 3):
        raise ValueError(
            "root_rotation must have shape [...,3,3], "
            f"got {tuple(root_rotation.shape)}"
        )
    if not torch.is_floating_point(root_rotation):
        raise TypeError("root_rotation must be floating point")
    if not torch.isfinite(root_rotation).all():
        raise ValueError("root_rotation contains non-finite values")


def _axis(index: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    value = torch.zeros(3, dtype=dtype, device=device)
    value[index] = 1.0
    return value


def _safe_unit(value: torch.Tensor, fallback: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize ``value`` with a finite, differentiable low-norm fallback.

    The denominator is smooth (``sqrt(norm² + eps²)``), while the returned
    value switches to a deterministic unit fallback below ``eps``.  Clamping
    only the denominator would avoid NaNs but would leave a non-unit direction
    at the singularity; the explicit fallback keeps the yaw basis meaningful.
    """

    norm_sq = (value * value).sum(dim=-1, keepdim=True)
    norm = torch.sqrt(norm_sq)
    denominator = torch.sqrt(norm_sq + float(eps) ** 2)
    candidate = value / denominator
    fallback = fallback.to(device=value.device, dtype=value.dtype).expand_as(candidate)
    return torch.where(norm > float(eps), candidate, fallback), norm


def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive number")
    return eps


def _forward_and_heading(
    root_rotation: torch.Tensor,
    *,
    local_forward_axis: int,
    up_axis: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return forward, horizontal heading, up and raw horizontal norm."""

    forward_local = _axis(
        local_forward_axis, dtype=root_rotation.dtype, device=root_rotation.device
    )
    forward = root_rotation @ forward_local
    up = _axis(up_axis, dtype=root_rotation.dtype, device=root_rotation.device)
    horizontal = forward - (forward * up).sum(dim=-1, keepdim=True) * up
    # In the canonical x/y horizontal plane and with canonical +z up, +y is
    # the zero-yaw forward direction.  The generic axis choice keeps the
    # helper useful for tests while the public v2 default remains +z up.
    reference_index = 1 if up_axis == 2 else (0 if up_axis == 1 else 1)
    reference = _axis(reference_index, dtype=root_rotation.dtype, device=root_rotation.device)
    heading, horizontal_norm = _safe_unit(horizontal, reference, eps)
    return forward, heading, up, horizontal_norm.squeeze(-1)


def person_forward_horizontal_axis(
    root_rotation: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-frame horizontal forward/yaw axis and its norm.

    The output shape is ``root_rotation.shape[:-2] + (3,)`` and the norm has
    shape ``root_rotation.shape[:-2]``.  Norms below ``eps`` use canonical
    ``+y`` as a finite fallback direction.  For normal rotation matrices this
    axis is a unit vector and is the direction used to remove person yaw.
    """

    _validate_root_rotation(root_rotation)
    eps = _validate_eps(eps)
    _, heading, _, horizontal_norm = _forward_and_heading(
        root_rotation, local_forward_axis=2, up_axis=2, eps=eps
    )
    return heading, horizontal_norm


def person_yaw_radians(root_rotation: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Return each frame's yaw around canonical ``+z`` in radians.

    Yaw zero means local SMPL-X ``+z`` faces canonical ``+y``.  Positive yaw
    follows the right-handed canonical ``+z`` rotation, so ``+y`` moves toward
    canonical ``-x``.  The angle is exposed for audit/debugging; the tilt
    function itself uses the heading basis directly to avoid wrap-gradient
    issues.
    """

    heading, _ = person_forward_horizontal_axis(root_rotation, eps=eps)
    return torch.atan2(-heading[..., 0], heading[..., 1])


def remove_person_yaw(root_rotation: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Left-multiply ``root_rotation`` by the inverse per-frame yaw.

    After this operation a non-degenerate local ``+z`` forward axis points
    along canonical ``+y`` in the horizontal plane.  The returned matrices
    remain differentiable with respect to the input rotation.
    """

    _validate_root_rotation(root_rotation)
    yaw = person_yaw_radians(root_rotation, eps=eps)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    inverse_yaw = torch.stack(
        (c, s, zero, -s, c, zero, zero, zero, one), dim=-1
    ).reshape(*root_rotation.shape[:-2], 3, 3)
    return inverse_yaw @ root_rotation


def person_right_axis(root_rotation: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Return the horizontal normal of the person's sagittal plane.

    This is the axis required by a G1-style per-frame SO(3) left-multiplying
    correction.  It is ``heading cross up``, so it stays normal to the
    heading-defined sagittal plane even when the pelvis itself has local
    roll.  The safe normalization gives a finite canonical ``+x`` fallback
    at a degenerate heading.
    """

    _validate_root_rotation(root_rotation)
    eps = _validate_eps(eps)
    heading, _ = person_forward_horizontal_axis(root_rotation, eps=eps)
    up = _axis(2, dtype=root_rotation.dtype, device=root_rotation.device)
    right = torch.cross(heading, up.expand_as(heading), dim=-1)
    normalized, _ = _safe_unit(
        right,
        _axis(0, dtype=root_rotation.dtype, device=root_rotation.device),
        eps,
    )
    return normalized


def person_sagittal_basis(
    root_rotation: torch.Tensor, *, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(forward, up, right)`` for each local sagittal plane.

    ``forward`` is the horizontal direction after removing yaw, ``up`` is
    canonical ``+z``, and ``right`` is the corresponding horizontal plane
    normal.  The right axis is exposed separately as a convenience for
    per-frame left multiplication.
    """

    _validate_root_rotation(root_rotation)
    eps = _validate_eps(eps)
    _, forward, up, _ = _forward_and_heading(
        root_rotation, local_forward_axis=2, up_axis=2, eps=eps
    )
    right = torch.cross(forward, up.expand_as(forward), dim=-1)
    right, _ = _safe_unit(
        right,
        _axis(0, dtype=root_rotation.dtype, device=root_rotation.device),
        eps,
    )
    return forward, up.expand_as(forward), right


def pelvis_sagittal_tilt_radians(
    root_rotation: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Measure signed root forward tilt in the yaw-removed sagittal plane.

    The angle is ``atan2(forward·up, forward·heading)``.  A positive local
    right-axis rotation therefore maps forward ``+y`` toward up ``+z`` and
    increases the returned angle.  The result has shape ``root_rotation``'s
    leading shape, e.g. ``[T]`` for ``[T,3,3]`` or ``[B,T]`` for ``[B,T,3,3]``.
    """

    _validate_root_rotation(root_rotation)
    eps = _validate_eps(eps)
    forward, heading, up, horizontal_norm = _forward_and_heading(
        root_rotation, local_forward_axis=2, up_axis=2, eps=eps
    )
    vertical = (forward * up).sum(dim=-1)
    along_heading = (forward * heading).sum(dim=-1)
    # ``along_heading`` is non-negative by construction.  The explicit
    # horizontal-norm term documents the low-projection fallback and keeps a
    # useful audit quantity in this path without dividing by it.
    del horizontal_norm
    return torch.atan2(vertical, along_heading)


def pelvis_sagittal_tilt_degrees(
    root_rotation: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Return :func:`pelvis_sagittal_tilt_radians` in degrees."""

    return pelvis_sagittal_tilt_radians(root_rotation, eps=eps) * (180.0 / math.pi)


def _axis_angle_matrix(axis: torch.Tensor, angle_radians: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues matrix for a per-frame unit axis."""

    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(skew)
    angle = angle_radians.unsqueeze(-1).unsqueeze(-1)
    return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def apply_person_right_axis_rotation(
    root_rotation: torch.Tensor,
    delta_degrees: float | torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply a per-frame SO(3) correction by left multiplication.

    ``delta_degrees`` may be scalar or broadcastable to
    ``root_rotation.shape[:-2]``.  The correction is
    ``R_delta[t] @ root_rotation[t]`` with ``R_delta[t]`` rotating about the
    current person's right axis.  Thus a positive ``0.5`` degree correction
    increases the v2 tilt by approximately ``0.5`` degree at every pure yaw.
    """

    _validate_root_rotation(root_rotation)
    eps = _validate_eps(eps)
    axis = person_right_axis(root_rotation, eps=eps)
    delta = torch.as_tensor(delta_degrees, dtype=root_rotation.dtype, device=root_rotation.device)
    prefix = root_rotation.shape[:-2]
    try:
        delta = torch.broadcast_to(delta, prefix)
    except RuntimeError as error:
        raise ValueError(
            f"delta_degrees must be scalar or broadcastable to {tuple(prefix)}, "
            f"got {tuple(delta.shape)}"
        ) from error
    correction = _axis_angle_matrix(axis, delta * (math.pi / 180.0))
    return correction @ root_rotation


# Explicit aliases keep likely call sites readable while retaining one
# implementation and one protocol definition.
local_sagittal_tilt_degrees = pelvis_sagittal_tilt_degrees
pelvis_pitch_degrees_v2 = pelvis_sagittal_tilt_degrees
root_rotation_to_person_yaw = person_yaw_radians
root_rotation_to_person_right_axis = person_right_axis
left_multiply_person_right_delta = apply_person_right_axis_rotation


__all__ = [
    "V2_PROTOCOL",
    "apply_person_right_axis_rotation",
    "left_multiply_person_right_delta",
    "local_sagittal_tilt_degrees",
    "pelvis_pitch_degrees_v2",
    "pelvis_sagittal_tilt_degrees",
    "pelvis_sagittal_tilt_radians",
    "person_forward_horizontal_axis",
    "person_right_axis",
    "person_sagittal_basis",
    "person_yaw_radians",
    "remove_person_yaw",
    "root_rotation_to_person_right_axis",
    "root_rotation_to_person_yaw",
]
