"""Full, differentiable 276-D consistency projection (protocol v2).

The legacy representation stores a direct pose view and three redundant
forward-difference views.  ``vimogen_276d_consistency_v2`` makes the direct
pose authority explicit: body-local rotations, a fused root rotation and a
fused root translation are first assembled into a ``T+1`` pose stream; the
stream is then passed through a small differentiable SMPL-X body FK; all 22
joint positions and all three velocity views are finally packed again.

The module is intentionally opt-in.  The v1 reconciliation and finalizers
remain untouched and callers can inject a tiny skeleton in tests or for a
different body model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import torch

from .finalize import FinalizedMotion, finalize_motion
from .phase1 import MOTION_LAYOUT, decode_rot6d_safe, validate_motion_tensor
from .rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle


PROTOCOL_NAME = "vimogen_276d_consistency_v2"
SMPLX_NEUTRAL_22_SKELETON_PATH = "data/body_models/smplx/SMPLX_NEUTRAL.npz"
SMPLX_NEUTRAL_22_SKELETON_ASSET = "motion_rep/smplx_neutral_22_skeleton.json"

# SMPL-X's body-only order is the one frozen in phase1.py.  The first entry
# is the root; all remaining entries index one body-local rotation channel.
SMPLX_22_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
)


@dataclass(frozen=True)
class Skeleton22:
    """A minimal FK skeleton independent of the heavyweight SMPL-X module."""

    rest_offsets: torch.Tensor
    parents: tuple[int, ...] = SMPLX_22_PARENTS
    source: str = SMPLX_NEUTRAL_22_SKELETON_PATH
    # SMPL's transl is an offset from the model origin.  Keep the neutral
    # pelvis location separately: the root is ``transl + root_offset`` and is
    # *not* rotated by the root orientation.  Tiny injected test skeletons
    # can leave this unset, in which case their first rest offset is used.
    root_offset: torch.Tensor | None = None

    def __post_init__(self) -> None:
        offsets = self.rest_offsets
        if not isinstance(offsets, torch.Tensor) or offsets.shape != (22, 3):
            raise ValueError("rest_offsets must have shape [22,3]")
        if not torch.is_floating_point(offsets) or not torch.isfinite(offsets).all():
            raise ValueError("rest_offsets must be finite floating point")
        if len(self.parents) != 22 or self.parents[0] != -1:
            raise ValueError("parents must contain 22 entries with parents[0] == -1")
        for index, parent in enumerate(self.parents[1:], 1):
            if parent < 0 or parent >= index:
                raise ValueError("parents must be a topologically ordered tree")
        root = offsets[0] if self.root_offset is None else self.root_offset
        if root.shape != (3,) or not torch.isfinite(root).all():
            raise ValueError("root_offset must be finite with shape [3]")
        object.__setattr__(self, "root_offset", root)
        # Internal child offsets are relative offsets; the root offset is
        # carried explicitly so FK cannot accidentally rotate it.
        if torch.any(offsets[0] != 0):
            normalized = offsets.clone()
            normalized[0] = 0
            object.__setattr__(self, "rest_offsets", normalized)


@dataclass(frozen=True)
class FKResult:
    """Differentiable FK output with global rotations for auditability."""

    joints: torch.Tensor
    global_rotations: torch.Tensor


@dataclass(frozen=True)
class FKConsistentMotion(FinalizedMotion):
    """Packed v2 result and the frozen FK protocol metadata."""

    protocol: str = PROTOCOL_NAME
    skeleton_source: str = SMPLX_NEUTRAL_22_SKELETON_PATH


@dataclass(frozen=True)
class ConsistencyV2Config:
    """Serializable choices for the opt-in v2 projection."""

    fusion_window: int = 9
    anchor_weight: float = 1.0
    root_rotation_anchor_weight: float = 1.0
    skeleton_path: str | None = None

    def __post_init__(self) -> None:
        if self.fusion_window < 1 or self.fusion_window % 2 == 0:
            raise ValueError("fusion_window must be a positive odd integer")
        if not 0 <= self.anchor_weight <= 1 or not 0 <= self.root_rotation_anchor_weight <= 1:
            raise ValueError("v2 anchor weights must lie in [0,1]")


def _as_dense_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return torch.as_tensor(value)


def _candidate_skeleton_paths(path: str | Path | None) -> list[Path]:
    if path is not None:
        return [Path(path)]
    root = Path(__file__).resolve().parents[1]
    return [
        root / SMPLX_NEUTRAL_22_SKELETON_ASSET,
        Path.cwd() / SMPLX_NEUTRAL_22_SKELETON_PATH,
        root / SMPLX_NEUTRAL_22_SKELETON_PATH,
    ]


def _skeleton_from_json(path: Path) -> Skeleton22:
    values = json.loads(path.read_text(encoding="utf-8"))
    joints = torch.as_tensor(values["rest_joints"], dtype=torch.float32)
    if joints.shape != (22, 3):
        raise ValueError(f"{path} rest_joints must have shape [22,3]")
    offsets = torch.as_tensor(values.get("rest_offsets"), dtype=torch.float32) if "rest_offsets" in values else joints.clone()
    if offsets.shape != (22, 3):
        raise ValueError(f"{path} rest_offsets must have shape [22,3]")
    if "rest_offsets" not in values:
        for index, parent in enumerate(SMPLX_22_PARENTS):
            if parent >= 0:
                offsets[index] = joints[index] - joints[parent]
        offsets[0] = 0
    return Skeleton22(
        offsets,
        tuple(int(item) for item in values.get("parents", SMPLX_22_PARENTS)),
        str(values.get("source", path)),
        joints[0],
    )


def _skeleton_from_npz(path: Path) -> Skeleton22:
    import numpy as np

    values = np.load(path, allow_pickle=True)
    if "J_regressor" not in values or "v_template" not in values:
        raise ValueError(f"{path} lacks J_regressor or v_template")
    regressor = _as_dense_tensor(values["J_regressor"][:22]).float()
    template = _as_dense_tensor(values["v_template"]).float()
    if regressor.ndim != 2 or regressor.shape[0] != 22:
        raise ValueError("SMPL-X J_regressor must provide its first 22 joints")
    joints = regressor @ template
    offsets = joints.clone()
    for index, parent in enumerate(SMPLX_22_PARENTS):
        if parent >= 0:
            offsets[index] = joints[index] - joints[parent]
    root_offset = joints[0]
    offsets[0] = 0
    return Skeleton22(offsets, source=str(path), root_offset=root_offset)


def _skeleton_from_pickle(path: Path) -> Skeleton22:
    with path.open("rb") as handle:
        values = pickle.load(handle, encoding="latin1")
    regressor = _as_dense_tensor(values["J_regressor"][:22]).float()
    template = _as_dense_tensor(values.get("v_template", values.get("vt"))).float()
    joints = regressor @ template
    offsets = joints.clone()
    for index, parent in enumerate(SMPLX_22_PARENTS):
        if parent >= 0:
            offsets[index] = joints[index] - joints[parent]
    root_offset = joints[0]
    offsets[0] = 0
    return Skeleton22(offsets, source=str(path), root_offset=root_offset)


def _skeleton_from_torch(path: Path) -> Skeleton22:
    values = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(values, Skeleton22):
        return values
    if not isinstance(values, Mapping) or "rest_offsets" not in values:
        raise ValueError(f"{path} must contain a rest_offsets tensor")
    parents = tuple(int(item) for item in values.get("parents", SMPLX_22_PARENTS))
    return Skeleton22(
        torch.as_tensor(values["rest_offsets"]).float(),
        parents,
        str(path),
        None if "root_offset" not in values else torch.as_tensor(values["root_offset"]).float(),
    )


def load_smplx_neutral_22_skeleton(path: str | Path | None = None) -> Skeleton22:
    """Load the frozen neutral SMPL-X 22-joint skeleton.

    ``path`` may point to the source ``.npz``/``.pkl`` or a frozen ``.pt``
    artifact produced by ``freeze_representation_v2_protocol.py``.  Loading
    is lazy so importing old v1 code never requires body-model assets.
    """

    candidates = _candidate_skeleton_paths(path)
    for candidate in candidates:
        if not candidate.exists():
            continue
        suffix = candidate.suffix.lower()
        if suffix == ".npz":
            return _skeleton_from_npz(candidate)
        if suffix in {".pkl", ".pickle"}:
            return _skeleton_from_pickle(candidate)
        if suffix in {".pt", ".pth"}:
            return _skeleton_from_torch(candidate)
        if suffix == ".json":
            return _skeleton_from_json(candidate)
        raise ValueError(f"unsupported skeleton file type: {candidate}")
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"SMPL-X neutral skeleton not found; searched {searched}")


@lru_cache(maxsize=4)
def default_smplx_neutral_22_skeleton() -> Skeleton22:
    return load_smplx_neutral_22_skeleton()


def _resolve_skeleton(
    skeleton: Skeleton22 | Mapping[str, Any] | None,
    rest_offsets: torch.Tensor | None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None,
) -> Skeleton22:
    if skeleton is not None and (rest_offsets is not None or parents is not None):
        raise ValueError("provide skeleton or rest_offsets/parents, not both")
    if skeleton is not None:
        if isinstance(skeleton, Skeleton22):
            return skeleton
        if isinstance(skeleton, Mapping):
            return Skeleton22(
                torch.as_tensor(skeleton["rest_offsets"]).float(),
                tuple(int(item) for item in skeleton.get("parents", SMPLX_22_PARENTS)),
                str(skeleton.get("source", "injected")),
                None if "root_offset" not in skeleton else torch.as_tensor(skeleton["root_offset"]).float(),
            )
        raise TypeError("skeleton must be Skeleton22 or a mapping")
    if rest_offsets is None:
        if parents is not None:
            raise ValueError("parents requires rest_offsets")
        return default_smplx_neutral_22_skeleton()
    parent_tuple = SMPLX_22_PARENTS if parents is None else tuple(int(item) for item in parents)
    return Skeleton22(torch.as_tensor(rest_offsets).float(), parent_tuple, "injected")


def _validate_fk_inputs(
    local_rotations: torch.Tensor,
    root_rotation: torch.Tensor,
    root_translation: torch.Tensor,
) -> None:
    if local_rotations.ndim not in (4, 5) or local_rotations.shape[-3:] != (21, 3, 3):
        raise ValueError("local_rotations must have shape [T,21,3,3] or [B,T,21,3,3]")
    batched = local_rotations.ndim == 5
    expected_prefix = local_rotations.shape[:1] if not batched else local_rotations.shape[:2]
    if root_rotation.shape != expected_prefix + (3, 3):
        raise ValueError(f"root_rotation must have shape {expected_prefix + (3,3)}")
    if root_translation.shape != expected_prefix + (3,):
        raise ValueError(f"root_translation must have shape {expected_prefix + (3,)}")
    for value, name in ((local_rotations, "local_rotations"), (root_rotation, "root_rotation"), (root_translation, "root_translation")):
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point")
    if local_rotations.shape[-4 if batched else -4] < 1:
        raise ValueError("FK requires at least one frame")


def differentiable_forward_kinematics(
    local_rotations: torch.Tensor,
    root_rotation: torch.Tensor,
    root_translation: torch.Tensor,
    *,
    skeleton: Skeleton22 | Mapping[str, Any] | None = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> FKResult:
    """Run differentiable FK for a body-only 22-joint SMPL-X tree."""

    _validate_fk_inputs(local_rotations, root_rotation, root_translation)
    tree = _resolve_skeleton(skeleton, rest_offsets, parents)
    offsets = tree.rest_offsets.to(device=local_rotations.device, dtype=local_rotations.dtype)
    global_rotations = [root_rotation]
    root_offset = tree.root_offset.to(device=root_translation.device, dtype=root_translation.dtype)
    # Root translation is an origin offset, while root rotation only affects
    # child offsets.  In particular, do not apply root_rotation to this
    # neutral pelvis offset.
    joints = [root_translation + root_offset]
    for index in range(1, 22):
        parent = tree.parents[index]
        global_rotation = global_rotations[parent] @ local_rotations[..., index - 1, :, :]
        position = joints[parent] + global_rotations[parent] @ offsets[index]
        global_rotations.append(global_rotation)
        joints.append(position)
    return FKResult(joints=torch.stack(joints, dim=-2), global_rotations=torch.stack(global_rotations, dim=-3))


def _moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    if window == 1 or values.shape[-2] == 0:
        return values
    radius = window // 2
    left = values[..., :1, :].expand(*values.shape[:-2], radius, values.shape[-1])
    right = values[..., -1:, :].expand(*values.shape[:-2], radius, values.shape[-1])
    padded = torch.cat((left, values, right), dim=-2)
    return padded.unfold(-2, window, 1).mean(dim=-1)


def _extrapolate_last(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-2] == 1:
        return values[..., -1:, :]
    return values[..., -1:, :] + (values[..., -1:, :] - values[..., -2:-1, :])


def _fuse_translation(direct: torch.Tensor, velocity: torch.Tensor, window: int, weight: float) -> torch.Tensor:
    direct_stream = torch.cat((direct, _extrapolate_last(direct)), dim=-2)
    velocity_stream = torch.cat((direct[..., :1, :], direct[..., :1, :] + torch.cumsum(velocity, dim=-2)), dim=-2)
    correction = _moving_average(direct_stream - velocity_stream, window)
    return velocity_stream + float(weight) * correction


def _fuse_root_rotation(direct: torch.Tensor, velocity: torch.Tensor, window: int, weight: float) -> torch.Tensor:
    velocity_matrix = decode_rot6d_safe(velocity)
    if direct.shape[-3] == 1:
        direct_next = direct[..., -1:, :, :]
    else:
        increment = direct[..., -1:, :, :] @ direct[..., -2:-1, :, :].transpose(-1, -2)
        direct_next = increment @ direct[..., -1:, :, :]
    direct_stream = torch.cat((direct, direct_next), dim=-3)
    frames = [direct[..., :1, :, :]]
    for index in range(velocity_matrix.shape[-3]):
        frames.append(velocity_matrix[..., index:index + 1, :, :] @ frames[-1])
    velocity_stream = torch.cat(frames, dim=-3)
    correction = direct_stream @ velocity_stream.transpose(-1, -2)
    correction_axis = mat3x3_to_axis_angle(correction)
    smoothed = _moving_average(correction_axis, window)
    return axis_angle_to_mat3x3(float(weight) * smoothed) @ velocity_stream


def _single_consistency(
    motion: torch.Tensor,
    *,
    tree: Skeleton22,
    fusion_window: int,
    anchor_weight: float,
    root_rotation_anchor_weight: float,
    valid_mask: torch.Tensor | None,
) -> FinalizedMotion:
    validate_motion_tensor(motion)
    if motion.ndim != 2 or motion.shape[0] < 1:
        raise ValueError("motion must have shape [T,276] with T >= 1")
    frames = motion.shape[0]
    # A row mask is allowed to contain tail padding.  Replace that padding by
    # the last observed row before fusion, then place the hidden T+1 pose
    # immediately after the last valid row (rather than at the physical end
    # of a padded tensor).  This preserves the final observed output row.
    source = motion
    if valid_mask is None:
        pose_mask = torch.ones(frames + 1, dtype=torch.bool, device=motion.device)
    else:
        if valid_mask.shape != (frames,) or valid_mask.dtype is not torch.bool:
            raise ValueError(f"valid_mask must have shape {(frames,)} and dtype bool")
        if torch.any(valid_mask[1:] & ~valid_mask[:-1]):
            raise ValueError("valid_mask must be a contiguous valid prefix followed by tail padding")
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        pose_mask = torch.zeros(frames + 1, dtype=torch.bool, device=motion.device)
        pose_mask[:frames] = valid_mask
        if valid_indices.numel():
            last = int(valid_indices[-1].item())
            if last + 1 < frames:
                source = motion.clone()
                source[last + 1 :] = source[last : last + 1]
            pose_mask[last + 1] = True
    body = decode_rot6d_safe(source[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6)).float()
    body_stream = torch.cat((body, body[-1:]), dim=0)
    direct_root = decode_rot6d_safe(source[:, MOTION_LAYOUT.root_rotation]).float()
    fused_root = _fuse_root_rotation(
        direct_root,
        source[:, MOTION_LAYOUT.root_rotation_velocity].float(),
        fusion_window,
        root_rotation_anchor_weight,
    )
    fused_translation = _fuse_translation(
        source[:, MOTION_LAYOUT.root_translation].float(),
        source[:, MOTION_LAYOUT.root_translation_velocity].float(),
        fusion_window,
        anchor_weight,
    )
    fk = differentiable_forward_kinematics(
        body_stream,
        fused_root,
        fused_translation,
        skeleton=tree,
    )
    return finalize_motion(
        body_stream,
        fk.joints,
        fused_root,
        fused_translation,
        valid_mask=pose_mask,
    )


def _align_statistic(value: torch.Tensor, motion: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 1 and value.shape == (276,):
        return value.to(device=motion.device, dtype=motion.dtype)
    if value.ndim == 2 and motion.ndim == 3 and value.shape == (motion.shape[0], 276):
        return value.to(device=motion.device, dtype=motion.dtype).unsqueeze(1)
    if value.ndim == 2 and motion.ndim == 2 and value.shape == (1, 276):
        return value[0].to(device=motion.device, dtype=motion.dtype)
    raise ValueError(f"{name} must have shape [276] or [B,276]")


def reconcile_motion_tensor_v2(
    motion: torch.Tensor,
    *,
    config: ConsistencyV2Config | None = None,
    fusion_window: int = 9,
    anchor_weight: float = 1.0,
    root_rotation_anchor_weight: float | None = None,
    valid_mask: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    input_standardized: bool = False,
    output_standardized: bool = False,
    output_dtype: torch.dtype | None = None,
    skeleton: Skeleton22 | Mapping[str, Any] | None = None,
    rest_offsets: torch.Tensor | None = None,
    parents: tuple[int, ...] | list[int] | torch.Tensor | None = None,
) -> FKConsistentMotion:
    """Project physical or standardized ``[T,276]``/``[B,T,276]`` data.

    Geometry is always evaluated in FP32 and remains differentiable with
    respect to a floating input motion.  Only the requested final cast is
    lossy; this is important for the guidance loss and for the 1e-4-degree
    root-channel audit.
    """

    if config is not None:
        fusion_window = config.fusion_window
        anchor_weight = config.anchor_weight
        root_rotation_anchor_weight = config.root_rotation_anchor_weight
        if skeleton is None and config.skeleton_path is not None:
            skeleton = load_smplx_neutral_22_skeleton(config.skeleton_path)
    validate_motion_tensor(motion)
    if fusion_window < 1 or fusion_window % 2 == 0:
        raise ValueError("fusion_window must be a positive odd integer")
    if not 0 <= anchor_weight <= 1:
        raise ValueError("anchor_weight must lie in [0,1]")
    root_weight = anchor_weight if root_rotation_anchor_weight is None else float(root_rotation_anchor_weight)
    if not 0 <= root_weight <= 1:
        raise ValueError("root_rotation_anchor_weight must lie in [0,1]")
    if (input_standardized or output_standardized) and (mean is None or std is None):
        raise ValueError("mean and std must be supplied for standardized input/output")
    if not (input_standardized or output_standardized) and (mean is not None or std is not None):
        raise ValueError("mean and std require an explicit standardization flag")
    if input_standardized:
        mean_aligned = _align_statistic(mean, motion, "mean")
        std_aligned = _align_statistic(std, motion, "std")
        if not torch.isfinite(mean_aligned).all() or not torch.isfinite(std_aligned).all() or torch.any(std_aligned <= 0):
            raise ValueError("mean/std must be finite and std strictly positive")
        physical = (motion * std_aligned + mean_aligned).float()
    else:
        physical = motion.float()
    if valid_mask is not None and (tuple(valid_mask.shape) != tuple(physical.shape[:-1]) or valid_mask.dtype is not torch.bool):
        raise ValueError(f"valid_mask must have shape {tuple(physical.shape[:-1])} and dtype bool")
    tree = _resolve_skeleton(skeleton, rest_offsets, parents)
    if physical.ndim == 3:
        values = [
            _single_consistency(
                physical[index],
                tree=tree,
                fusion_window=fusion_window,
                anchor_weight=float(anchor_weight),
                root_rotation_anchor_weight=root_weight,
                valid_mask=None if valid_mask is None else valid_mask[index],
            )
            for index in range(physical.shape[0])
        ]
        result_motion = torch.stack([item.motion for item in values], dim=0)
        result_mask = torch.stack([item.valid_mask for item in values], dim=0)
    else:
        value = _single_consistency(
            physical,
            tree=tree,
            fusion_window=fusion_window,
            anchor_weight=float(anchor_weight),
            root_rotation_anchor_weight=root_weight,
            valid_mask=valid_mask,
        )
        result_motion, result_mask = value.motion, value.valid_mask
    if output_standardized:
        mean_aligned = _align_statistic(mean, result_motion, "mean")
        std_aligned = _align_statistic(std, result_motion, "std")
        result_motion = (result_motion - mean_aligned) / std_aligned
        result_motion = result_motion.masked_fill(~result_mask.unsqueeze(-1), 0)
    if output_dtype is not None:
        result_motion = result_motion.to(output_dtype)
    return FKConsistentMotion(
        motion=result_motion,
        valid_mask=result_mask,
        protocol=PROTOCOL_NAME,
        skeleton_source=tree.source,
    )


def freeze_smplx_neutral_22_skeleton(source: str | Path | None = None, output: str | Path | None = None) -> dict[str, Any]:
    """Materialize an auditable skeleton artifact without running a model."""

    tree = load_smplx_neutral_22_skeleton(source)
    report: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "source": tree.source,
        "parents": list(tree.parents),
        "joint_count": 22,
        "rest_offsets_shape": list(tree.rest_offsets.shape),
        "rest_offsets": tree.rest_offsets.tolist(),
        "root_offset": tree.root_offset.tolist(),
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rest_offsets": tree.rest_offsets.cpu(), "root_offset": tree.root_offset.cpu(), "parents": tree.parents, "source": tree.source, "protocol": PROTOCOL_NAME}, path)
        path.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = [
    "ConsistencyV2Config",
    "FKConsistentMotion",
    "FKResult",
    "PROTOCOL_NAME",
    "SMPLX_22_PARENTS",
    "SMPLX_NEUTRAL_22_SKELETON_PATH",
    "SMPLX_NEUTRAL_22_SKELETON_ASSET",
    "Skeleton22",
    "default_smplx_neutral_22_skeleton",
    "differentiable_fk_22",
    "differentiable_forward_kinematics",
    "full_fk_consistency",
    "freeze_smplx_neutral_22_skeleton",
    "load_smplx_neutral_22_skeleton",
    "reconcile_motion_tensor_v2",
]

# Descriptive aliases make the boundary discoverable without changing the
# canonical function names above.
differentiable_fk_22 = differentiable_forward_kinematics
full_fk_consistency = reconcile_motion_tensor_v2
