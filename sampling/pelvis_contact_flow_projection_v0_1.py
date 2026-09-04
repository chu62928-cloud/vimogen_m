"""Sampling-time pelvis/contact endpoint projection for frozen ViMoGen.

This module implements the independent protocol
``vimogen_pelvis_contact_flow_projection_v0_1``.  It edits only authoritative
direct pose streams, rebuilds the redundant 276-D representation after every
accepted update, and never optimises the initial noise.  The only controlled
quantities are the frozen-v1.3 pelvis target, frozen-v3.0.1 heel/toe anchors,
and currently active ground penetrations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from evaluation.pelvis_contact_compensation_v3 import (
    patch_centres,
    target_root_rotation,
)
from motion_rep.phase1 import (
    MOTION_LAYOUT,
    SMPLX_22_JOINT_INDEX,
    decode_rot6d_safe,
    encode_rot6d,
)
from motion_rep.pose_authority import authority_project
from motion_rep.rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle


PROTOCOL_NAME = "vimogen_pelvis_contact_flow_projection_v0_1"
METHOD_NAME = "ProjFlow-inspired iterative kinematic sampling projection"
EUCLIDEAN_METRIC = "euclidean"
KINEMATIC_TEMPORAL_METRIC = "kinematic_temporal"
PENETRATION_METHOD = "active_equality_approximation"

ACTIVE_JOINT_NAMES = (
    "spine1",
    "spine2",
    "spine3",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
)
ACTIVE_BODY_INDICES = tuple(
    SMPLX_22_JOINT_INDEX[name] - 1 for name in ACTIVE_JOINT_NAMES
)
ROTATION_NAMES = ("root",) + ACTIVE_JOINT_NAMES
VARIABLES_PER_FRAME = 3 + 3 * len(ROTATION_NAMES)


def _broadcast_sigma(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    sigma = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    while sigma.ndim < reference.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def predict_clean_endpoint(
    x_sigma: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Recover the clean endpoint for ViMoGen's flow parameterisation."""

    if x_sigma.shape != velocity.shape:
        raise ValueError("x_sigma and velocity must have identical shapes")
    return x_sigma - _broadcast_sigma(sigma, x_sigma) * velocity


def recompose_velocity(
    x_sigma: torch.Tensor,
    clean_endpoint: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Recompose velocity from an endpoint without embedding scheduler logic."""

    if x_sigma.shape != clean_endpoint.shape:
        raise ValueError("x_sigma and clean_endpoint must have identical shapes")
    value = _broadcast_sigma(sigma, x_sigma)
    if torch.any(value.abs() < eps):
        raise ValueError("velocity recomposition is undefined at sigma=0")
    return (x_sigma - clean_endpoint) / value


def so3_exp(tangent: torch.Tensor) -> torch.Tensor:
    if tangent.shape[-1] != 3:
        raise ValueError("SO(3) tangent vectors must end in 3")
    return axis_angle_to_mat3x3(tangent)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("SO(3) rotations must end in [3,3]")
    return mat3x3_to_axis_angle(rotation)


def project_increment_norms(
    increment: torch.Tensor,
    *,
    max_joint_increment_deg: float,
    max_root_translation_m: float,
) -> torch.Tensor:
    """Project each physical vector onto an L2 trust-region ball."""

    if increment.ndim != 2 or increment.shape[-1] != VARIABLES_PER_FRAME:
        raise ValueError(
            f"increment must have shape [N,{VARIABLES_PER_FRAME}]"
        )
    result = increment.clone()
    translation = result[:, :3]
    translation_norm = torch.linalg.vector_norm(
        translation, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    result[:, :3] = translation * (
        max_root_translation_m / translation_norm
    ).clamp(max=1.0)
    rotations = result[:, 3:].reshape(result.shape[0], len(ROTATION_NAMES), 3)
    rotation_norm = torch.linalg.vector_norm(
        rotations, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    limit = math.radians(max_joint_increment_deg)
    rotations = rotations * (limit / rotation_norm).clamp(max=1.0)
    result[:, 3:] = rotations.reshape(result.shape[0], -1)
    return result


def autograd_jacobian(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
) -> torch.Tensor:
    """Return a dense Jacobian used by the real projector and toy tests."""

    with torch.enable_grad():
        value = point.detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            function,
            value,
            create_graph=False,
            strict=False,
            vectorize=True,
        )
    return jacobian.reshape(-1, point.numel()).detach()


def finite_difference_jacobian(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    *,
    step: float = 1.0e-4,
) -> torch.Tensor:
    """Central finite-difference reference for Jacobian validation tests."""

    flat = point.detach().reshape(-1)
    columns = []
    for index in range(flat.numel()):
        offset = torch.zeros_like(flat)
        offset[index] = step
        plus = function((flat + offset).reshape_as(point)).reshape(-1)
        minus = function((flat - offset).reshape_as(point)).reshape(-1)
        columns.append((plus - minus) / (2.0 * step))
    return torch.stack(columns, dim=-1)


@dataclass(frozen=True)
class ProjectorConfig:
    protocol: str = PROTOCOL_NAME
    metric: str = KINEMATIC_TEMPORAL_METRIC
    enabled: bool = True
    sigma_min: float = 0.0662879
    sigma_max: float = 0.65
    lambda_root: float = 10.0
    lambda_skel: float = 1.0
    lambda_vel: float = 1.0
    lambda_acc: float = 5.0
    epsilon: float = 1.0e-6
    contact_weight: float = 1.0e6
    max_relinearization_iters: int = 5
    pelvis_tolerance_deg: float = 0.25
    contact_tolerance_m: float = 0.001
    penetration_epsilon_m: float = 0.0005
    relative_improvement_tolerance: float = 1.0e-3
    max_joint_increment_deg: float = 5.0
    max_root_translation_m: float = 0.010
    backtracking_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)

    def validate(self) -> None:
        if self.protocol != PROTOCOL_NAME:
            raise ValueError(f"protocol must be {PROTOCOL_NAME}")
        if self.metric not in {EUCLIDEAN_METRIC, KINEMATIC_TEMPORAL_METRIC}:
            raise ValueError("metric must be euclidean or kinematic_temporal")
        if not 0.0 <= self.sigma_min <= self.sigma_max <= 1.0:
            raise ValueError("sigma window must lie in [0,1]")
        positive = (
            self.lambda_root,
            self.epsilon,
            self.contact_weight,
            self.pelvis_tolerance_deg,
            self.contact_tolerance_m,
            self.max_joint_increment_deg,
            self.max_root_translation_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("positive projector bounds and weights are required")
        if min(self.lambda_skel, self.lambda_vel, self.lambda_acc) < 0.0:
            raise ValueError("metric regularisation weights cannot be negative")
        if self.max_relinearization_iters < 1:
            raise ValueError("max_relinearization_iters must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ProjectorConfig":
        values = dict(values or {})
        defaults = asdict(cls())
        for key in defaults:
            if key in values:
                defaults[key] = values[key]
        defaults["backtracking_alphas"] = tuple(defaults["backtracking_alphas"])
        result = cls(**defaults)
        result.validate()
        return result


@dataclass
class ProjectionResult:
    projected_clean_motion: torch.Tensor
    pre_residuals: dict[str, float | int | None]
    post_residuals: dict[str, float | int | None]
    pelvis_residual: float
    heel_residual: float
    toe_residual: float
    penetration_residual: float
    delta_q_norm: float
    delta_root_translation_norm: float
    num_relinearization_iters: int
    jacobian_condition: float | None
    active_penetration_constraints: int
    converged: bool
    finite: bool
    records: list[dict[str, Any]] = field(default_factory=list)

    def diagnostics(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "projected_clean_motion"
        }


@dataclass(frozen=True)
class ProjectionFinalOutputs:
    g0: torch.Tensor
    protocol: str
    summary: dict[str, Any]


def _add_quadratic(matrix: torch.Tensor, indices: Sequence[int], weight: float) -> None:
    for index in indices:
        matrix[index, index] += weight


def build_projection_metric(
    frame_count: int,
    config: ProjectorConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the configured dense pilot metric for one contact window."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    size = frame_count * VARIABLES_PER_FRAME
    if config.metric == EUCLIDEAN_METRIC:
        return torch.eye(size, device=device, dtype=dtype)
    matrix = torch.eye(size, device=device, dtype=dtype) * config.epsilon
    for frame in range(frame_count):
        base = frame * VARIABLES_PER_FRAME
        _add_quadratic(matrix, range(base, base + 3), config.lambda_root)
        _add_quadratic(
            matrix,
            range(base + 3, base + VARIABLES_PER_FRAME),
            1.0,
        )
        # Adjacent active rotations share a deterministic skeleton-coupling
        # penalty.  No per-joint hand tuning is introduced in v0.1.
        for left in range(len(ROTATION_NAMES) - 1):
            right = left + 1
            for axis in range(3):
                i = base + 3 + left * 3 + axis
                j = base + 3 + right * 3 + axis
                matrix[i, i] += config.lambda_skel
                matrix[j, j] += config.lambda_skel
                matrix[i, j] -= config.lambda_skel
                matrix[j, i] -= config.lambda_skel
    identity = torch.eye(VARIABLES_PER_FRAME, device=device, dtype=dtype)
    for frame in range(1, frame_count):
        left = slice((frame - 1) * VARIABLES_PER_FRAME, frame * VARIABLES_PER_FRAME)
        right = slice(frame * VARIABLES_PER_FRAME, (frame + 1) * VARIABLES_PER_FRAME)
        matrix[left, left] += config.lambda_vel * identity
        matrix[right, right] += config.lambda_vel * identity
        matrix[left, right] -= config.lambda_vel * identity
        matrix[right, left] -= config.lambda_vel * identity
    for frame in range(1, frame_count - 1):
        blocks = (frame - 1, frame, frame + 1)
        coeffs = (1.0, -2.0, 1.0)
        for a, coefficient_a in zip(blocks, coeffs):
            for b, coefficient_b in zip(blocks, coeffs):
                sa = slice(a * VARIABLES_PER_FRAME, (a + 1) * VARIABLES_PER_FRAME)
                sb = slice(b * VARIABLES_PER_FRAME, (b + 1) * VARIABLES_PER_FRAME)
                matrix[sa, sb] += (
                    config.lambda_acc * coefficient_a * coefficient_b * identity
                )
    return matrix


def solve_local_projection(
    metric: torch.Tensor,
    pelvis_jacobian: torch.Tensor,
    pelvis_target: torch.Tensor,
    contact_jacobian: torch.Tensor,
    contact_target: torch.Tensor,
    penetration_jacobian: torch.Tensor,
    penetration_target: torch.Tensor,
    *,
    contact_weight: float,
) -> tuple[torch.Tensor, dict[str, float | int | None]]:
    """Solve the equality-constrained local quadratic projection."""

    variables = metric.shape[0]
    if metric.shape != (variables, variables):
        raise ValueError("metric must be square")
    q = metric.clone()
    rhs = torch.zeros(variables, device=metric.device, dtype=metric.dtype)
    if contact_jacobian.numel():
        q = q + contact_weight * contact_jacobian.T @ contact_jacobian
        rhs = rhs + contact_weight * contact_jacobian.T @ contact_target
    equality_parts = [
        item for item in (pelvis_jacobian, penetration_jacobian) if item.numel()
    ]
    equality_target_parts = [
        item for item in (pelvis_target, penetration_target) if item.numel()
    ]
    equality = (
        torch.cat(equality_parts, dim=0)
        if equality_parts
        else torch.zeros((0, variables), device=metric.device, dtype=metric.dtype)
    )
    equality_target = (
        torch.cat(equality_target_parts, dim=0)
        if equality_target_parts
        else torch.zeros(0, device=metric.device, dtype=metric.dtype)
    )
    q = q + torch.eye(variables, device=q.device, dtype=q.dtype) * 1.0e-8
    if equality.numel():
        zeros = torch.zeros(
            (equality.shape[0], equality.shape[0]),
            device=q.device,
            dtype=q.dtype,
        )
        kkt = torch.cat(
            [
                torch.cat([q, equality.T], dim=1),
                torch.cat([equality, zeros], dim=1),
            ],
            dim=0,
        )
        solution = torch.linalg.lstsq(
            kkt, torch.cat([rhs, equality_target]), rcond=1.0e-7
        ).solution[:variables]
    else:
        solution = torch.linalg.lstsq(q, rhs, rcond=1.0e-7).solution
    diagnostic_jacobian = torch.cat(
        [
            item
            for item in (pelvis_jacobian, contact_jacobian, penetration_jacobian)
            if item.numel()
        ],
        dim=0,
    )
    if diagnostic_jacobian.numel():
        singular = torch.linalg.svdvals(diagnostic_jacobian)
        rank = int(
            torch.linalg.matrix_rank(diagnostic_jacobian, tol=1.0e-6).item()
        )
        condition = None
        if singular.numel() and float(singular[-1]) > 1.0e-12:
            condition = float((singular[0] / singular[-1]).item())
    else:
        rank = 0
        condition = None
    return solution, {"jacobian_rank": rank, "jacobian_condition": condition}


def _strict_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _strict_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _strict_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("strict JSON forbids NaN and Infinity")
    return value


def write_strict_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _strict_value(value),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


class PelvisContactFlowProjector:
    """Iteratively project one clean endpoint inside a frozen sampling run."""

    def __init__(
        self,
        *,
        config: ProjectorConfig | None = None,
        baseline_motion: torch.Tensor | None = None,
        contact_data: Mapping[str, Any] | None = None,
        target_dose: float = 0.0,
        motion_mean: torch.Tensor | None = None,
        motion_std: torch.Tensor | None = None,
    ) -> None:
        self.config = config or ProjectorConfig()
        self.config.validate()
        self.baseline_motion = baseline_motion
        self.contact_data = dict(contact_data or {})
        self.target_dose = float(target_dose)
        self.motion_mean = motion_mean
        self.motion_std = motion_std
        self.step_records: list[dict[str, Any]] = []
        self.last_result: ProjectionResult | None = None

    @classmethod
    def from_frozen_protocol(
        cls,
        *,
        protocol_root: Path,
        sample_id: str,
        side: str,
        baseline_motion_norm: torch.Tensor,
        valid_mask: torch.Tensor,
        motion_mean: torch.Tensor,
        motion_std: torch.Tensor,
        target_dose: float,
        config: ProjectorConfig,
        device: torch.device,
        model_path: Path | None = None,
        allow_m0_mismatch: bool = False,
    ) -> "PelvisContactFlowProjector":
        """Bind frozen v3.0.1 evidence to one sampling-projection run."""

        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        protocol_root = Path(protocol_root)
        protocol = json.loads(
            (protocol_root / "protocol.json").read_text(encoding="utf-8")
        )
        cases = [
            case
            for case in protocol.get("cases", [])
            if str(case.get("sample_id")) == str(sample_id)
        ]
        if len(cases) != 1:
            raise ValueError(f"frozen protocol has {len(cases)} cases for {sample_id}")
        case = cases[0]
        window = case["sides"][side]["stable_window"]
        if window.get("status") != "PASS" or not window.get("frames"):
            raise ValueError(f"frozen {side} stable window is not evaluable")
        patches = json.loads(
            (protocol_root / "foot_patches.json").read_text(encoding="utf-8")
        )
        frozen_m0 = torch.load(
            protocol_root / "m0_physical.pt", map_location="cpu", weights_only=True
        ).float()
        frozen_valid = torch.load(
            protocol_root / "valid_mask.pt", map_location="cpu", weights_only=True
        ).bool()
        row_index = int(case.get("row_index", case.get("source_index", 0)))
        if row_index < 0 or row_index >= frozen_m0.shape[0]:
            raise ValueError("frozen case row_index is outside m0_physical.pt")
        m0 = frozen_m0[row_index : row_index + 1].to(device)
        expected_valid = frozen_valid[row_index : row_index + 1].to(device)
        if valid_mask.shape != expected_valid.shape or not torch.equal(
            valid_mask.bool().to(device), expected_valid
        ):
            raise ValueError("current valid mask differs from frozen v3.0.1 mask")
        mean = motion_mean.to(device=device, dtype=torch.float32)
        std = motion_std.to(device=device, dtype=torch.float32)
        while mean.ndim < baseline_motion_norm.ndim:
            mean = mean.unsqueeze(-2)
            std = std.unsqueeze(-2)
        current_m0 = baseline_motion_norm.float().to(device) * std + mean
        direct_spans = (
            MOTION_LAYOUT.body_pose,
            MOTION_LAYOUT.root_rotation,
            MOTION_LAYOUT.root_translation,
        )
        direct_max = max(
            float(
                (current_m0[..., span] - m0[..., span])[
                    expected_valid
                ].abs().max().item()
            )
            for span in direct_spans
        )
        if direct_max > 2.0e-3 and not allow_m0_mismatch:
            raise ValueError(
                "current paired M0 differs from frozen v3.0.1 direct pose: "
                f"max={direct_max:.6g}"
            )
        if model_path is None:
            frozen_model_path = protocol.get("inputs", {}).get("smplx_model", {}).get("path")
            if frozen_model_path:
                model_path = Path(frozen_model_path)
            else:
                from motion_rep.smplx_utils import default_smpl_model_path

                model_path = Path(
                    default_smpl_model_path(
                        "smplx", Path(__file__).resolve().parents[1]
                    )
                )
        from smplx import SMPLX

        model = SMPLX(
            model_path=str(model_path),
            gender="neutral",
            num_betas=10,
            batch_size=len(window["frames"]),
            use_pca=False,
        ).to(device)
        projection_mask = torch.zeros(
            expected_valid.shape[-1], dtype=torch.bool, device=device
        )
        projection_mask[
            int(window["window_start"]) : int(window["window_end_exclusive"])
        ] = True
        contact_data = {
            "model": model,
            "model_path": str(model_path),
            "patches": patches,
            "projection_mask": projection_mask,
            "stable_masks": {
                name: torch.as_tensor(
                    case["sides"][name]["evidence"]["valid_masks"]["flat_contact"],
                    dtype=torch.bool,
                    device=device,
                )
                for name in ("left", "right")
            },
            "floor_height_m": {
                name: float(case["sides"][name]["evidence"]["floor_height_m"])
                for name in ("left", "right")
            },
            "sample_id": str(sample_id),
            "side": side,
            "window": window,
            "frozen_protocol": str(protocol_root),
            "baseline_direct_max_abs": direct_max,
            "m0_match_status": "PASS" if direct_max <= 2.0e-3 else "MISMATCH_ALLOWED",
            "allow_m0_mismatch": bool(allow_m0_mismatch),
        }
        return cls(
            config=config,
            baseline_motion=m0,
            contact_data=contact_data,
            target_dose=target_dose,
            motion_mean=motion_mean,
            motion_std=motion_std,
        )

    def slice(self, index: int) -> "PelvisContactFlowProjector":
        if self.baseline_motion is None:
            baseline = None
        elif self.baseline_motion.ndim == 3:
            baseline = self.baseline_motion[index : index + 1]
        else:
            baseline = self.baseline_motion
        mean = self.motion_mean
        std = self.motion_std
        if mean is not None and mean.ndim > 1:
            mean = mean[index : index + 1]
            std = std[index : index + 1] if std is not None else None
        return PelvisContactFlowProjector(
            config=self.config,
            baseline_motion=baseline,
            contact_data=self.contact_data,
            target_dose=self.target_dose,
            motion_mean=mean,
            motion_std=std,
        )

    def _physical(self, motion: torch.Tensor) -> torch.Tensor:
        if self.motion_mean is None or self.motion_std is None:
            return motion.float()
        mean = self.motion_mean.to(device=motion.device, dtype=torch.float32)
        std = self.motion_std.to(device=motion.device, dtype=torch.float32)
        while mean.ndim < motion.ndim:
            mean = mean.unsqueeze(-2)
            std = std.unsqueeze(-2)
        return motion.float() * std + mean

    def _normalised(self, physical: torch.Tensor) -> torch.Tensor:
        if self.motion_mean is None or self.motion_std is None:
            return physical
        mean = self.motion_mean.to(device=physical.device, dtype=torch.float32)
        std = self.motion_std.to(device=physical.device, dtype=torch.float32)
        while mean.ndim < physical.ndim:
            mean = mean.unsqueeze(-2)
            std = std.unsqueeze(-2)
        return (physical - mean) / std

    @staticmethod
    def _model_output(
        model: Any,
        body: torch.Tensor,
        root: torch.Tensor,
        translation: torch.Tensor,
    ) -> Any:
        frames = body.shape[0]
        if hasattr(model, "batch_size"):
            model.batch_size = frames
        body_aa = so3_log(body).reshape(frames, 63)
        root_aa = so3_log(root).reshape(frames, 3)

        def zeros(name: str) -> torch.Tensor:
            parameter = getattr(model, name)
            return torch.zeros(
                (frames, parameter.shape[-1]),
                device=body.device,
                dtype=body.dtype,
            )

        return model(
            body_pose=body_aa,
            global_orient=root_aa,
            transl=translation,
            betas=zeros("betas"),
            expression=zeros("expression"),
            left_hand_pose=zeros("left_hand_pose"),
            right_hand_pose=zeros("right_hand_pose"),
            jaw_pose=zeros("jaw_pose"),
            leye_pose=zeros("leye_pose"),
            reye_pose=zeros("reye_pose"),
            return_verts=True,
        )

    @staticmethod
    def _apply_increment(
        body: torch.Tensor,
        root: torch.Tensor,
        translation: torch.Tensor,
        increment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shaped = increment.reshape(body.shape[0], VARIABLES_PER_FRAME)
        updated_translation = translation + shaped[:, :3]
        rotations = shaped[:, 3:].reshape(body.shape[0], len(ROTATION_NAMES), 3)
        updated_root = so3_exp(rotations[:, 0]) @ root
        updated_body = body.clone()
        for column, body_index in enumerate(ACTIVE_BODY_INDICES, start=1):
            updated_body[:, body_index] = (
                so3_exp(rotations[:, column]) @ body[:, body_index]
            )
        return updated_body, updated_root, updated_translation

    def _residual_bundle(
        self,
        increment: torch.Tensor,
        *,
        body: torch.Tensor,
        root: torch.Tensor,
        translation: torch.Tensor,
        target_root: torch.Tensor,
        model: Any,
        patches: Mapping[str, Mapping[str, list[int]]],
        stable_masks: Mapping[str, torch.Tensor],
        anchors: Mapping[str, Mapping[str, torch.Tensor]],
        floors: Mapping[str, float],
        active_penetration: Mapping[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        next_body, next_root, next_translation = self._apply_increment(
            body, root, translation, increment
        )
        output = self._model_output(model, next_body, next_root, next_translation)
        pelvis = so3_log(target_root.transpose(-1, -2) @ next_root).reshape(-1)
        contact_parts: list[torch.Tensor] = []
        heel_values: list[torch.Tensor] = []
        toe_values: list[torch.Tensor] = []
        penetration_values: list[torch.Tensor] = []
        for side in ("left", "right"):
            heel, toe = patch_centres(output.vertices, patches[side])
            stable = stable_masks[side]
            heel_error = heel[stable] - anchors[side]["heel"][stable]
            toe_error = toe[stable] - anchors[side]["toe"][stable]
            if heel_error.numel():
                contact_parts.extend((heel_error.reshape(-1), toe_error.reshape(-1)))
                heel_values.append(heel_error)
                toe_values.append(toe_error)
            if active_penetration is not None:
                marker = torch.cat((heel[:, 2], toe[:, 2]))
                active = active_penetration[side]
                target_height = float(floors[side]) + self.config.penetration_epsilon_m
                if bool(active.any()):
                    penetration_values.append(marker[active] - target_height)
        contact = (
            torch.cat(contact_parts)
            if contact_parts
            else torch.zeros(0, device=body.device, dtype=body.dtype)
        )
        penetration = (
            torch.cat(penetration_values)
            if penetration_values
            else torch.zeros(0, device=body.device, dtype=body.dtype)
        )
        diagnostics = {
            "heel": torch.cat([value.reshape(-1, 3) for value in heel_values], dim=0)
            if heel_values
            else torch.zeros((0, 3), device=body.device, dtype=body.dtype),
            "toe": torch.cat([value.reshape(-1, 3) for value in toe_values], dim=0)
            if toe_values
            else torch.zeros((0, 3), device=body.device, dtype=body.dtype),
            "next_body": next_body,
            "next_root": next_root,
            "next_translation": next_translation,
            "vertices": output.vertices,
        }
        return pelvis, contact, penetration, diagnostics

    @staticmethod
    def _rms(value: torch.Tensor) -> float:
        if not value.numel():
            return 0.0
        return float(torch.sqrt(value.square().mean()).detach().cpu())

    def _residual_summary(
        self,
        pelvis: torch.Tensor,
        penetration: torch.Tensor,
        diagnostics: Mapping[str, torch.Tensor],
    ) -> dict[str, float | int | None]:
        return {
            "pelvis_geodesic_rms_deg": self._rms(pelvis) * 180.0 / math.pi,
            "heel_rms_m": self._rms(diagnostics["heel"]),
            "toe_rms_m": self._rms(diagnostics["toe"]),
            "penetration_active_rms_m": self._rms(penetration),
            "active_penetration_count": int(penetration.numel()),
        }

    def _violation(self, summary: Mapping[str, float | int | None]) -> float:
        values = (
            float(summary["pelvis_geodesic_rms_deg"] or 0.0)
            / self.config.pelvis_tolerance_deg,
            float(summary["heel_rms_m"] or 0.0) / self.config.contact_tolerance_m,
            float(summary["toe_rms_m"] or 0.0) / self.config.contact_tolerance_m,
            float(summary["penetration_active_rms_m"] or 0.0)
            / self.config.contact_tolerance_m,
        )
        return max(values)

    def project_clean_endpoint(
        self,
        clean_motion: torch.Tensor,
        m0_motion: torch.Tensor,
        contact_data: Mapping[str, Any],
        target_dose: float,
        sampling_state: Mapping[str, Any],
    ) -> ProjectionResult:
        """Project one batch-one endpoint with iterative relinearisation."""

        if clean_motion.ndim == 2:
            clean_motion = clean_motion.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False
        if clean_motion.ndim != 3 or clean_motion.shape[0] != 1 or clean_motion.shape[-1] != 276:
            raise ValueError("v0.1 projector requires batch-one [1,T,276]")
        valid_mask = torch.as_tensor(
            sampling_state["valid_mask"], device=clean_motion.device, dtype=torch.bool
        )
        if valid_mask.ndim == 2:
            valid_mask = valid_mask[0]
        projection_mask = torch.as_tensor(
            contact_data["projection_mask"],
            device=clean_motion.device,
            dtype=torch.bool,
        ) & valid_mask
        frame_indices = torch.nonzero(projection_mask, as_tuple=False).flatten()
        if not frame_indices.numel():
            raise ValueError("projection mask contains no valid frames")
        physical = self._physical(clean_motion)
        if m0_motion.ndim == 2:
            m0_motion = m0_motion.unsqueeze(0)
        m0_physical = self._physical(m0_motion) if bool(contact_data.get("m0_standardized", False)) else m0_motion.float()
        source = physical[0]
        base_body = decode_rot6d_safe(
            source[:, MOTION_LAYOUT.body_pose].reshape(source.shape[0], 21, 6)
        )[frame_indices]
        base_root = decode_rot6d_safe(source[:, MOTION_LAYOUT.root_rotation])[frame_indices]
        base_translation = source[:, MOTION_LAYOUT.root_translation][frame_indices]
        m0_source = m0_physical[0]
        m0_root = decode_rot6d_safe(m0_source[:, MOTION_LAYOUT.root_rotation])[frame_indices]
        target_root = target_root_rotation(m0_root, float(target_dose))
        model = contact_data["model"]
        patches = contact_data["patches"]
        stable_masks = {
            side: torch.as_tensor(
                contact_data["stable_masks"][side],
                device=clean_motion.device,
                dtype=torch.bool,
            )[frame_indices]
            for side in ("left", "right")
        }
        m0_body = decode_rot6d_safe(
            m0_source[:, MOTION_LAYOUT.body_pose].reshape(m0_source.shape[0], 21, 6)
        )[frame_indices]
        m0_translation = m0_source[:, MOTION_LAYOUT.root_translation][frame_indices]
        with torch.no_grad():
            m0_output = self._model_output(model, m0_body, m0_root, m0_translation)
            anchors = {}
            floors = {}
            active_penetration = {}
            for side in ("left", "right"):
                heel, toe = patch_centres(m0_output.vertices, patches[side])
                anchors[side] = {"heel": heel.detach(), "toe": toe.detach()}
                floors[side] = float(contact_data["floor_height_m"][side])
        zero = torch.zeros(
            (frame_indices.numel(), VARIABLES_PER_FRAME),
            device=clean_motion.device,
            dtype=torch.float32,
        )
        # Freeze the active set at each relinearisation from current marker
        # heights.  Concatenation order is heel then toe for each side.
        current_body, current_root, current_translation = (
            base_body.float(),
            base_root.float(),
            base_translation.float(),
        )
        records: list[dict[str, Any]] = []
        total_increment = torch.zeros_like(zero)
        previous_violation: float | None = None
        stalled = 0
        converged = False
        condition: float | None = None
        pre_summary: dict[str, float | int | None] | None = None
        final_summary: dict[str, float | int | None] | None = None
        active_count = 0

        for iteration in range(1, self.config.max_relinearization_iters + 1):
            with torch.no_grad():
                current_output = self._model_output(
                    model, current_body, current_root, current_translation
                )
                for side in ("left", "right"):
                    heel, toe = patch_centres(current_output.vertices, patches[side])
                    heights = torch.cat((heel[:, 2], toe[:, 2]))
                    active_penetration[side] = heights < (
                        floors[side] - self.config.penetration_epsilon_m
                    )

            def residual_vector(value: torch.Tensor) -> torch.Tensor:
                pelvis, contact, penetration, _ = self._residual_bundle(
                    value,
                    body=current_body,
                    root=current_root,
                    translation=current_translation,
                    target_root=target_root,
                    model=model,
                    patches=patches,
                    stable_masks=stable_masks,
                    anchors=anchors,
                    floors=floors,
                    active_penetration=active_penetration,
                )
                return torch.cat((pelvis, contact, penetration))

            with torch.enable_grad():
                pelvis0, contact0, penetration0, diagnostics0 = self._residual_bundle(
                    zero,
                    body=current_body,
                    root=current_root,
                    translation=current_translation,
                    target_root=target_root,
                    model=model,
                    patches=patches,
                    stable_masks=stable_masks,
                    anchors=anchors,
                    floors=floors,
                    active_penetration=active_penetration,
                )
                rows = (pelvis0.numel(), contact0.numel(), penetration0.numel())
                jacobian = autograd_jacobian(residual_vector, zero)
            summary_before = self._residual_summary(
                pelvis0, penetration0, diagnostics0
            )
            if pre_summary is None:
                pre_summary = dict(summary_before)
            p_end = rows[0]
            c_end = p_end + rows[1]
            pelvis_jacobian = jacobian[:p_end]
            contact_jacobian = jacobian[p_end:c_end]
            penetration_jacobian = jacobian[c_end:]
            metric = build_projection_metric(
                frame_indices.numel(),
                self.config,
                device=clean_motion.device,
                dtype=torch.float32,
            )
            flat_step, solve_diagnostics = solve_local_projection(
                metric,
                pelvis_jacobian,
                -pelvis0.detach(),
                contact_jacobian,
                -contact0.detach(),
                penetration_jacobian,
                -penetration0.detach(),
                contact_weight=self.config.contact_weight,
            )
            condition = solve_diagnostics["jacobian_condition"]
            proposed = project_increment_norms(
                flat_step.reshape_as(zero),
                max_joint_increment_deg=self.config.max_joint_increment_deg,
                max_root_translation_m=self.config.max_root_translation_m,
            )
            violation_before = self._violation(summary_before)
            accepted_alpha = None
            accepted_state = None
            accepted_summary = None
            for alpha in self.config.backtracking_alphas:
                candidate_increment = proposed * float(alpha)
                with torch.no_grad():
                    p1, _, g1, d1 = self._residual_bundle(
                        candidate_increment,
                        body=current_body,
                        root=current_root,
                        translation=current_translation,
                        target_root=target_root,
                        model=model,
                        patches=patches,
                        stable_masks=stable_masks,
                        anchors=anchors,
                        floors=floors,
                        active_penetration=active_penetration,
                    )
                    summary_after = self._residual_summary(p1, g1, d1)
                    if self._violation(summary_after) < violation_before - 1.0e-9:
                        accepted_alpha = float(alpha)
                        accepted_state = (
                            d1["next_body"].detach(),
                            d1["next_root"].detach(),
                            d1["next_translation"].detach(),
                        )
                        accepted_summary = summary_after
                        break
            if accepted_state is None:
                final_summary = summary_before
                records.append(
                    {
                        "relinearization_iter": iteration,
                        **summary_before,
                        **solve_diagnostics,
                        "backtracking_alpha": None,
                        "accepted": False,
                        "penetration_method": PENETRATION_METHOD,
                    }
                )
                break
            current_body, current_root, current_translation = accepted_state
            total_increment += proposed * float(accepted_alpha)
            final_summary = accepted_summary
            active_count = int(summary_before["active_penetration_count"] or 0)
            record = {
                "relinearization_iter": iteration,
                "pelvis_residual_before_deg": summary_before["pelvis_geodesic_rms_deg"],
                "pelvis_residual_after_deg": accepted_summary["pelvis_geodesic_rms_deg"],
                "heel_residual_before_m": summary_before["heel_rms_m"],
                "heel_residual_after_m": accepted_summary["heel_rms_m"],
                "toe_residual_before_m": summary_before["toe_rms_m"],
                "toe_residual_after_m": accepted_summary["toe_rms_m"],
                "penetration_before_m": summary_before["penetration_active_rms_m"],
                "penetration_after_m": accepted_summary["penetration_active_rms_m"],
                "delta_q_norm": float(torch.linalg.vector_norm(proposed).cpu()),
                "delta_root_translation": float(
                    torch.linalg.vector_norm(proposed[:, :3], dim=-1).max().cpu()
                ),
                "max_joint_increment_deg": float(
                    torch.linalg.vector_norm(
                        proposed[:, 3:].reshape(-1, len(ROTATION_NAMES), 3), dim=-1
                    ).max().cpu()
                    * 180.0
                    / math.pi
                ),
                **solve_diagnostics,
                "active_penetration_count": active_count,
                "backtracking_alpha": accepted_alpha,
                "accepted": True,
                "penetration_method": PENETRATION_METHOD,
            }
            records.append(record)
            violation_after = self._violation(accepted_summary)
            if (
                float(accepted_summary["pelvis_geodesic_rms_deg"] or 0.0)
                <= self.config.pelvis_tolerance_deg
                and float(accepted_summary["heel_rms_m"] or 0.0)
                <= self.config.contact_tolerance_m
                and float(accepted_summary["toe_rms_m"] or 0.0)
                <= self.config.contact_tolerance_m
                and float(accepted_summary["penetration_active_rms_m"] or 0.0)
                <= self.config.contact_tolerance_m
            ):
                converged = True
                break
            if previous_violation is not None:
                improvement = (previous_violation - violation_after) / max(
                    previous_violation, 1.0e-12
                )
                stalled = stalled + 1 if improvement < self.config.relative_improvement_tolerance else 0
            previous_violation = violation_after
            if stalled >= 2:
                break

        if pre_summary is None or final_summary is None:
            raise RuntimeError("projection produced no residual summary")
        rebuilt = physical.clone()
        rebuilt_body = decode_rot6d_safe(
            rebuilt[0, :, MOTION_LAYOUT.body_pose].reshape(rebuilt.shape[1], 21, 6)
        )
        rebuilt_body[frame_indices] = current_body
        rebuilt_root = decode_rot6d_safe(rebuilt[0, :, MOTION_LAYOUT.root_rotation])
        rebuilt_root[frame_indices] = current_root
        rebuilt[0, :, MOTION_LAYOUT.body_pose] = encode_rot6d(rebuilt_body).reshape(
            rebuilt.shape[1], 126
        )
        rebuilt[0, :, MOTION_LAYOUT.root_rotation] = encode_rot6d(rebuilt_root)
        rebuilt[0, frame_indices, MOTION_LAYOUT.root_translation] = current_translation
        rebuilt = authority_project(
            rebuilt,
            valid_mask=valid_mask.unsqueeze(0),
            output_dtype=torch.float32,
        ).physical_motion
        projected = self._normalised(rebuilt)
        finite = bool(torch.isfinite(projected).all())
        if not finite:
            projected = clean_motion.clone()
            converged = False
        result = ProjectionResult(
            projected_clean_motion=projected[0] if unbatched else projected,
            pre_residuals=pre_summary,
            post_residuals=final_summary,
            pelvis_residual=float(final_summary["pelvis_geodesic_rms_deg"] or 0.0),
            heel_residual=float(final_summary["heel_rms_m"] or 0.0),
            toe_residual=float(final_summary["toe_rms_m"] or 0.0),
            penetration_residual=float(
                final_summary["penetration_active_rms_m"] or 0.0
            ),
            delta_q_norm=float(torch.linalg.vector_norm(total_increment).cpu()),
            delta_root_translation_norm=float(
                torch.linalg.vector_norm(total_increment[:, :3], dim=-1).max().cpu()
            ),
            num_relinearization_iters=len(records),
            jacobian_condition=condition,
            active_penetration_constraints=active_count,
            converged=converged,
            finite=finite,
            records=records,
        )
        self.last_result = result
        return result

    def correct_velocity(
        self,
        *,
        x_sigma: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor | float,
        valid_mask: torch.Tensor,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sigma_value = float(torch.as_tensor(sigma).detach().cpu())
        diagnostics: dict[str, Any] = {
            "protocol": PROTOCOL_NAME,
            "method": METHOD_NAME,
            "metric": self.config.metric,
            "sigma": sigma_value,
            "active": False,
            "projection_enabled": False,
            "target_delta_deg": self.target_dose,
        }
        endpoint = predict_clean_endpoint(x_sigma.float(), velocity.float(), sigma_value)
        inactive = (
            not self.config.enabled
            or abs(self.target_dose) <= 1.0e-12
            or sigma_value <= 1.0e-12
            or sigma_value + 1.0e-8 < self.config.sigma_min
            or sigma_value - 1.0e-8 > self.config.sigma_max
        )
        if inactive:
            if return_trace:
                diagnostics["trace"] = {
                    "velocity_model": velocity.detach().float().clone(),
                    "x0_hat": endpoint.detach().clone(),
                    "x0_guided": endpoint.detach().clone(),
                    "x0_reconciled": endpoint.detach().clone(),
                }
            self.step_records.append(diagnostics)
            return velocity, diagnostics
        if self.baseline_motion is None or not self.contact_data:
            raise RuntimeError("active projection requires frozen M0 and contact data")
        try:
            # The flow sampler invokes hooks inside its bfloat16 autocast
            # region.  SMPL-X/LBS and the KKT solve are deliberately FP32;
            # disable autocast locally to avoid mixed Float/BFloat16 kernels.
            autocast_device = endpoint.device.type
            if autocast_device == "cuda":
                with torch.autocast(device_type="cuda", enabled=False):
                    result = self.project_clean_endpoint(
                        endpoint,
                        self.baseline_motion.to(endpoint.device),
                        self.contact_data,
                        self.target_dose,
                        {"valid_mask": valid_mask},
                    )
            else:
                result = self.project_clean_endpoint(
                    endpoint,
                    self.baseline_motion.to(endpoint.device),
                    self.contact_data,
                    self.target_dose,
                    {"valid_mask": valid_mask},
                )
            corrected = recompose_velocity(
                x_sigma.float(), result.projected_clean_motion, sigma_value
            ).to(dtype=velocity.dtype)
            diagnostics.update(
                {
                    "active": True,
                    "projection_enabled": True,
                    "accepted": result.finite,
                    "projection_converged": result.converged,
                    **result.diagnostics(),
                }
            )
        except Exception as exc:
            corrected = velocity
            diagnostics.update(
                {
                    "active": True,
                    "projection_enabled": True,
                    "accepted": False,
                    "projection_converged": False,
                    "rejected_reason": repr(exc),
                }
            )
        if return_trace:
            diagnostics["trace"] = {
                "velocity_model": velocity.detach().float().clone(),
                "x0_hat": endpoint.detach().clone(),
                "x0_guided": predict_clean_endpoint(
                    x_sigma.float(), corrected.float(), sigma_value
                ).detach(),
                "x0_reconciled": predict_clean_endpoint(
                    x_sigma.float(), corrected.float(), sigma_value
                ).detach(),
            }
        self.step_records.append(
            {key: value for key, value in diagnostics.items() if key != "trace"}
        )
        return corrected, diagnostics

    def finalize_outputs(self, official: torch.Tensor, valid_mask: torch.Tensor) -> ProjectionFinalOutputs:
        """Project the terminal clean endpoint, then rebuild the representation.

        The last active flow step can leave a small scheduler integration
        residual after its endpoint correction.  Reprojecting the terminal
        clean endpoint uses the same frozen contact constraints (rather than a
        separate pose target) and makes the saved candidate auditable against
        the same v3.0.1 gates.
        """

        physical = self._physical(official)
        terminal = self.project_clean_endpoint(
            official.float(),
            self.baseline_motion.to(device=official.device),
            self.contact_data,
            self.target_dose,
            {"valid_mask": valid_mask},
        )
        projected_norm = terminal.projected_clean_motion
        projected_physical = self._physical(projected_norm)
        rebuilt = authority_project(
            projected_physical,
            valid_mask=valid_mask.to(device=physical.device, dtype=torch.bool),
            output_dtype=torch.float32,
        ).physical_motion
        g0 = self._normalised(rebuilt)
        return ProjectionFinalOutputs(
            g0=g0,
            protocol=PROTOCOL_NAME,
            summary=self.protocol_record(),
        )

    def protocol_record(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "method": METHOD_NAME,
            "config": asdict(self.config),
            "target_delta_deg": self.target_dose,
            "active_joints": list(ACTIVE_JOINT_NAMES),
            "penetration_method": PENETRATION_METHOD,
            "case": {
                key: self.contact_data.get(key)
                for key in (
                    "sample_id",
                    "side",
                    "window",
                    "frozen_protocol",
                    "model_path",
                    "baseline_direct_max_abs",
                )
                if key in self.contact_data
            },
            "step_records": _strict_value(self.step_records),
        }


__all__ = [
    "ACTIVE_BODY_INDICES",
    "ACTIVE_JOINT_NAMES",
    "EUCLIDEAN_METRIC",
    "KINEMATIC_TEMPORAL_METRIC",
    "METHOD_NAME",
    "PENETRATION_METHOD",
    "PROTOCOL_NAME",
    "PelvisContactFlowProjector",
    "ProjectionResult",
    "ProjectorConfig",
    "VARIABLES_PER_FRAME",
    "autograd_jacobian",
    "build_projection_metric",
    "finite_difference_jacobian",
    "predict_clean_endpoint",
    "project_increment_norms",
    "recompose_velocity",
    "so3_exp",
    "so3_log",
    "solve_local_projection",
    "write_strict_json",
]
