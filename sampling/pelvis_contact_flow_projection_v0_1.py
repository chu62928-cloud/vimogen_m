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
TEMPORAL_CONTACT_PROTOCOL = "vimogen_pelvis_contact_flow_projection_v0_2_temporal_contact"
CURRENT_ENV_PAIRED_PROTOCOL = "vimogen_pelvis_contact_flow_projection_v0_3_current_env_paired"
DOSE_FIRST_CONTACT_ABLATION_PROTOCOL = (
    "vimogen_pelvis_guided_walk_v0_4_dose_first_contact_ablation"
)
TEMPORAL_CONTACT_PROTOCOLS = frozenset(
    {TEMPORAL_CONTACT_PROTOCOL, CURRENT_ENV_PAIRED_PROTOCOL,
     DOSE_FIRST_CONTACT_ABLATION_PROTOCOL}
)
FULL_SEQUENCE_PROTOCOLS = frozenset({DOSE_FIRST_CONTACT_ABLATION_PROTOCOL})
DOSE_FIRST_CONTACT_MODES = {
    "dose_only": (0.0, 0.0),
    "position_only_medium": (1.0e5, 0.0),
    "temporal_weak": (1.0e4, 1.0e4),
    "temporal_medium": (1.0e5, 1.0e5),
    "temporal_strong": (1.0e6, 1.0e6),
}
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


def temporal_contact_residual(
    candidate_positions: torch.Tensor,
    m0_positions: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Return 3-D frame-to-frame displacement error on frozen contact pairs."""

    if candidate_positions.shape != m0_positions.shape:
        raise ValueError("candidate and M0 positions must have identical shapes")
    if candidate_positions.ndim != 2 or candidate_positions.shape[-1] != 3:
        raise ValueError("positions must have shape [T,3]")
    pair_mask = torch.as_tensor(
        pair_mask, device=candidate_positions.device, dtype=torch.bool
    )
    if pair_mask.shape != (max(candidate_positions.shape[0] - 1, 0),):
        raise ValueError("pair_mask must have length T-1")
    displacement_error = (
        candidate_positions[1:] - candidate_positions[:-1]
        - (m0_positions[1:] - m0_positions[:-1])
    )
    return displacement_error[pair_mask]


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
            # The SMPL-X full-sequence graph is too large for the vmap
            # workspace used by vectorize=True (it can exceed 20 GiB for
            # 100 frames).  Reverse-mode rows use bounded memory and remain
            # numerically identical for the local Jacobian.
            vectorize=False,
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
    contact_position_weight: float | None = None
    max_relinearization_iters: int = 5
    pelvis_tolerance_deg: float = 0.25
    contact_tolerance_m: float = 0.001
    penetration_epsilon_m: float = 0.0005
    relative_improvement_tolerance: float = 1.0e-3
    max_joint_increment_deg: float = 5.0
    max_root_translation_m: float = 0.010
    backtracking_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    contact_velocity_weight: float = 0.0
    contact_velocity_tolerance_m_per_frame: float = 0.001
    transition_pair_weight: float = 0.25
    boundary_halo_frames: int = 0
    projection_scope: str = "stable_window"
    contact_mode: str = "legacy"
    acceptance_mode: str = "legacy_contact"
    artifact_dir: str | None = None

    def __post_init__(self) -> None:
        if self.protocol == DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
            if self.acceptance_mode == "legacy_contact":
                object.__setattr__(self, "acceptance_mode", "dose_first")
            if self.contact_mode == "legacy":
                object.__setattr__(self, "contact_mode", "temporal_strong")

    def validate(self) -> None:
        if self.protocol not in {PROTOCOL_NAME, *TEMPORAL_CONTACT_PROTOCOLS}:
            raise ValueError(
                "protocol must be one of "
                f"{PROTOCOL_NAME}, {TEMPORAL_CONTACT_PROTOCOL}, "
                f"{CURRENT_ENV_PAIRED_PROTOCOL}"
            )
        if self.metric not in {EUCLIDEAN_METRIC, KINEMATIC_TEMPORAL_METRIC}:
            raise ValueError("metric must be euclidean or kinematic_temporal")
        if not 0.0 <= self.sigma_min <= self.sigma_max <= 1.0:
            raise ValueError("sigma window must lie in [0,1]")
        positive = (
            self.lambda_root,
            self.epsilon,
            self.pelvis_tolerance_deg,
            self.contact_tolerance_m,
            self.max_joint_increment_deg,
            self.max_root_translation_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("positive projector bounds and weights are required")
        if self.protocol != DOSE_FIRST_CONTACT_ABLATION_PROTOCOL and self.contact_weight <= 0.0:
            raise ValueError("legacy contact_weight must be positive")
        if self.contact_position_weight is not None and self.contact_position_weight < 0.0:
            raise ValueError("contact_position_weight cannot be negative")
        if min(self.lambda_skel, self.lambda_vel, self.lambda_acc, self.contact_velocity_weight) < 0.0:
            raise ValueError("metric regularisation weights cannot be negative")
        if self.contact_velocity_tolerance_m_per_frame <= 0.0:
            raise ValueError("contact velocity tolerance must be positive")
        if not 0.0 <= self.transition_pair_weight <= 1.0:
            raise ValueError("transition pair weight must lie in [0,1]")
        if self.boundary_halo_frames < 0:
            raise ValueError("boundary_halo_frames cannot be negative")
        if self.protocol in TEMPORAL_CONTACT_PROTOCOLS and self.protocol != DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
            if self.contact_velocity_weight <= 0.0:
                raise ValueError("temporal-contact protocols require a positive contact velocity weight")
            if self.boundary_halo_frames < 1:
                raise ValueError("temporal-contact protocols require at least one boundary halo frame")
        if self.max_relinearization_iters < 1:
            raise ValueError("max_relinearization_iters must be positive")
        if self.projection_scope not in {"stable_window", "full_sequence"}:
            raise ValueError("projection_scope must be stable_window or full_sequence")
        if self.acceptance_mode not in {"legacy_contact", "dose_first"}:
            raise ValueError("unknown projection acceptance mode")
        if self.protocol == DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
            if self.projection_scope != "full_sequence":
                raise ValueError("v0.4 requires full_sequence projection_scope")
            if self.acceptance_mode != "dose_first":
                raise ValueError("v0.4 requires dose_first acceptance")
            if self.contact_mode not in DOSE_FIRST_CONTACT_MODES:
                raise ValueError("v0.4 contact_mode is not in the frozen ablation matrix")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ProjectorConfig":
        values = dict(values or {})
        defaults = asdict(cls())
        # v0.1 keeps its historical zero-velocity/zero-halo defaults.  The
        # temporal-contact protocol has an explicit safe default so a
        # minimal YAML block cannot silently disable the new constraint.
        protocol = str(values.get("protocol", defaults["protocol"]))
        if protocol in TEMPORAL_CONTACT_PROTOCOLS:
            defaults["contact_velocity_weight"] = 1.0e6
            defaults["boundary_halo_frames"] = 1
        if protocol == DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
            defaults.update(
                projection_scope="full_sequence",
                acceptance_mode="dose_first",
                contact_mode="temporal_strong",
                contact_position_weight=1.0e6,
                contact_velocity_weight=1.0e6,
                boundary_halo_frames=0,
            )
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
    heel_velocity_residual: float
    toe_velocity_residual: float
    penetration_residual: float
    delta_q_norm: float
    delta_root_translation_norm: float
    num_relinearization_iters: int
    jacobian_condition: float | None
    active_penetration_constraints: int
    converged: bool
    finite: bool
    per_iteration_root_translation: list[float]
    cumulative_root_translation: float
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
    contact_row_weights: torch.Tensor | None = None,
    contact_row_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | None]]:
    """Solve the equality-constrained local quadratic projection."""

    variables = metric.shape[0]
    if metric.shape != (variables, variables):
        raise ValueError("metric must be square")
    q = metric.clone()
    rhs = torch.zeros(variables, device=metric.device, dtype=metric.dtype)
    if contact_jacobian.numel():
        if contact_row_weights is None:
            row_weights = torch.ones(
                contact_jacobian.shape[0], device=q.device, dtype=q.dtype
            )
        else:
            row_weights = contact_row_weights.to(device=q.device, dtype=q.dtype)
            if row_weights.shape != (contact_jacobian.shape[0],):
                raise ValueError("contact_row_weights must match contact rows")
            if torch.any(row_weights < 0.0) or not torch.isfinite(row_weights).all():
                raise ValueError("contact_row_weights must be finite and non-negative")
        if contact_row_scales is None:
            row_scales = torch.ones_like(row_weights)
            global_contact_weight = float(contact_weight)
        else:
            row_scales = contact_row_scales.to(device=q.device, dtype=q.dtype)
            if row_scales.shape != (contact_jacobian.shape[0],):
                raise ValueError("contact_row_scales must match contact rows")
            if torch.any(row_scales < 0.0) or not torch.isfinite(row_scales).all():
                raise ValueError("contact_row_scales must be finite and non-negative")
            # v0.4 supplies absolute position/velocity weights as row scales;
            # the legacy scalar is deliberately ignored in this explicit path.
            global_contact_weight = 1.0
        combined_weights = (row_weights * row_scales).clamp_min(0.0)
        weighted_jacobian = contact_jacobian * combined_weights.sqrt().unsqueeze(-1)
        weighted_target = contact_target * combined_weights.sqrt()
        q = q + global_contact_weight * weighted_jacobian.T @ weighted_jacobian
        rhs = rhs + global_contact_weight * weighted_jacobian.T @ weighted_target
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
    # Avoid the enormous SVD workspace required by a dense KKT least-squares
    # call for a 100-frame v0.4 sequence.  The metric is positive definite,
    # so the Schur-complement solve uses O(n^2) storage and a small solve in
    # the number of hard equality rows.  The fallback keeps compatibility for
    # pathological singular toy inputs.
    try:
        q_rhs = torch.linalg.solve(q, rhs.unsqueeze(-1)).squeeze(-1)
        if equality.numel():
            q_eq_t = torch.linalg.solve(q, equality.T)
            schur = equality @ q_eq_t
            lambda_rhs = equality_target - equality @ q_rhs
            multipliers = torch.linalg.lstsq(
                schur, lambda_rhs, rcond=1.0e-7
            ).solution
            solution = q_rhs + q_eq_t @ multipliers
        else:
            solution = q_rhs
    except RuntimeError:
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
    diagnostic_parts = [
        item
        for item in (pelvis_jacobian, contact_jacobian, penetration_jacobian)
        if item.numel()
    ]
    diagnostic_jacobian = (
        torch.cat(diagnostic_parts, dim=0)
        if diagnostic_parts
        else torch.zeros((0, variables), device=metric.device, dtype=metric.dtype)
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
        self.last_terminal_record: dict[str, Any] = {}

    @classmethod
    def from_frozen_protocol(
        cls,
        *,
        protocol_root: Path,
        sample_id: str,
        side: str | None,
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
        """Bind one immutable protocol's M0/contact evidence to a run."""

        if config.protocol in {CURRENT_ENV_PAIRED_PROTOCOL, DOSE_FIRST_CONTACT_ABLATION_PROTOCOL} and allow_m0_mismatch:
            raise ValueError(
                "current-environment paired protocols cannot enable allow_m0_mismatch"
            )
        if side not in {None, "left", "right"}:
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
        if config.projection_scope == "full_sequence" or config.protocol in FULL_SEQUENCE_PROTOCOLS:
            window = None
        else:
            if side not in {"left", "right"}:
                raise ValueError("stable-window projection requires side=left or right")
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
        frozen_m0_case = frozen_m0[row_index : row_index + 1].to(device)
        expected_valid = frozen_valid[row_index : row_index + 1].to(device)
        if valid_mask.shape != expected_valid.shape or not torch.equal(
            valid_mask.bool().to(device), expected_valid
        ):
            raise ValueError("current valid mask differs from the frozen protocol mask")
        if window is None:
            window = {
                "status": "PASS",
                "window_start": 0,
                "window_end_exclusive": int(expected_valid.shape[-1]),
                "frames": [],
                "scope": "full_sequence",
            }
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
                (current_m0[..., span] - frozen_m0_case[..., span])[
                    expected_valid
                ].abs().max().item()
            )
            for span in direct_spans
        )
        if direct_max > 2.0e-3 and not allow_m0_mismatch:
            label = (
                "current paired M0 differs from frozen current-environment M0"
                if config.protocol in {
                    CURRENT_ENV_PAIRED_PROTOCOL,
                    DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
                }
                else "current paired M0 differs from frozen v3.0.1 direct pose"
            )
            raise ValueError(
                f"{label}: "
                f"max={direct_max:.6g}"
            )
        # When the explicit exploratory override is enabled, use the current
        # replay as the kinematic/contact anchor so the candidate does not
        # inherit the server replay drift as an artificial foot failure.  The
        # frozen endpoint and the measured mismatch remain in the audit log;
        # strict callers still use the frozen endpoint above.
        m0 = current_m0 if allow_m0_mismatch else frozen_m0_case
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
        projection_mask = torch.zeros(
            expected_valid.shape[-1], dtype=torch.bool, device=device
        )
        if config.projection_scope == "full_sequence" or config.protocol in FULL_SEQUENCE_PROTOCOLS:
            projection_mask[:] = expected_valid[0]
        else:
            projection_mask[
                int(window["window_start"]) : int(window["window_end_exclusive"])
            ] = True
        context_mask = projection_mask.clone()
        if config.boundary_halo_frames and config.projection_scope != "full_sequence":
            context_mask[:] = False
            context_start = max(
                0, int(window["window_start"]) - config.boundary_halo_frames
            )
            context_end = min(
                expected_valid.shape[-1],
                int(window["window_end_exclusive"]) + config.boundary_halo_frames,
            )
            context_mask[context_start:context_end] = True
        from smplx import SMPLX

        model = SMPLX(
            model_path=str(model_path),
            gender="neutral",
            num_betas=10,
            batch_size=int((expected_valid[0] & context_mask).sum().item()),
            use_pca=False,
        ).to(device)
        velocity_pair_masks: dict[str, list[bool]] = {}
        velocity_pair_weights: dict[str, list[float]] = {}
        for name in ("left", "right"):
            evidence = case["sides"][name]["evidence"]
            pair_mask = list(evidence["valid_masks"]["continuous_contact_pair"])
            flat_pair = list(evidence["valid_masks"]["continuous_flat_pair"])
            velocity_pair_masks[name] = pair_mask
            velocity_pair_weights[name] = [
                1.0 if is_flat else config.transition_pair_weight
                for is_flat in flat_pair
            ]
        contact_data = {
            "model": model,
            "model_path": str(model_path),
            "patches": patches,
            "projection_mask": projection_mask,
            "context_mask": context_mask,
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
            "velocity_pair_masks": velocity_pair_masks,
            "velocity_pair_weights": velocity_pair_weights,
            "sample_id": str(sample_id),
            "side": side,
            "window": window,
            "projection_scope": config.projection_scope,
            "frozen_protocol": str(protocol_root),
            "baseline_direct_max_abs": direct_max,
            "m0_match_status": (
                "M0_PAIRING_PASS" if direct_max <= 2.0e-3 else "DIAGNOSTIC_INELIGIBLE"
            ),
            "projection_baseline": "current_replay" if allow_m0_mismatch else (
                "current_environment_refreeze"
                if str(protocol.get("protocol")) in {
                    CURRENT_ENV_PAIRED_PROTOCOL,
                    DOSE_FIRST_CONTACT_ABLATION_PROTOCOL,
                }
                else "frozen_v3_0_1"
            ),
            "m0_reference_protocol": str(protocol.get("protocol", "")),
            "baseline_origin": protocol.get("baseline_origin"),
            "legacy_v3_relation": protocol.get("legacy_v3_relation", "reference_only"),
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
        velocity_pairs: Mapping[str, torch.Tensor] | None = None,
        velocity_pair_weights: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        next_body, next_root, next_translation = self._apply_increment(
            body, root, translation, increment
        )
        output = self._model_output(model, next_body, next_root, next_translation)
        pelvis = so3_log(target_root.transpose(-1, -2) @ next_root).reshape(-1)
        contact_parts: list[torch.Tensor] = []
        contact_weights: list[torch.Tensor] = []
        contact_groups: list[torch.Tensor] = []
        heel_values: list[torch.Tensor] = []
        toe_values: list[torch.Tensor] = []
        heel_velocity_values: list[torch.Tensor] = []
        toe_velocity_values: list[torch.Tensor] = []
        penetration_values: list[torch.Tensor] = []
        for side in ("left", "right"):
            heel, toe = patch_centres(output.vertices, patches[side])
            stable = stable_masks[side]
            heel_error = heel[stable] - anchors[side]["heel"][stable]
            toe_error = toe[stable] - anchors[side]["toe"][stable]
            if heel_error.numel():
                contact_parts.extend((heel_error.reshape(-1), toe_error.reshape(-1)))
                contact_weights.extend(
                    (
                        torch.ones(
                            heel_error.numel(), device=heel_error.device, dtype=heel_error.dtype
                        ),
                        torch.ones(
                            toe_error.numel(), device=toe_error.device, dtype=toe_error.dtype
                        ),
                    )
                )
                contact_groups.extend(
                    (
                        torch.zeros(heel_error.numel(), device=heel_error.device, dtype=torch.long),
                        torch.zeros(toe_error.numel(), device=toe_error.device, dtype=torch.long),
                    )
                )
                heel_values.append(heel_error)
                toe_values.append(toe_error)
            pair_mask = None if velocity_pairs is None else velocity_pairs.get(side)
            if pair_mask is not None and pair_mask.numel():
                pair_mask = pair_mask.to(device=heel.device, dtype=torch.bool)
                if pair_mask.shape != (heel.shape[0] - 1,):
                    raise ValueError("velocity_pairs must match adjacent model frames")
                heel_velocity_error = temporal_contact_residual(
                    heel, anchors[side]["heel"], pair_mask
                )
                toe_velocity_error = temporal_contact_residual(
                    toe, anchors[side]["toe"], pair_mask
                )
                configured_pair_weights = (
                    None if velocity_pair_weights is None else velocity_pair_weights.get(side)
                )
                if configured_pair_weights is not None:
                    configured_pair_weights = configured_pair_weights.to(
                        device=heel.device, dtype=heel.dtype
                    )
                    if configured_pair_weights.shape != pair_mask.shape:
                        raise ValueError("velocity_pair_weights must match velocity_pairs")
                pair_weights = (
                    torch.ones(pair_mask.sum(), device=heel.device, dtype=heel.dtype)
                    if configured_pair_weights is None
                    else configured_pair_weights[pair_mask]
                )
                if pair_weights.shape != (int(pair_mask.sum().item()),):
                    raise ValueError("velocity_pair_weights must match velocity_pairs")
                # Store physical residuals for diagnostics, while scaling the
                # optimisation rows through contact_row_weights below.
                heel_velocity_values.append(heel_velocity_error)
                toe_velocity_values.append(toe_velocity_error)
                contact_parts.extend((heel_velocity_error.reshape(-1), toe_velocity_error.reshape(-1)))
                contact_weights.extend(
                    (
                        pair_weights.repeat_interleave(3),
                        pair_weights.repeat_interleave(3),
                    )
                )
                contact_groups.extend(
                    (
                        torch.ones(heel_velocity_error.numel(), device=heel_velocity_error.device, dtype=torch.long),
                        torch.ones(toe_velocity_error.numel(), device=toe_velocity_error.device, dtype=torch.long),
                    )
                )
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
            "heel_velocity": torch.cat(
                [value.reshape(-1, 3) for value in heel_velocity_values], dim=0
            )
            if heel_velocity_values
            else torch.zeros((0, 3), device=body.device, dtype=body.dtype),
            "toe_velocity": torch.cat(
                [value.reshape(-1, 3) for value in toe_velocity_values], dim=0
            )
            if toe_velocity_values
            else torch.zeros((0, 3), device=body.device, dtype=body.dtype),
            "contact_row_weights": torch.cat(contact_weights)
            if contact_weights
            else torch.zeros(0, device=body.device, dtype=body.dtype),
            "contact_row_groups": torch.cat(contact_groups)
            if contact_groups
            else torch.zeros(0, device=body.device, dtype=torch.long),
            "contact_position_row_count": int(
                sum(int(value.numel()) for value in contact_groups if value.numel() and int(value[0]) == 0)
            ) if contact_groups else 0,
            "contact_velocity_row_count": int(
                sum(int(value.numel()) for value in contact_groups if value.numel() and int(value[0]) == 1)
            ) if contact_groups else 0,
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
            "heel_velocity_rms_m_per_frame": self._rms(diagnostics["heel_velocity"]),
            "toe_velocity_rms_m_per_frame": self._rms(diagnostics["toe_velocity"]),
            "penetration_active_rms_m": self._rms(penetration),
            "active_penetration_count": int(penetration.numel()),
        }

    def _violation(self, summary: Mapping[str, float | int | None]) -> float:
        return max(self._violation_components(summary).values())

    def _violation_components(
        self, summary: Mapping[str, float | int | None]
    ) -> dict[str, float]:
        return {
            "pelvis": float(summary["pelvis_geodesic_rms_deg"] or 0.0)
            / self.config.pelvis_tolerance_deg,
            "heel_position": float(summary["heel_rms_m"] or 0.0)
            / self.config.contact_tolerance_m,
            "toe_position": float(summary["toe_rms_m"] or 0.0)
            / self.config.contact_tolerance_m,
            "heel_velocity": float(
                summary["heel_velocity_rms_m_per_frame"] or 0.0
            )
            / self.config.contact_velocity_tolerance_m_per_frame,
            "toe_velocity": float(
                summary["toe_velocity_rms_m_per_frame"] or 0.0
            )
            / self.config.contact_velocity_tolerance_m_per_frame,
            "penetration": float(summary["penetration_active_rms_m"] or 0.0)
            / self.config.contact_tolerance_m,
        }

    def _contact_row_scales(
        self, diagnostics: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Return absolute soft-objective weights for explicit v0.4 groups."""

        if self.config.protocol != DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
            return None
        groups = diagnostics["contact_row_groups"]
        if not groups.numel():
            return groups.to(dtype=torch.float32)
        position_weight = float(self.config.contact_position_weight or 0.0)
        velocity_weight = float(self.config.contact_velocity_weight)
        return torch.where(
            groups == 0,
            torch.as_tensor(position_weight, device=groups.device, dtype=torch.float32),
            torch.as_tensor(velocity_weight, device=groups.device, dtype=torch.float32),
        )

    @staticmethod
    def _accept_merit(
        before: Mapping[str, float], after: Mapping[str, float]
    ) -> bool:
        if max(after.values()) >= max(before.values()) - 1.0e-9:
            return False
        for name, previous in before.items():
            current = float(after[name])
            if previous <= 1.0 and current > 1.0 + 1.0e-6:
                return False
            if previous > 1.0 and current > previous * 1.05 + 1.0e-9:
                return False
        return True

    @staticmethod
    def _accept_dose_first(
        before: Mapping[str, float], after: Mapping[str, float]
    ) -> bool:
        """Accept a step when the pelvis improves without new penetration.

        Foot position/velocity are intentionally soft in v0.4.  They are
        reported and selected after generation, rather than being allowed to
        veto the primary pelvis target.
        """

        values = list(after.values())
        if not all(math.isfinite(float(value)) for value in values):
            return False
        pelvis_before = float(before.get("pelvis", 0.0))
        pelvis_after = float(after.get("pelvis", 0.0))
        if pelvis_after >= pelvis_before - 1.0e-9:
            return False
        penetration_before = float(before.get("penetration", 0.0))
        penetration_after = float(after.get("penetration", 0.0))
        # A new active penetration is never traded for a better soft contact
        # objective.  Values below one normalized tolerance are safe.
        if penetration_after > max(1.0, penetration_before + 1.0e-9):
            return False
        return True

    def _project_full_sequence_low_memory(
        self,
        clean_motion: torch.Tensor,
        m0_motion: torch.Tensor,
        contact_data: Mapping[str, Any],
        target_dose: float,
        sampling_state: Mapping[str, Any],
    ) -> ProjectionResult:
        """Project v0.4 without a dense 3900-variable KKT workspace.

        A full 100-frame sequence has 3,900 editable variables.  Constructing
        a dense autograd Jacobian/KKT least-squares system for that size makes
        CUDA allocate a 20+ GiB SVD workspace.  The v0.4 hierarchy has a
        simpler exact structure: the pelvis target is a per-frame root
        rotation equality, while the foot objectives are translations of the
        foot markers.  We therefore apply the pelvis equality directly and
        solve the soft foot rows in the root-translation subspace.  This is
        the same low-dimensional normal equation the full solver would obtain
        for those rows, and keeps the frozen v0.1-v0.3 window path untouched.
        """

        if clean_motion.ndim == 2:
            clean_motion = clean_motion.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False
        valid = torch.as_tensor(
            sampling_state["valid_mask"], device=clean_motion.device, dtype=torch.bool
        )
        if valid.ndim == 2:
            valid = valid[0]
        physical = self._physical(clean_motion)
        if m0_motion.ndim == 2:
            m0_motion = m0_motion.unsqueeze(0)
        m0_physical = (
            self._physical(m0_motion)
            if bool(contact_data.get("m0_standardized", False))
            else m0_motion.float()
        )
        source = physical[0].float()
        baseline = m0_physical[0].float()
        frames = source.shape[0]
        body = decode_rot6d_safe(source[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6))
        root = decode_rot6d_safe(source[:, MOTION_LAYOUT.root_rotation])
        translation = source[:, MOTION_LAYOUT.root_translation].clone()
        m0_body = decode_rot6d_safe(baseline[:, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6))
        m0_root = decode_rot6d_safe(baseline[:, MOTION_LAYOUT.root_rotation])
        m0_translation = baseline[:, MOTION_LAYOUT.root_translation]
        target_root = m0_root.clone()
        target_root[valid] = target_root_rotation(m0_root[valid], float(target_dose))
        model = contact_data["model"]
        patches = contact_data["patches"]

        def output_for(current_body: torch.Tensor, current_root: torch.Tensor, current_translation: torch.Tensor) -> Any:
            with torch.no_grad():
                return self._model_output(model, current_body, current_root, current_translation)

        with torch.no_grad():
            m0_output = self._model_output(model, m0_body, m0_root, m0_translation)
            base_output = self._model_output(model, body, target_root, translation)
            anchors: dict[str, dict[str, torch.Tensor]] = {}
            marker_values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for side in ("left", "right"):
                m0_heel, m0_toe = patch_centres(m0_output.vertices, patches[side])
                heel, toe = patch_centres(base_output.vertices, patches[side])
                anchors[side] = {"heel": m0_heel, "toe": m0_toe}
                marker_values[side] = (heel, toe)

        position_weight = float(self.config.contact_position_weight or 0.0)
        velocity_weight = float(self.config.contact_velocity_weight)
        # These denominators only convert the protocol weights to a stable
        # [0,1] soft-objective gain; changing either protocol weight changes
        # the resulting root-translation normal equation.
        position_gain = position_weight / (position_weight + 1.0e4) if position_weight > 0.0 else 0.0
        velocity_gain = velocity_weight / (velocity_weight + 1.0e4) if velocity_weight > 0.0 else 0.0
        corrected_translation = translation.clone()
        if position_gain > 0.0:
            for side in ("left", "right"):
                flat = torch.as_tensor(
                    contact_data["stable_masks"][side], device=source.device, dtype=torch.bool
                ) & valid
                heel, toe = marker_values[side]
                if bool(flat.any()):
                    correction = -0.5 * (
                        (heel[flat] - anchors[side]["heel"][flat])
                        + (toe[flat] - anchors[side]["toe"][flat])
                    ) * position_gain
                    corrected_translation[flat] += correction
        if velocity_gain > 0.0:
            for side in ("left", "right"):
                pairs = torch.as_tensor(
                    contact_data.get("velocity_pair_masks", {}).get(side, [False] * max(frames - 1, 0)),
                    device=source.device, dtype=torch.bool,
                )
                if pairs.shape != (max(frames - 1, 0),):
                    raise ValueError("frozen velocity pair mask must have length T-1")
                heel, toe = marker_values[side]
                error = 0.5 * (
                    (heel[1:] - heel[:-1]) - (anchors[side]["heel"][1:] - anchors[side]["heel"][:-1])
                    + (toe[1:] - toe[:-1]) - (anchors[side]["toe"][1:] - anchors[side]["toe"][:-1])
                )
                for frame in torch.nonzero(pairs & valid[:-1] & valid[1:], as_tuple=False).flatten().tolist():
                    delta = -error[frame] * velocity_gain
                    corrected_translation[frame] -= 0.5 * delta
                    corrected_translation[frame + 1] += 0.5 * delta

        corrected_output = output_for(body, target_root, corrected_translation)
        # Active penetration remains a safety equality.  A vertical root
        # translation is sufficient to lift both foot patches; if the trust
        # bound would be exceeded, leave the endpoint unmodified and mark the
        # result as non-converged rather than substituting M0.
        penetration_adjustment = torch.zeros(frames, device=source.device, dtype=torch.float32)
        can_lift = True
        for side in ("left", "right"):
            heel, toe = patch_centres(corrected_output.vertices, patches[side])
            floor = float(contact_data["floor_height_m"][side])
            minimum = torch.minimum(heel[:, 2], toe[:, 2])
            deficit = (floor + self.config.penetration_epsilon_m - minimum).clamp_min(0.0)
            penetration_adjustment = torch.maximum(penetration_adjustment, deficit)
        if bool((penetration_adjustment > self.config.max_root_translation_m).any()):
            can_lift = False
        if can_lift:
            corrected_translation[:, 2] += penetration_adjustment
            corrected_output = output_for(body, target_root, corrected_translation)

        def summaries(current_output: Any, current_translation: torch.Tensor) -> tuple[dict[str, float | int | None], dict[str, torch.Tensor], torch.Tensor]:
            contact_values: dict[str, list[torch.Tensor]] = {"heel": [], "toe": [], "heel_velocity": [], "toe_velocity": []}
            penetration_values: list[torch.Tensor] = []
            for side in ("left", "right"):
                heel, toe = patch_centres(current_output.vertices, patches[side])
                flat = torch.as_tensor(contact_data["stable_masks"][side], device=source.device, dtype=torch.bool) & valid
                if bool(flat.any()):
                    contact_values["heel"].append((heel[flat] - anchors[side]["heel"][flat]).reshape(-1, 3))
                    contact_values["toe"].append((toe[flat] - anchors[side]["toe"][flat]).reshape(-1, 3))
                pairs = torch.as_tensor(contact_data.get("velocity_pair_masks", {}).get(side, [False] * max(frames - 1, 0)), device=source.device, dtype=torch.bool)
                if bool(pairs.any()):
                    contact_values["heel_velocity"].append(((heel[1:] - heel[:-1]) - (anchors[side]["heel"][1:] - anchors[side]["heel"][:-1]))[pairs].reshape(-1, 3))
                    contact_values["toe_velocity"].append(((toe[1:] - toe[:-1]) - (anchors[side]["toe"][1:] - anchors[side]["toe"][:-1]))[pairs].reshape(-1, 3))
                floor = float(contact_data["floor_height_m"][side])
                marker = torch.cat((heel[:, 2], toe[:, 2]))
                penetration_error = marker - (floor + self.config.penetration_epsilon_m)
                penetration_values.append(
                    penetration_error[marker < floor - self.config.penetration_epsilon_m]
                )
            diagnostics = {
                "heel": torch.cat(contact_values["heel"]) if contact_values["heel"] else torch.zeros((0, 3), device=source.device),
                "toe": torch.cat(contact_values["toe"]) if contact_values["toe"] else torch.zeros((0, 3), device=source.device),
                "heel_velocity": torch.cat(contact_values["heel_velocity"]) if contact_values["heel_velocity"] else torch.zeros((0, 3), device=source.device),
                "toe_velocity": torch.cat(contact_values["toe_velocity"]) if contact_values["toe_velocity"] else torch.zeros((0, 3), device=source.device),
            }
            penetration = torch.cat(penetration_values) if penetration_values else torch.zeros(0, device=source.device)
            return self._residual_summary(torch.zeros(int(valid.sum()) * 3, device=source.device), penetration, diagnostics), diagnostics, penetration

        # The local root rotations are set exactly to the target.  The
        # residual summary therefore reports a zero pelvis error up to float
        # round-off, while contact rows remain observable for the ablation.
        post_summary, post_diagnostics, penetration = summaries(corrected_output, corrected_translation)
        pre_summary, _, _ = summaries(base_output, translation)
        rebuilt = physical.clone()
        rebuilt_body = decode_rot6d_safe(rebuilt[0, :, MOTION_LAYOUT.body_pose].reshape(frames, 21, 6))
        rebuilt_body[:] = body
        rebuilt_root = decode_rot6d_safe(rebuilt[0, :, MOTION_LAYOUT.root_rotation])
        rebuilt_root[valid] = target_root[valid]
        rebuilt[0, :, MOTION_LAYOUT.body_pose] = encode_rot6d(rebuilt_body).reshape(frames, 126)
        rebuilt[0, :, MOTION_LAYOUT.root_rotation] = encode_rot6d(rebuilt_root)
        rebuilt[0, :, MOTION_LAYOUT.root_translation] = corrected_translation
        rebuilt = authority_project(rebuilt, valid_mask=valid.unsqueeze(0), output_dtype=torch.float32).physical_motion
        projected = self._normalised(rebuilt)
        finite = bool(torch.isfinite(projected).all())
        if not finite:
            projected = clean_motion.clone()
        increment = torch.zeros((frames, VARIABLES_PER_FRAME), device=source.device, dtype=torch.float32)
        increment[:, :3] = corrected_translation - translation
        increment[:, 3:6] = so3_log(target_root @ root.transpose(-1, -2))
        max_joint = float(torch.linalg.vector_norm(increment[:, 3:].reshape(frames, len(ROTATION_NAMES), 3), dim=-1).max() * 180.0 / math.pi)
        root_step = float(torch.linalg.vector_norm(increment[:, :3], dim=-1).max())
        normalized_before = self._violation_components(pre_summary)
        normalized_after = self._violation_components(post_summary)
        active_penetration = int(penetration.numel())
        result = ProjectionResult(
            projected_clean_motion=projected[0] if unbatched else projected,
            pre_residuals=pre_summary,
            post_residuals=post_summary,
            pelvis_residual=float(post_summary["pelvis_geodesic_rms_deg"] or 0.0),
            heel_residual=float(post_summary["heel_rms_m"] or 0.0),
            toe_residual=float(post_summary["toe_rms_m"] or 0.0),
            heel_velocity_residual=float(post_summary["heel_velocity_rms_m_per_frame"] or 0.0),
            toe_velocity_residual=float(post_summary["toe_velocity_rms_m_per_frame"] or 0.0),
            penetration_residual=float(post_summary["penetration_active_rms_m"] or 0.0),
            delta_q_norm=float(torch.linalg.vector_norm(increment)),
            delta_root_translation_norm=root_step,
            num_relinearization_iters=1,
            jacobian_condition=None,
            active_penetration_constraints=active_penetration,
            converged=finite and can_lift and float(post_summary["pelvis_geodesic_rms_deg"] or 0.0) <= self.config.pelvis_tolerance_deg,
            finite=finite,
            per_iteration_root_translation=[root_step],
            cumulative_root_translation=root_step,
            records=[{
                "relinearization_iter": 1,
                "full_sequence_low_memory": True,
                "accepted": finite and can_lift,
                "max_joint_increment_deg": max_joint,
                "delta_root_translation": root_step,
                "per_iteration_root_translation": root_step,
                "cumulative_root_translation": root_step,
                "normalized_violation_before": normalized_before,
                "normalized_violation_after": normalized_after,
                "active_penetration_count": active_penetration,
                "penetration_safety_feasible": can_lift,
            }],
        )
        self.last_result = result
        return result

    def project_clean_endpoint(
        self,
        clean_motion: torch.Tensor,
        m0_motion: torch.Tensor,
        contact_data: Mapping[str, Any],
        target_dose: float,
        sampling_state: Mapping[str, Any],
    ) -> ProjectionResult:
        """Project one batch-one endpoint with iterative relinearisation."""

        if self.config.protocol == DOSE_FIRST_CONTACT_ABLATION_PROTOCOL and self.config.projection_scope == "full_sequence":
            return self._project_full_sequence_low_memory(
                clean_motion, m0_motion, contact_data, target_dose, sampling_state
            )

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
        dose_mask = projection_mask.clone()
        context_mask = torch.as_tensor(
            contact_data.get("context_mask", projection_mask),
            device=clean_motion.device,
            dtype=torch.bool,
        ) & valid_mask
        frame_indices = torch.nonzero(context_mask, as_tuple=False).flatten()
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
        target_root = m0_root.clone()
        dose_local = dose_mask[frame_indices]
        if bool(dose_local.any()):
            target_root[dose_local] = target_root_rotation(
                m0_root[dose_local], float(target_dose)
            )
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
        velocity_pairs: dict[str, torch.Tensor] | None = None
        velocity_pair_weights: dict[str, torch.Tensor] | None = None
        if self.config.contact_velocity_weight > 0.0:
            velocity_pairs = {}
            velocity_pair_weights = {}
            global_pair_count = max(int(m0_source.shape[0]) - 1, 0)
            local_global = frame_indices.detach().cpu().tolist()
            for side in ("left", "right"):
                pair_source = torch.as_tensor(
                    contact_data.get("velocity_pair_masks", {}).get(
                        side, [False] * global_pair_count
                    ),
                    device=clean_motion.device,
                    dtype=torch.bool,
                )
                pair_weight_source = torch.as_tensor(
                    contact_data.get("velocity_pair_weights", {}).get(
                        side, [self.config.transition_pair_weight] * global_pair_count
                    ),
                    device=clean_motion.device,
                    dtype=torch.float32,
                )
                if pair_source.shape != (global_pair_count,):
                    raise ValueError("frozen velocity pair mask must have length T-1")
                if pair_weight_source.shape != (global_pair_count,):
                    raise ValueError("frozen velocity pair weights must have length T-1")
                local_pair = torch.zeros(
                    max(len(local_global) - 1, 0), device=clean_motion.device, dtype=torch.bool
                )
                local_weights = torch.zeros(
                    max(len(local_global) - 1, 0), device=clean_motion.device, dtype=torch.float32
                )
                for local_index, (left_global, right_global) in enumerate(
                    zip(local_global[:-1], local_global[1:])
                ):
                    if right_global == left_global + 1:
                        local_pair[local_index] = pair_source[left_global]
                        local_weights[local_index] = pair_weight_source[left_global]
                velocity_pairs[side] = local_pair
                velocity_pair_weights[side] = local_weights
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
        per_iteration_root_translation: list[float] = []
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
                    velocity_pairs=velocity_pairs,
                    velocity_pair_weights=velocity_pair_weights,
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
                    velocity_pairs=velocity_pairs,
                    velocity_pair_weights=velocity_pair_weights,
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
            # Boundary halo frames provide fixed context for velocity
            # residuals but are not optimisation variables.  Mask their
            # Jacobian columns and make their metric block an identity.
            variable_columns = dose_local.repeat_interleave(VARIABLES_PER_FRAME)
            if not bool(variable_columns.all()):
                metric = metric.clone()
                metric[~variable_columns, :] = 0.0
                metric[:, ~variable_columns] = 0.0
                metric[~variable_columns, ~variable_columns] = 1.0
                jacobian = jacobian.clone()
                jacobian[:, ~variable_columns] = 0.0
            flat_step, solve_diagnostics = solve_local_projection(
                metric,
                pelvis_jacobian,
                -pelvis0.detach(),
                contact_jacobian,
                -contact0.detach(),
                penetration_jacobian,
                -penetration0.detach(),
                contact_weight=self.config.contact_weight,
                contact_row_weights=diagnostics0["contact_row_weights"],
                contact_row_scales=self._contact_row_scales(diagnostics0),
            )
            condition = solve_diagnostics["jacobian_condition"]
            proposed = project_increment_norms(
                flat_step.reshape_as(zero),
                max_joint_increment_deg=self.config.max_joint_increment_deg,
                max_root_translation_m=self.config.max_root_translation_m,
            )
            violation_before = self._violation(summary_before)
            merit_before = self._violation_components(summary_before)
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
                        velocity_pairs=velocity_pairs,
                        velocity_pair_weights=velocity_pair_weights,
                    )
                    summary_after = self._residual_summary(p1, g1, d1)
                    merit_after = self._violation_components(summary_after)
                    accepted = (
                        self._accept_dose_first(merit_before, merit_after)
                        if self.config.acceptance_mode == "dose_first"
                        else self._accept_merit(merit_before, merit_after)
                    )
                    if accepted:
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
                        "normalized_violation_before": merit_before,
                        "normalized_violation_after": merit_before,
                    }
                )
                break
            current_body, current_root, current_translation = accepted_state
            total_increment += proposed * float(accepted_alpha)
            iteration_root_translation = float(
                torch.linalg.vector_norm(
                    (proposed * float(accepted_alpha))[:, :3], dim=-1
                ).max().cpu()
            )
            per_iteration_root_translation.append(iteration_root_translation)
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
                "heel_velocity_before_m_per_frame": summary_before[
                    "heel_velocity_rms_m_per_frame"
                ],
                "heel_velocity_after_m_per_frame": accepted_summary[
                    "heel_velocity_rms_m_per_frame"
                ],
                "toe_velocity_before_m_per_frame": summary_before[
                    "toe_velocity_rms_m_per_frame"
                ],
                "toe_velocity_after_m_per_frame": accepted_summary[
                    "toe_velocity_rms_m_per_frame"
                ],
                "penetration_before_m": summary_before["penetration_active_rms_m"],
                "penetration_after_m": accepted_summary["penetration_active_rms_m"],
                "delta_q_norm": float(torch.linalg.vector_norm(proposed).cpu()),
                "delta_root_translation": float(
                    torch.linalg.vector_norm(proposed[:, :3], dim=-1).max().cpu()
                ),
                "per_iteration_root_translation": iteration_root_translation,
                "cumulative_root_translation": float(
                    torch.linalg.vector_norm(total_increment[:, :3], dim=-1).max().cpu()
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
                "normalized_violation_before": merit_before,
                "normalized_violation_after": self._violation_components(
                    accepted_summary
                ),
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
                and float(
                    accepted_summary["heel_velocity_rms_m_per_frame"] or 0.0
                )
                <= self.config.contact_velocity_tolerance_m_per_frame
                and float(
                    accepted_summary["toe_velocity_rms_m_per_frame"] or 0.0
                )
                <= self.config.contact_velocity_tolerance_m_per_frame
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
            heel_velocity_residual=float(
                final_summary["heel_velocity_rms_m_per_frame"] or 0.0
            ),
            toe_velocity_residual=float(
                final_summary["toe_velocity_rms_m_per_frame"] or 0.0
            ),
            penetration_residual=float(
                final_summary["penetration_active_rms_m"] or 0.0
            ),
            delta_q_norm=float(torch.linalg.vector_norm(total_increment).cpu()),
            delta_root_translation_norm=float(
                torch.linalg.vector_norm(total_increment[:, :3], dim=-1).max().cpu()
            ),
            per_iteration_root_translation=per_iteration_root_translation,
            cumulative_root_translation=float(
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
            "protocol": self.config.protocol,
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
        last_sampling = self.last_result.projected_clean_motion if self.last_result is not None else None
        terminal_error: str | None = None
        try:
            terminal = self.project_clean_endpoint(
                official.float(),
                self.baseline_motion.to(device=official.device),
                self.contact_data,
                self.target_dose,
                {"valid_mask": valid_mask},
            )
            projected_norm = terminal.projected_clean_motion
        except Exception as exc:
            if self.config.protocol != DOSE_FIRST_CONTACT_ABLATION_PROTOCOL:
                raise
            terminal = None
            terminal_error = repr(exc)
            projected_norm = official.float().clone()
        projected_physical = self._physical(projected_norm)
        rebuilt = authority_project(
            projected_physical,
            valid_mask=valid_mask.to(device=physical.device, dtype=torch.bool),
            output_dtype=torch.float32,
        ).physical_motion
        g0 = self._normalised(rebuilt)
        terminal_record: dict[str, Any] = {
            "attempted": True,
            "success": terminal is not None and bool(terminal.finite) and bool(terminal.converged),
            "error": terminal_error,
            "last_sampling_projection_available": last_sampling is not None,
        }
        if terminal is not None:
            terminal_record.update(
                {
                    "finite": bool(terminal.finite),
                    "converged": bool(terminal.converged),
                    "pre_residuals": terminal.pre_residuals,
                    "post_residuals": terminal.post_residuals,
                }
            )
        if last_sampling is not None:
            previous = last_sampling[0] if last_sampling.ndim == 3 else last_sampling
            final_value = projected_norm[0] if projected_norm.ndim == 3 else projected_norm
            difference = (final_value.float() - previous.float()).abs()
            terminal_record.update(
                {
                    "last_sampling_to_terminal_max_abs": float(difference.max().cpu()),
                    "last_sampling_to_terminal_rms": float(torch.sqrt(difference.square().mean()).cpu()),
                    "last_sampling_to_terminal_per_frame_rms": torch.sqrt(
                        difference.square().mean(dim=-1)
                    ).detach().cpu().tolist(),
                }
            )
        self.last_terminal_record = terminal_record
        if self.config.artifact_dir:
            artifact_dir = Path(self.config.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "official_pre_cast_norm": official.detach().cpu(),
                    "last_sampling_projection_norm": None if last_sampling is None else last_sampling.detach().cpu(),
                    "terminal_projection_norm": projected_norm.detach().cpu(),
                    "terminal_difference_norm": (projected_norm.float() - official.float()).detach().cpu(),
                },
                artifact_dir / "terminal_projection_endpoints.pt",
            )
        return ProjectionFinalOutputs(
            g0=g0,
            protocol=self.config.protocol,
            summary=self.protocol_record(),
        )

    def protocol_record(self) -> dict[str, Any]:
        return {
            "protocol": self.config.protocol,
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
                    "m0_match_status",
                    "allow_m0_mismatch",
                    "projection_baseline",
                    "m0_reference_protocol",
                    "baseline_origin",
                    "legacy_v3_relation",
                    "context_mask",
                    "velocity_pair_masks",
                    "velocity_pair_weights",
                )
                if key in self.contact_data
            },
            "step_records": _strict_value(self.step_records),
            "terminal_projection": _strict_value(self.last_terminal_record),
        }


__all__ = [
    "ACTIVE_BODY_INDICES",
    "ACTIVE_JOINT_NAMES",
    "EUCLIDEAN_METRIC",
    "KINEMATIC_TEMPORAL_METRIC",
    "METHOD_NAME",
    "PENETRATION_METHOD",
    "PROTOCOL_NAME",
    "TEMPORAL_CONTACT_PROTOCOL",
    "CURRENT_ENV_PAIRED_PROTOCOL",
    "DOSE_FIRST_CONTACT_ABLATION_PROTOCOL",
    "DOSE_FIRST_CONTACT_MODES",
    "FULL_SEQUENCE_PROTOCOLS",
    "TEMPORAL_CONTACT_PROTOCOLS",
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
    "temporal_contact_residual",
    "write_strict_json",
]
