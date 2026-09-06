"""Small lower-body terminal compensation prototype for v0.4.2.

The solver is deliberately terminal-only.  It keeps the pre-cast root
translation and foot-marker positions as the reference, changes only the
anatomical lower-body tangent coordinates, and reports rank/trust diagnostics.
It is not used by the v0.1-v0.4 samplers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from evaluation.pelvis_contact_compensation_v3 import patch_centres
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe, encode_rot6d
from motion_rep.pose_authority import authority_project
from sampling.pelvis_contact_flow_projection_v0_1 import PelvisContactFlowProjector, so3_exp
from sampling.terminal_projection_policy import TerminalProjectionPolicy, build_terminal_root_target


LOWER_BODY_DOF_PROTOCOL = "vimogen_pelvis_guided_walk_v0_4_2_lower_body_terminal_v1"
LOWER_BODY_JOINTS = (
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
)
MM = 1000.0


@dataclass(frozen=True)
class LowerBodyDof:
    joint: str
    axis: tuple[float, float, float]
    anatomical_role: str


@dataclass(frozen=True)
class LowerBodyDofMap:
    """Frozen anatomical subspace; no joint is implicitly 3-DOF."""

    dofs: tuple[LowerBodyDof, ...]
    source: str = "smplx_local_axes_v0_4_2"

    @classmethod
    def default(cls) -> "LowerBodyDofMap":
        x, y, z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        dofs: list[LowerBodyDof] = []
        for side in ("left", "right"):
            dofs.append(LowerBodyDof(f"{side}_hip", x, "hip_flexion"))
            dofs.append(LowerBodyDof(f"{side}_hip", y, "hip_abduction"))
            dofs.append(LowerBodyDof(f"{side}_hip", z, "hip_rotation"))
            dofs.append(LowerBodyDof(f"{side}_knee", x, "knee_flexion"))
            dofs.append(LowerBodyDof(f"{side}_ankle", x, "ankle_flexion"))
            dofs.append(LowerBodyDof(f"{side}_ankle", y, "ankle_inversion"))
            dofs.append(LowerBodyDof(f"{side}_foot", x, "foot_toe_flexion"))
        return cls(tuple(dofs))

    @classmethod
    def full_so3_diagnostic(cls) -> "LowerBodyDofMap":
        """Return a labelled 24-coordinate diagnostic, never the formal map."""

        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        dofs = tuple(
            LowerBodyDof(joint, axis, "full_so3_diagnostic")
            for joint in LOWER_BODY_JOINTS
            for axis in axes
        )
        return cls(dofs, source="smplx_native_so3_diagnostic_only")

    def validate(self) -> None:
        if not self.dofs:
            raise ValueError("lower-body DOF map cannot be empty")
        names = set(LOWER_BODY_JOINTS)
        for item in self.dofs:
            if item.joint not in names:
                raise ValueError(f"unknown lower-body joint: {item.joint}")
            axis = torch.tensor(item.axis, dtype=torch.float64)
            if not torch.isfinite(axis).all() or float(torch.linalg.vector_norm(axis)) <= 0.0:
                raise ValueError(f"invalid axis for {item.joint}")

    def jsonable(self) -> dict[str, Any]:
        self.validate()
        return {"source": self.source, "dof_count": len(self.dofs), "dofs": [asdict(item) for item in self.dofs]}


@dataclass(frozen=True)
class LowerBodySolverConfig:
    max_iterations: int = 8
    damping: float = 1.0e-4
    finite_difference_step: float = 1.0e-4
    max_joint_increment_deg: float = 5.0
    penetration_epsilon_m: float = 5.0e-4
    smoothness_weight: float = 0.0

    def validate(self) -> None:
        if self.max_iterations < 1 or self.damping <= 0.0 or self.finite_difference_step <= 0.0:
            raise ValueError("invalid lower-body solver bounds")
        if self.max_joint_increment_deg <= 0.0 or self.penetration_epsilon_m < 0.0:
            raise ValueError("invalid lower-body trust/ground bounds")
        if self.smoothness_weight < 0.0:
            raise ValueError("smoothness weight cannot be negative")


@dataclass
class LowerBodySolveResult:
    projected_physical: torch.Tensor
    target_root: torch.Tensor
    pelvis_active: torch.Tensor
    records: list[dict[str, Any]]
    finite: bool
    root_translation_locked: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "protocol": LOWER_BODY_DOF_PROTOCOL,
            "finite": self.finite,
            "root_translation_locked": self.root_translation_locked,
            "pelvis_active_frame_count": int(self.pelvis_active.sum().item()),
            "records": self.records,
        }


def _model_markers(model: Any, body: torch.Tensor, root: torch.Tensor, translation: torch.Tensor, patches: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    output = PelvisContactFlowProjector._model_output(model, body, root, translation)
    left_heel, left_toe = patch_centres(output.vertices, patches["left"])
    right_heel, right_toe = patch_centres(output.vertices, patches["right"])
    return torch.stack((left_heel, left_toe, right_heel, right_toe), dim=1), output.vertices


def _apply_dof_delta(body: torch.Tensor, dof_map: LowerBodyDofMap, delta: torch.Tensor) -> torch.Tensor:
    result = body.clone()
    for index, item in enumerate(dof_map.dofs):
        body_index = SMPLX_22_JOINT_INDEX[item.joint] - 1
        axis = torch.as_tensor(item.axis, device=body.device, dtype=body.dtype)
        result[body_index] = so3_exp(axis * delta[index]) @ result[body_index]
    return result


def _finite_difference_jacobian(function: Any, point: torch.Tensor, step: float) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    for index in range(point.numel()):
        offset = torch.zeros_like(point)
        offset[index] = step
        columns.append((function(point + offset) - function(point - offset)) / (2.0 * step))
    return torch.stack(columns, dim=-1)


def solve_lower_body_position(
    pre_cast_physical: torch.Tensor,
    m0_physical: torch.Tensor,
    valid_mask: torch.Tensor,
    target_dose_deg: float,
    *,
    model: Any,
    patches: Mapping[str, Any],
    dead_zone_deg: float,
    dof_map: LowerBodyDofMap | None = None,
    config: LowerBodySolverConfig | None = None,
    floor_heights: Mapping[str, float] | None = None,
) -> LowerBodySolveResult:
    """Solve lower-body foot preservation with the root translation locked."""

    dof_map = dof_map or LowerBodyDofMap.default()
    dof_map.validate()
    config = config or LowerBodySolverConfig()
    config.validate()
    pre = pre_cast_physical.float()
    m0 = m0_physical.float()
    if pre.ndim != 2 or m0.ndim != 2 or pre.shape != m0.shape or pre.shape[-1] != 276:
        raise ValueError("pre-cast and M0 must both have shape [T,276]")
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=pre.device).reshape(-1)
    if valid.shape[0] != pre.shape[0] or not bool(valid.all()):
        raise ValueError("lower-body prototype requires a contiguous fully-valid sequence")
    body = decode_rot6d_safe(pre[:, MOTION_LAYOUT.body_pose].reshape(pre.shape[0], 21, 6)).float()
    source_root = decode_rot6d_safe(pre[:, MOTION_LAYOUT.root_rotation]).float()
    m0_root = decode_rot6d_safe(m0[:, MOTION_LAYOUT.root_rotation]).float()
    translation = pre[:, MOTION_LAYOUT.root_translation].clone()
    target_root, signed_error, pelvis_active = build_terminal_root_target(
        m0_root, source_root, target_dose_deg, valid_mask=valid,
        policy=TerminalProjectionPolicy(dead_zone_deg=dead_zone_deg),
    )
    with torch.no_grad():
        targets, _ = _model_markers(model, body, source_root, translation, patches)
    solved_body = body.clone()
    records: list[dict[str, Any]] = []
    floor_heights = dict(floor_heights or {})
    max_increment = math.radians(config.max_joint_increment_deg)
    for frame in torch.nonzero(pelvis_active, as_tuple=False).flatten().tolist():
        base_body = solved_body[frame].clone()
        base_root = target_root[frame]
        base_translation = translation[frame]
        target_markers = targets[frame].detach()
        zero = torch.zeros(len(dof_map.dofs), device=pre.device, dtype=torch.float32)

        def markers(delta: torch.Tensor) -> torch.Tensor:
            current_body = _apply_dof_delta(base_body, dof_map, delta)
            value, _ = _model_markers(model, current_body.unsqueeze(0), base_root.unsqueeze(0), base_translation.unsqueeze(0), patches)
            return value[0]

        initial = markers(zero)
        initial_error = (initial - target_markers).reshape(-1)
        current = zero.clone()
        rejected = 0
        rank = 0
        singular_values: list[float] = []
        for iteration in range(1, config.max_iterations + 1):
            residual = (markers(current) - target_markers).reshape(-1)
            jacobian = _finite_difference_jacobian(lambda point: (markers(point) - target_markers).reshape(-1), current, config.finite_difference_step)
            try:
                singular = torch.linalg.svdvals(jacobian)
                singular_values = [float(item) for item in singular.detach().cpu()]
                rank = int(torch.linalg.matrix_rank(jacobian, tol=1.0e-6).item())
                normal = jacobian.T @ jacobian + config.damping * torch.eye(jacobian.shape[1], device=jacobian.device)
                step = torch.linalg.solve(normal, -(jacobian.T @ residual))
            except RuntimeError:
                rejected += 1
                break
            step = step.clamp(-max_increment, max_increment)
            before_norm = float(torch.linalg.vector_norm(residual))
            accepted = False
            chosen = current
            after_norm = before_norm
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = current + alpha * step
                candidate_markers = markers(candidate)
                candidate_residual = candidate_markers - target_markers
                if floor_heights:
                    unsafe = False
                    for side, marker_indices in (("left", (0, 1)), ("right", (2, 3))):
                        if side not in floor_heights:
                            continue
                        candidate_min = candidate_markers[list(marker_indices), 2].min()
                        base_min = initial[list(marker_indices), 2].min()
                        if candidate_min < base_min - config.penetration_epsilon_m:
                            unsafe = True
                    if unsafe:
                        continue
                norm = float(torch.linalg.vector_norm(candidate_residual))
                if math.isfinite(norm) and norm < before_norm - 1.0e-9:
                    chosen, after_norm, accepted = candidate, norm, True
                    break
            if not accepted:
                rejected += 1
                break
            current = chosen
            if after_norm <= 0.001 * math.sqrt(12.0):
                break
        solved_body[frame] = _apply_dof_delta(base_body, dof_map, current)
        final_markers, _ = _model_markers(model, solved_body[frame].unsqueeze(0), base_root.unsqueeze(0), base_translation.unsqueeze(0), patches)
        final_error = (final_markers[0] - target_markers).reshape(-1)
        records.append({
            "frame": int(frame),
            "pelvis_error_deg": float(signed_error[frame]),
            "initial_foot_residual_mm": float(torch.linalg.vector_norm(initial_error).item() * MM),
            "final_foot_residual_mm": float(torch.linalg.vector_norm(final_error).item() * MM),
            "max_dof_increment_deg": float(current.abs().max().item() * 180.0 / math.pi),
            "jacobian_rank": rank,
            "jacobian_singular_values": singular_values,
            "rejected_steps": rejected,
            "root_translation_increment_mm": 0.0,
            "root_translation_locked": True,
        })
    rebuilt = pre.clone()
    rebuilt[:, MOTION_LAYOUT.body_pose] = encode_rot6d(solved_body).reshape(pre.shape[0], 126)
    rebuilt[:, MOTION_LAYOUT.root_rotation] = encode_rot6d(target_root)
    rebuilt[:, MOTION_LAYOUT.root_translation] = translation
    rebuilt = authority_project(rebuilt.unsqueeze(0), valid_mask=valid.unsqueeze(0), output_dtype=torch.float32).physical_motion[0]
    _finite = bool(torch.isfinite(rebuilt).all())
    root_locked = bool(torch.equal(rebuilt[:, MOTION_LAYOUT.root_translation], pre[:, MOTION_LAYOUT.root_translation]))
    return LowerBodySolveResult(rebuilt, target_root, pelvis_active, records, _finite, root_locked)


__all__ = [
    "LOWER_BODY_DOF_PROTOCOL", "LowerBodyDof", "LowerBodyDofMap",
    "LowerBodySolverConfig", "LowerBodySolveResult", "solve_lower_body_position",
]
