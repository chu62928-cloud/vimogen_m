"""Deterministic contact-aware whole-body compensation for pelvis v3.

The solver edits only direct pose streams.  The root rotation is constructed
analytically from the frozen M0 sagittal frame; the trainable variables are
root translation and a small set of lower-body/spine local rotations.  The
three optimisation stages are lexicographic in practice: contact is restored
first, trunk orientation is added only after contact, and posture/time
regularisation is added last.  Every returned motion is authority-projected
before it leaves this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from evaluation.pelvis_contact_compensation_v3 import (
    CONTACT_HEIGHT_M,
    CONTACT_SPEED_M_PER_FRAME,
    FLAT_GAP_M,
    STABLE_CONFIDENCE,
    contact_evidence,
    patch_centres,
    target_root_rotation,
)
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe, encode_rot6d
from motion_rep.pose_authority import authority_project
from motion_rep.rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle


ACTIVE_JOINT_NAMES = (
    "spine1", "spine2", "spine3",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
)
ACTIVE_BODY_INDICES = tuple(SMPLX_22_JOINT_INDEX[name] - 1 for name in ACTIVE_JOINT_NAMES)


def project_trust_region(
    body_delta: torch.Tensor,
    translation_delta: torch.Tensor,
    *,
    max_rotation_deg: float = 30.0,
    max_translation_m: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Lie-algebra rotations and translations onto norm balls."""

    if body_delta.ndim < 1 or body_delta.shape[-1] != 3:
        raise ValueError("body_delta must end in a 3-vector")
    if translation_delta.ndim < 1 or translation_delta.shape[-1] != 3:
        raise ValueError("translation_delta must end in a 3-vector")
    if max_rotation_deg <= 0.0 or max_translation_m <= 0.0:
        raise ValueError("trust-region bounds must be positive")
    max_rotation = math.radians(max_rotation_deg)
    rotation_norm = torch.linalg.vector_norm(body_delta, dim=-1, keepdim=True).clamp_min(1.0e-12)
    body_projected = body_delta * (max_rotation / rotation_norm).clamp(max=1.0)
    translation_norm = torch.linalg.vector_norm(translation_delta, dim=-1, keepdim=True).clamp_min(1.0e-12)
    translation_projected = translation_delta * (max_translation_m / translation_norm).clamp(max=1.0)
    return body_projected, translation_projected


@dataclass(frozen=True)
class PelvisCompensationConfig:
    protocol: str = "vimogen_pelvis_contact_compensation_v3_0_1"
    max_rotation_deg: float = 30.0
    max_translation_m: float = 0.05
    stable_confidence: float = STABLE_CONFIDENCE
    contact_height_m: float = CONTACT_HEIGHT_M
    contact_speed_m_per_frame: float = CONTACT_SPEED_M_PER_FRAME
    flat_gap_m: float = FLAT_GAP_M
    contact_outer_iterations: int = 6
    trunk_outer_iterations: int = 4
    posture_outer_iterations: int = 3
    inner_iterations: int = 80
    learning_rate: float = 0.01
    penalty_start: float = 10.0
    penalty_multiplier: float = 5.0
    trunk_tolerance_deg: float = 2.0
    pelvis_neck_tolerance_deg: float = 2.0
    pelvis_head_tolerance_deg: float = 3.0
    pelvis_support_drift_m: float = 0.020
    contact_tolerance_m: float = 0.001
    smooth_velocity_weight: float = 0.10
    smooth_acceleration_weight: float = 0.05
    posture_weight: float = 0.01

    def validate(self) -> None:
        if self.max_rotation_deg <= 0 or self.max_translation_m <= 0:
            raise ValueError("trust-region bounds must be positive")
        if min(self.contact_outer_iterations, self.trunk_outer_iterations, self.posture_outer_iterations, self.inner_iterations) < 1:
            raise ValueError("solver iteration counts must be positive")
        if self.learning_rate <= 0 or self.penalty_start <= 0 or self.penalty_multiplier <= 1:
            raise ValueError("invalid optimiser configuration")
        if not 0 < self.stable_confidence <= 1:
            raise ValueError("stable_confidence must lie in (0,1]")


class PelvisContactCompensationSolver:
    """Solve one M0 sequence or contact window on a frozen SMPL-X model."""

    def __init__(
        self,
        m0_motion: torch.Tensor,
        model: Any,
        patches: Mapping[str, Mapping[str, list[int]]],
        *,
        valid_mask: torch.Tensor | None = None,
        stable_masks: Mapping[str, torch.Tensor] | None = None,
        config: PelvisCompensationConfig | None = None,
        device: str | torch.device = "cuda:0",
    ) -> None:
        if m0_motion.ndim == 3 and m0_motion.shape[0] == 1:
            m0_motion = m0_motion[0]
        if m0_motion.ndim != 2 or m0_motion.shape[-1] != 276:
            raise ValueError("m0_motion must be [T,276]")
        self.config = config or PelvisCompensationConfig()
        self.config.validate()
        self.device = torch.device(device)
        self.model = model
        self.patches = patches
        self.m0 = m0_motion.float().to(self.device)
        length = self.m0.shape[0] if valid_mask is None else int(valid_mask.sum().item())
        if length < 1 or length > self.m0.shape[0]:
            raise ValueError("valid_mask has invalid length")
        self.m0 = self.m0[:length]
        self.frames = length
        # ``SMPLX.forward`` repeats landmark barycentric coordinates using the
        # module's constructor batch size rather than the input batch.  A
        # window solve has a shorter batch, so keep that internal value in
        # lockstep with the current sequence.
        if hasattr(self.model, "batch_size"):
            self.model.batch_size = self.frames
        self.valid_mask = torch.ones(length, dtype=torch.bool, device=self.device)
        self.base_body = decode_rot6d_safe(self.m0[..., MOTION_LAYOUT.body_pose].reshape(length, 21, 6))
        self.base_root = decode_rot6d_safe(self.m0[..., MOTION_LAYOUT.root_rotation])
        self.base_translation = self.m0[..., MOTION_LAYOUT.root_translation]
        self.body_delta = torch.zeros((length, len(ACTIVE_BODY_INDICES), 3), dtype=torch.float32, device=self.device, requires_grad=True)
        self.translation_delta = torch.zeros((length, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        with torch.no_grad():
            base_output = self._model(self.base_body, self.base_root, self.base_translation)
            self.base_vertices = base_output.vertices.detach()
            self.base_joints = base_output.joints[..., :22, :].detach()
        self.contact: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            heel, toe = patch_centres(self.base_vertices, patches[side])
            self.contact[side] = {
                "heel": heel.detach(), "toe": toe.detach(),
                "stable_mask": (
                    torch.as_tensor(stable_masks[side], device=self.device, dtype=torch.bool)[:length]
                    if stable_masks is not None and side in stable_masks
                    else None
                ),
                "evidence": contact_evidence(
                    heel.detach(), toe.detach(),
                    contact_height_m=self.config.contact_height_m,
                    contact_speed_m_per_frame=self.config.contact_speed_m_per_frame,
                    flat_gap_m=self.config.flat_gap_m,
                ),
            }
        trunk = self.base_joints[..., SMPLX_22_JOINT_INDEX["neck"], :] - self.base_joints[..., SMPLX_22_JOINT_INDEX["spine1"], :]
        self.base_trunk = F.normalize(trunk, dim=-1)
        pelvis_index = SMPLX_22_JOINT_INDEX["pelvis"]
        neck_index = SMPLX_22_JOINT_INDEX["neck"]
        head_index = SMPLX_22_JOINT_INDEX["head"]
        self.base_pelvis_neck = F.normalize(
            self.base_joints[..., neck_index, :] - self.base_joints[..., pelvis_index, :], dim=-1
        )
        self.base_pelvis_head = F.normalize(
            self.base_joints[..., head_index, :] - self.base_joints[..., pelvis_index, :], dim=-1
        )
        self.floor = {side: torch.as_tensor(self.contact[side]["evidence"]["floor_height_m"], device=self.device, dtype=torch.float32) for side in ("left", "right")}
        self._target_root: torch.Tensor | None = None

    def _model(self, body: torch.Tensor, root: torch.Tensor, translation: torch.Tensor) -> Any:
        body_aa = mat3x3_to_axis_angle(body).reshape(self.frames, 63)
        root_aa = mat3x3_to_axis_angle(root).reshape(self.frames, 3)
        # SMPL-X stores default hands/face/shape at the constructor batch
        # size.  Explicitly broadcast the frozen streams to this sequence's
        # length so window solves (which are shorter than the full batch) do
        # not mix a 13-frame pose with a 100-frame default expression.
        def zeros_like_parameter(name: str) -> torch.Tensor:
            value = getattr(self.model, name)
            return torch.zeros((self.frames, value.shape[-1]), device=self.device, dtype=body_aa.dtype)

        return self.model(
            body_pose=body_aa,
            global_orient=root_aa,
            transl=translation,
            betas=zeros_like_parameter("betas"),
            expression=zeros_like_parameter("expression"),
            left_hand_pose=zeros_like_parameter("left_hand_pose"),
            right_hand_pose=zeros_like_parameter("right_hand_pose"),
            jaw_pose=zeros_like_parameter("jaw_pose"),
            leye_pose=zeros_like_parameter("leye_pose"),
            reye_pose=zeros_like_parameter("reye_pose"),
            return_verts=True,
        )

    def _state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body = self.base_body.clone()
        deltas = axis_angle_to_mat3x3(self.body_delta)
        for index, body_index in enumerate(ACTIVE_BODY_INDICES):
            body[:, body_index] = deltas[:, index] @ self.base_body[:, body_index]
        if self._target_root is None:
            raise RuntimeError("target root has not been set")
        translation = self.base_translation + self.translation_delta
        return body, self._target_root, translation

    def _forward(self) -> Any:
        body, root, translation = self._state()
        return self._model(body, root, translation)

    def _residuals(self, output: Any) -> dict[str, torch.Tensor]:
        residuals: dict[str, list[torch.Tensor]] = {
            "contact": [],
            "orientation": [],
            "penetration": [],
            "trunk": [],
            "pelvis_neck": [],
            "pelvis_head": [],
            "support_drift": [],
        }
        for side in ("left", "right"):
            heel, toe = patch_centres(output.vertices, self.patches[side])
            evidence = self.contact[side]["evidence"]
            confidence = torch.as_tensor(evidence["confidence"], device=self.device, dtype=torch.float32)
            frozen_stable = self.contact[side]["stable_mask"]
            stable = (confidence >= self.config.stable_confidence) if frozen_stable is None else frozen_stable
            stable = stable & self.valid_mask
            if bool(stable.any()):
                target_heel = self.contact[side]["heel"]
                target_toe = self.contact[side]["toe"]
                weight = confidence[stable].sqrt().unsqueeze(-1)
                residuals["contact"].extend(((heel[stable] - target_heel[stable]) * weight).unbind(0))
                residuals["contact"].extend(((toe[stable] - target_toe[stable]) * weight).unbind(0))
                base_dir = F.normalize(target_toe[stable] - target_heel[stable], dim=-1)
                candidate_dir = F.normalize(toe[stable] - heel[stable], dim=-1)
                residuals["orientation"].extend(torch.cross(candidate_dir, base_dir, dim=-1).unbind(0))
            sole = output.vertices[:, torch.as_tensor(self.patches[side]["sole"], device=self.device)].amin(dim=1)[:, 2]
            residuals["penetration"].append(F.relu(self.floor[side] - sole))
        trunk = output.joints[..., SMPLX_22_JOINT_INDEX["neck"], :] - output.joints[..., SMPLX_22_JOINT_INDEX["spine1"], :]
        residuals["trunk"].append(torch.cross(F.normalize(trunk, dim=-1), self.base_trunk, dim=-1))
        pelvis_index = SMPLX_22_JOINT_INDEX["pelvis"]
        neck = output.joints[..., SMPLX_22_JOINT_INDEX["neck"], :] - output.joints[..., pelvis_index, :]
        head = output.joints[..., SMPLX_22_JOINT_INDEX["head"], :] - output.joints[..., pelvis_index, :]
        residuals["pelvis_neck"].append(torch.cross(F.normalize(neck, dim=-1), self.base_pelvis_neck, dim=-1))
        residuals["pelvis_head"].append(torch.cross(F.normalize(head, dim=-1), self.base_pelvis_head, dim=-1))
        pelvis = output.joints[..., pelvis_index, :]
        for side in ("left", "right"):
            heel, toe = patch_centres(output.vertices, self.patches[side])
            base_heel, base_toe = self.contact[side]["heel"], self.contact[side]["toe"]
            confidence = torch.as_tensor(self.contact[side]["evidence"]["confidence"], device=self.device, dtype=torch.float32)
            frozen_stable = self.contact[side]["stable_mask"]
            stable = (confidence >= self.config.stable_confidence) if frozen_stable is None else frozen_stable
            stable = stable & self.valid_mask
            if bool(stable.any()):
                base_anchor = 0.5 * (base_heel + base_toe)
                candidate_anchor = 0.5 * (heel + toe)
                drift = ((pelvis - candidate_anchor) - (self.base_joints[..., pelvis_index, :] - base_anchor))[stable, :2]
                residuals["support_drift"].extend(drift.unbind(0))

        dimensions = {"contact": 3, "orientation": 3, "penetration": 1, "trunk": 3, "pelvis_neck": 3, "pelvis_head": 3, "support_drift": 2}
        return {
            key: torch.cat([value.reshape(-1, dimensions[key]) for value in values], dim=0) if values else torch.zeros((0, dimensions[key]), device=self.device)
            for key, values in residuals.items()
        }

    def _objective(self, stage: int, penalty: float, multipliers: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self._forward()
        residuals = self._residuals(output)
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for key in ("contact", "orientation", "penetration"):
            residual = residuals[key]
            if residual.numel():
                lam = multipliers.get(key)
                if lam is None or lam.shape != residual.shape:
                    lam = torch.zeros_like(residual)
                    multipliers[key] = lam
                loss = loss + (lam * residual).mean() + 0.5 * penalty * residual.square().mean()
        if stage >= 2:
            for key in ("trunk", "pelvis_neck", "pelvis_head", "support_drift"):
                residual = residuals[key]
                if not residual.numel():
                    continue
                lam = multipliers.get(key)
                if lam is None or lam.shape != residual.shape:
                    lam = torch.zeros_like(residual)
                    multipliers[key] = lam
                loss = loss + (lam * residual).mean() + 0.5 * penalty * residual.square().mean()
        if stage >= 3:
            loss = loss + self.config.posture_weight * (
                (self.body_delta / math.radians(self.config.max_rotation_deg)).square().mean()
                + (self.translation_delta / self.config.max_translation_m).square().mean()
            )
            if self.frames > 1:
                loss = loss + self.config.smooth_velocity_weight * self.body_delta[1:].sub(self.body_delta[:-1]).square().mean()
                loss = loss + self.config.smooth_velocity_weight * self.translation_delta[1:].sub(self.translation_delta[:-1]).square().mean()
            if self.frames > 2:
                loss = loss + self.config.smooth_acceleration_weight * self.body_delta[2:].sub(2 * self.body_delta[1:-1]).add(self.body_delta[:-2]).square().mean()
                loss = loss + self.config.smooth_acceleration_weight * self.translation_delta[2:].sub(2 * self.translation_delta[1:-1]).add(self.translation_delta[:-2]).square().mean()
        return loss, residuals

    def _clip_bounds(self) -> None:
        with torch.no_grad():
            body, translation = project_trust_region(
                self.body_delta,
                self.translation_delta,
                max_rotation_deg=self.config.max_rotation_deg,
                max_translation_m=self.config.max_translation_m,
            )
            self.body_delta.copy_(body)
            self.translation_delta.copy_(translation)

    def _stage_keys(self, stage: int) -> tuple[str, ...]:
        base = ("contact", "orientation", "penetration")
        if stage == 1:
            return base
        return base + ("trunk", "pelvis_neck", "pelvis_head", "support_drift")

    def _residual_summary(self, residuals: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            key: float(torch.sqrt(value.square().mean()).item()) if value.numel() else 0.0
            for key, value in residuals.items()
        }

    def _stage_thresholds(self, stage: int) -> dict[str, float]:
        thresholds = {
            "contact": self.config.contact_tolerance_m,
            "penetration": self.config.contact_tolerance_m,
            "orientation": math.sin(math.radians(2.0)),
        }
        if stage >= 2:
            thresholds.update(
                {
                    "trunk": math.sin(math.radians(self.config.trunk_tolerance_deg)),
                    "pelvis_neck": math.sin(math.radians(self.config.pelvis_neck_tolerance_deg)),
                    "pelvis_head": math.sin(math.radians(self.config.pelvis_head_tolerance_deg)),
                    "support_drift": self.config.pelvis_support_drift_m,
                }
            )
        return thresholds

    def _stage_score(self, stage: int, summary: dict[str, float]) -> tuple[int, float, float]:
        thresholds = self._stage_thresholds(stage)
        violations = [max(summary.get(key, 0.0) / limit - 1.0, 0.0) for key, limit in thresholds.items()]
        return (sum(value > 0.0 for value in violations), max(violations, default=0.0), sum(violations))

    def _stage_feasible(self, stage: int, summary: dict[str, float]) -> bool:
        thresholds = self._stage_thresholds(stage)
        return all(summary.get(key, 0.0) <= limit for key, limit in thresholds.items())

    def _optimise_stage(self, stage: int, multipliers: dict[str, torch.Tensor]) -> dict[str, Any]:
        outer_iterations = {
            1: self.config.contact_outer_iterations,
            2: self.config.trunk_outer_iterations,
            3: self.config.posture_outer_iterations,
        }[stage]
        penalty = self.config.penalty_start
        history: list[dict[str, float]] = []
        best_state = (self.body_delta.detach().clone(), self.translation_delta.detach().clone())
        best_summary: dict[str, float] | None = None
        best_score: tuple[int, float, float] | None = None
        for outer in range(outer_iterations):
            optimizer = torch.optim.Adam([self.body_delta, self.translation_delta], lr=self.config.learning_rate)
            for _ in range(self.config.inner_iterations):
                optimizer.zero_grad(set_to_none=True)
                loss, residuals = self._objective(stage, penalty, multipliers)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite compensation loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.body_delta, self.translation_delta], 1.0)
                optimizer.step()
                self._clip_bounds()
            with torch.no_grad():
                _, residuals = self._objective(stage, penalty, multipliers)
                summary = self._residual_summary(residuals)
                score = self._stage_score(stage, summary)
                if best_score is None or score < best_score:
                    best_score = score
                    best_summary = dict(summary)
                    best_state = (self.body_delta.detach().clone(), self.translation_delta.detach().clone())
                row: dict[str, float] = {"stage": float(stage), "outer": float(outer), "penalty": float(penalty)}
                row.update({f"{key}_rms": value for key, value in summary.items() if key in self._stage_keys(stage)})
                history.append(row)
                for key in self._stage_keys(stage):
                    residual = residuals[key]
                    if not residual.numel():
                        continue
                    lam = multipliers.get(key)
                    if lam is None or lam.shape != residual.shape:
                        lam = torch.zeros_like(residual)
                    multipliers[key] = lam + penalty * residual.detach()
            penalty *= self.config.penalty_multiplier
        with torch.no_grad():
            self.body_delta.copy_(best_state[0])
            self.translation_delta.copy_(best_state[1])
            self._clip_bounds()
        if best_summary is None:
            best_summary = {key: 0.0 for key in ("contact", "orientation", "penetration", "trunk", "pelvis_neck", "pelvis_head", "support_drift")}
        return {
            "stage": stage,
            "history": history,
            "final_residuals": best_summary,
            "feasible": self._stage_feasible(stage, best_summary),
            "best_score": list(best_score or (0, 0.0, 0.0)),
        }

    def solve(self, target_delta_deg: float, *, initial_motion: torch.Tensor | None = None) -> dict[str, Any]:
        if not math.isfinite(float(target_delta_deg)) or not -10.0 <= float(target_delta_deg) <= 10.0:
            raise ValueError("target_delta_deg must lie in [-10,10]")
        self._target_root = target_root_rotation(self.base_root, float(target_delta_deg))
        if initial_motion is not None:
            self._initialise_from_motion(initial_motion)
        stages: list[dict[str, Any]] = []
        try:
            multipliers: dict[str, torch.Tensor] = {}
            stage1 = self._optimise_stage(1, multipliers)
            stages.append(stage1)
            stage1_motion = self._motion_from_state()
            if stage1["feasible"]:
                stage2 = self._optimise_stage(2, multipliers)
                # Later stages are lexicographic refinements.  A trunk/posture
                # update is not allowed to spend contact feasibility.  The
                # penalty terms remain useful for optimization, but the
                # acceptance rule below is an explicit hard guard over the
                # frozen previous-stage constraints.
                stage2["protected_stage"] = 1
                stage2["preserved_previous_stage"] = self._stage_feasible(1, stage2["final_residuals"])
                if not stage2["preserved_previous_stage"]:
                    stage2["final_residuals_before_restore"] = dict(stage2["final_residuals"])
                    self._initialise_from_motion(stage1_motion)
                    stage2["restored_to_previous_stage"] = True
                    stage2["feasible"] = False
                    stage2["final_residuals"] = dict(stage1["final_residuals"])
                    stages.append(stage2)
                    output = stage1_motion
                    final = stage1["final_residuals"]
                    return {
                        "protocol": self.config.protocol,
                        "status": "INFEASIBLE_WITHIN_BUDGET",
                        "feasible": False,
                        "target_delta_deg": float(target_delta_deg),
                        "frames": self.frames,
                        "active_joint_names": list(ACTIVE_JOINT_NAMES),
                        "trust_region": {"max_rotation_deg": self.config.max_rotation_deg, "max_translation_m": self.config.max_translation_m},
                        "stages": stages,
                        "motion": output,
                    }
                stages.append(stage2)
                if stage2["feasible"]:
                    stage2_motion = self._motion_from_state()
                    stage3 = self._optimise_stage(3, multipliers)
                    stage3["protected_stage"] = 2
                    stage3["preserved_previous_stage"] = self._stage_feasible(2, stage3["final_residuals"])
                    if not stage3["preserved_previous_stage"]:
                        stage3["final_residuals_before_restore"] = dict(stage3["final_residuals"])
                        self._initialise_from_motion(stage2_motion)
                        stage3["restored_to_previous_stage"] = True
                        stage3["feasible"] = False
                        stage3["final_residuals"] = dict(stage2["final_residuals"])
                    stages.append(stage3)
            output = self._motion_from_state()
            final = stages[-1]["final_residuals"]
            feasible = bool(stages[-1]["feasible"] and self._stage_feasible(3, final))
            return {
                "protocol": self.config.protocol,
                "status": "FEASIBLE" if feasible else "INFEASIBLE_WITHIN_BUDGET",
                "feasible": feasible,
                "target_delta_deg": float(target_delta_deg),
                "frames": self.frames,
                "active_joint_names": list(ACTIVE_JOINT_NAMES),
                "trust_region": {"max_rotation_deg": self.config.max_rotation_deg, "max_translation_m": self.config.max_translation_m},
                "stages": stages,
                "motion": output,
            }
        except Exception as exc:
            return {
                "protocol": self.config.protocol,
                "status": "FAILED",
                "target_delta_deg": float(target_delta_deg),
                "frames": self.frames,
                "error": repr(exc),
                "motion": self.m0.detach().clone(),
            }

    def solve_continuation(
        self,
        target_doses: tuple[float, ...] = (2.0, 5.0, 10.0),
        *,
        initial_motion: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        """Solve a fixed dose path while always carrying the best candidate."""

        current = initial_motion
        records: list[dict[str, Any]] = []
        for dose in target_doses:
            result = self.solve(float(dose), initial_motion=current)
            current = result["motion"]
            records.append(result)
        return records

    def _initialise_from_motion(self, motion: torch.Tensor) -> None:
        candidate = motion[0] if motion.ndim == 3 and motion.shape[0] == 1 else motion
        candidate = candidate[: self.frames].to(self.device).float()
        body = decode_rot6d_safe(candidate[..., MOTION_LAYOUT.body_pose].reshape(self.frames, 21, 6))
        for index, body_index in enumerate(ACTIVE_BODY_INDICES):
            delta = body[:, body_index] @ self.base_body[:, body_index].transpose(-1, -2)
            self.body_delta.data[:, index] = mat3x3_to_axis_angle(delta)
        self.translation_delta.data.copy_(candidate[..., MOTION_LAYOUT.root_translation] - self.base_translation)
        self._clip_bounds()

    def _motion_from_state(self) -> torch.Tensor:
        body, root, translation = self._state()
        direct = self.m0.clone()
        direct[..., MOTION_LAYOUT.body_pose] = encode_rot6d(body).reshape(self.frames, 126)
        direct[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(root)
        direct[..., MOTION_LAYOUT.root_translation] = translation
        return authority_project(direct.unsqueeze(0), valid_mask=self.valid_mask.unsqueeze(0), output_dtype=torch.float32).physical_motion[0].detach().cpu()


__all__ = ["PelvisCompensationConfig", "PelvisContactCompensationSolver", "ACTIVE_JOINT_NAMES", "ACTIVE_BODY_INDICES", "project_trust_region"]
