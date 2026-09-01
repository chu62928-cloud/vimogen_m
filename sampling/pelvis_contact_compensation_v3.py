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


@dataclass(frozen=True)
class PelvisCompensationConfig:
    protocol: str = "vimogen_pelvis_contact_compensation_v3"
    max_rotation_deg: float = 30.0
    max_translation_m: float = 0.05
    stable_confidence: float = STABLE_CONFIDENCE
    contact_height_m: float = CONTACT_HEIGHT_M
    contact_speed_m_per_frame: float = CONTACT_SPEED_M_PER_FRAME
    flat_gap_m: float = FLAT_GAP_M
    outer_iterations: int = 4
    inner_iterations: int = 30
    learning_rate: float = 0.03
    penalty_start: float = 10.0
    penalty_multiplier: float = 10.0
    trunk_tolerance_deg: float = 2.0
    contact_tolerance_m: float = 0.001
    smooth_velocity_weight: float = 0.10
    smooth_acceleration_weight: float = 0.05
    posture_weight: float = 0.01

    def validate(self) -> None:
        if self.max_rotation_deg <= 0 or self.max_translation_m <= 0:
            raise ValueError("trust-region bounds must be positive")
        if self.outer_iterations < 1 or self.inner_iterations < 1:
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
        residuals: dict[str, list[torch.Tensor]] = {"contact": [], "orientation": [], "penetration": [], "trunk": []}
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
        return {
            key: torch.cat([value.reshape(-1, 3 if key in ("contact", "orientation", "trunk") else 1) for value in values], dim=0) if values else torch.zeros((0, 3 if key in ("contact", "orientation", "trunk") else 1), device=self.device)
            for key, values in residuals.items()
        }

    def _objective(self, stage: int, penalty: float, multipliers: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self._forward()
        residuals = self._residuals(output)
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for key in ("contact", "orientation", "penetration"):
            residual = residuals[key]
            if residual.numel():
                lam = multipliers[key]
                if lam.shape != residual.shape:
                    lam = torch.zeros_like(residual)
                    multipliers[key] = lam
                loss = loss + (lam * residual).mean() + 0.5 * penalty * residual.square().mean()
        if stage >= 2 and residuals["trunk"].numel():
            residual = residuals["trunk"]
            lam = multipliers["trunk"]
            if lam.shape != residual.shape:
                lam = torch.zeros_like(residual)
                multipliers["trunk"] = lam
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
            self.body_delta.clamp_(-math.radians(self.config.max_rotation_deg), math.radians(self.config.max_rotation_deg))
            self.translation_delta.clamp_(-self.config.max_translation_m, self.config.max_translation_m)

    def _optimise_stage(self, stage: int) -> dict[str, Any]:
        multipliers = {
            "contact": torch.zeros((max(1, self.frames * 6), 1), device=self.device),
            "orientation": torch.zeros((max(1, self.frames), 3), device=self.device),
            "penetration": torch.zeros((max(1, self.frames * 2), 1), device=self.device),
            "trunk": torch.zeros((max(1, self.frames), 3), device=self.device),
        }
        penalty = self.config.penalty_start
        history: list[dict[str, float]] = []
        residuals: dict[str, torch.Tensor] = {}
        for outer in range(self.config.outer_iterations):
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
                for key, residual in residuals.items():
                    if residual.numel():
                        if multipliers[key].shape != residual.shape:
                            multipliers[key] = torch.zeros_like(residual)
                        multipliers[key] = multipliers[key] + penalty * residual.detach()
                        history.append({"stage": float(stage), "outer": float(outer), "penalty": float(penalty), f"{key}_rms": float(torch.sqrt(residual.square().mean()).item())})
            penalty *= self.config.penalty_multiplier
        return {"stage": stage, "history": history, "final_residuals": {key: float(torch.sqrt(value.square().mean()).item()) if value.numel() else 0.0 for key, value in residuals.items()}}

    def solve(self, target_delta_deg: float, *, initial_motion: torch.Tensor | None = None) -> dict[str, Any]:
        if not math.isfinite(float(target_delta_deg)) or not -10.0 <= float(target_delta_deg) <= 10.0:
            raise ValueError("target_delta_deg must lie in [-10,10]")
        self._target_root = target_root_rotation(self.base_root, float(target_delta_deg))
        if initial_motion is not None:
            self._initialise_from_motion(initial_motion)
        stages = []
        try:
            stages.append(self._optimise_stage(1))
            stages.append(self._optimise_stage(2))
            stages.append(self._optimise_stage(3))
            output = self._motion_from_state()
            final = stages[-1]["final_residuals"]
            feasible = bool(
                final.get("contact", 0.0) <= self.config.contact_tolerance_m
                and final.get("penetration", 0.0) <= self.config.contact_tolerance_m
                and final.get("orientation", 0.0) <= math.sin(math.radians(2.0))
                and final.get("trunk", 0.0) <= math.sin(math.radians(self.config.trunk_tolerance_deg))
            )
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


__all__ = ["PelvisCompensationConfig", "PelvisContactCompensationSolver", "ACTIVE_JOINT_NAMES", "ACTIVE_BODY_INDICES"]
