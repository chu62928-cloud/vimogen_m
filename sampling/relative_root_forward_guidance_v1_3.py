"""Shadow-pose hierarchical root-forward guidance (v1.3).

The v1.2 implementation rebuilt the complete 276-D representation at every
active diffusion step.  That is physically consistent, but it also sends the
large derived ``J/dJ/dR/dT`` projection back through a denoiser that was not
trained to treat those channels as an independent control signal.  v1.3 keeps
the rebuilt pose as a detached *physical shadow* and writes only the editable
direct pose channels back to the model endpoint.  The final output is still
fully pose-authoritative and is rebuilt once at the sampling boundary.

Root pitch/heading are the primary task.  A damped, iterated minimum-norm
Jacobian solve distributes a secondary trunk-direction correction over
spine1/2/3.  All acceptance checks use angular quantities separately; a
standardised 276-D change is recorded for diagnosis but is not a mixed-unit
control loss or trust region.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch

from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from motion_rep.pose_authority import _direct_streams, _prefix_source
from sampling.relative_root_forward_guidance_v1_2 import (
    EPS_DEG,
    PROTOCOL_NAME as V1_2_PROTOCOL_NAME,
    SPINE_BODY_INDICES,
    SPINE_JOINT_INDICES,
    UP,
    TrunkStabilizedRootForwardConfig,
    TrunkStabilizedRootForwardGuidance,
    _apply_world_axis,
    _trunk_direction,
    signed_axis_residual_deg,
    velocity_from_x0,
)


PROTOCOL_NAME = "vimogen_relative_root_forward_v1_3_shadow_pose_hierarchical"


@dataclass(frozen=True)
class ShadowPoseHierarchicalConfig(TrunkStabilizedRootForwardConfig):
    """Configuration for the v1.3 shadow-pose protocol."""

    protocol: str = PROTOCOL_NAME
    # v1.3 does not form a mixed-unit motion loss.  Keep the inherited field
    # for config compatibility, but make its protocol default explicit.
    motion_weight: float = 0.0
    max_solver_iterations: int = 4
    trunk_envelope_deg: float = 2.0
    temporal_step_limit_deg: float = 2.0
    jacobian_damping: float = 1e-3

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ShadowPoseHierarchicalConfig":
        values = dict(values or {})
        values.setdefault("protocol", PROTOCOL_NAME)
        # The parent implementation uses ``asdict(cls())`` and therefore
        # preserves the extra v1.3 fields when constructing this subclass.
        return super().from_mapping(values)  # type: ignore[return-value]

    def __post_init__(self) -> None:
        # Do not call v1.2/base post-init: both intentionally reject protocol
        # names that were unknown when those frozen protocols were written.
        if self.protocol != PROTOCOL_NAME:
            raise ValueError(f"protocol must be {PROTOCOL_NAME}")
        if self.guidance_strength < 0 or self.motion_weight < 0:
            raise ValueError("guidance_strength and motion_weight must be non-negative")
        if not 0 <= self.sigma_min <= self.sigma_max <= 1:
            raise ValueError("sigma window must satisfy 0 <= sigma_min <= sigma_max <= 1")
        if self.base_step_deg <= 0 or self.max_correction_rms <= 0:
            raise ValueError("base_step_deg and max_correction_rms must be positive")
        if self.residual_gain <= 0 or self.max_step_deg <= 0:
            raise ValueError("residual_gain and max_step_deg must be positive")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")
        if self.heading_gain <= 0 or self.trunk_gain <= 0:
            raise ValueError("heading_gain and trunk_gain must be positive")
        if self.max_heading_step_deg <= 0 or self.max_trunk_step_deg <= 0:
            raise ValueError("heading/trunk step limits must be positive")
        if self.finite_diff_deg <= 0 or self.control_tolerance_deg < 0:
            raise ValueError("finite_diff_deg must be positive and tolerance non-negative")
        if self.spine_budget_factor <= 0:
            raise ValueError("spine_budget_factor must be positive")
        if self.protocol != PROTOCOL_NAME:
            raise ValueError(f"protocol must be {PROTOCOL_NAME}")
        if self.max_solver_iterations < 1:
            raise ValueError("max_solver_iterations must be positive")
        if self.trunk_envelope_deg <= 0 or self.temporal_step_limit_deg <= 0:
            raise ValueError("trunk envelope and temporal step limits must be positive")
        if self.jacobian_damping <= 0:
            raise ValueError("jacobian_damping must be positive")


def _masked_rms(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0 or not torch.isfinite(selected).all():
        return torch.as_tensor(float("inf"), dtype=values.dtype, device=values.device)
    return torch.sqrt(selected.square().mean())


class ShadowPoseHierarchicalRootForwardGuidance(TrunkStabilizedRootForwardGuidance):
    """v1.3 guidance with a physical shadow and direct-pose-only injection."""

    PROTOCOL = PROTOCOL_NAME

    def __init__(self, *args, config=None, **kwargs) -> None:
        if config is None:
            config = ShadowPoseHierarchicalConfig()
        elif not isinstance(config, ShadowPoseHierarchicalConfig):
            config = ShadowPoseHierarchicalConfig.from_mapping(asdict(config))
        super().__init__(*args, config=config, **kwargs)

    @classmethod
    def _from_cached(cls, source: "ShadowPoseHierarchicalRootForwardGuidance", index: int):
        obj = object.__new__(cls)
        obj.config = source.config
        obj.mean = source.mean if source.mean.ndim == 1 else source.mean[index:index + 1]
        obj.std = source.std if source.std.ndim == 1 else source.std[index:index + 1]
        obj.valid_mask = source.valid_mask[index:index + 1]
        obj.skeleton = source.skeleton
        obj.baseline_motion_norm = source.baseline_motion_norm[index:index + 1]
        obj.baseline_physical = source.baseline_physical[index:index + 1]
        obj.baseline_projection_audits = (source.baseline_projection_audits[index],)
        targets = source.targets
        from motion_rep.pose_authority import RelativeRootForwardTargets

        obj.targets = RelativeRootForwardTargets(
            targets.f0[index:index + 1],
            targets.h0[index:index + 1],
            targets.r0[index:index + 1],
            targets.phi0_deg[index:index + 1],
            targets.target_forward[index:index + 1],
            targets.target_phi_deg[index:index + 1],
            targets.valid_mask[index:index + 1],
            targets.delta_deg,
        )
        obj.target_delta_deg = source.target_delta_deg
        obj.baseline_trunk = source.baseline_trunk[index:index + 1]
        obj.last_diagnostics = {"protocol": obj.PROTOCOL, "enabled": obj.enabled, "active": False}
        return obj

    def protocol_record(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "supersedes": V1_2_PROTOCOL_NAME,
            "state_boundary": "model_state_plus_physical_shadow_state",
            "authority": "body_pose/root_rotation/root_translation",
            "derived": "J/dJ/dR/dT_from_fk_and_forward_difference_at_final_boundary",
            "model_injected_channels": ["root_rotation", "body_pose.spine1", "body_pose.spine2", "body_pose.spine3"],
            "shadow_only_channels": ["J", "dJ", "dR", "dT"],
            "loss_framework": "separate_angular_constraints_with_physical_trust_regions",
            "control_constraints": ["pitch", "full_forward", "heading", "trunk_direction"],
            "solver": "iterated_damped_minimum_norm_root_plus_spine",
            "config": asdict(self.config),
            "baseline_projection_audits": list(self.baseline_projection_audits),
        }

    def _control_constraints_ok(
        self,
        candidate_metrics: dict[str, torch.Tensor],
        reference_metrics: dict[str, torch.Tensor],
        frame_mask: torch.Tensor,
    ) -> tuple[bool, bool]:
        """Apply primary root progress and bounded secondary envelopes.

        All quantities are degrees.  Root pitch and complete forward errors
        must not increase.  Heading and trunk may use the explicit 2-degree
        feasible envelope when the current endpoint is already better than
        that envelope; they may never cross it during intermediate denoising.
        """

        before = {name: _masked_rms(reference_metrics[name], frame_mask) for name in ("pitch", "forward", "heading", "trunk")}
        after = {name: _masked_rms(candidate_metrics[name], frame_mask) for name in before}
        if not all(torch.isfinite(value) for value in (*before.values(), *after.values())):
            return False, False
        root_ok = all(after[name] <= before[name] + self.config.control_tolerance_deg for name in ("pitch", "forward"))
        secondary_ok = (
            after["heading"] <= torch.maximum(before["heading"], torch.as_tensor(self.config.trunk_envelope_deg, device=after["heading"].device, dtype=after["heading"].dtype)) + self.config.control_tolerance_deg
            and after["trunk"] <= torch.maximum(before["trunk"], torch.as_tensor(self.config.trunk_envelope_deg, device=after["trunk"].device, dtype=after["trunk"].dtype)) + self.config.control_tolerance_deg
        )
        progress = any(after[name] < before[name] - self.config.control_tolerance_deg for name in ("pitch", "forward"))
        return bool(root_ok and secondary_ok), bool(progress)

    def _active_direct_mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(276, dtype=torch.bool, device=device)
        mask[MOTION_LAYOUT.root_rotation] = True
        for body_index in SPINE_BODY_INDICES:
            mask[body_index * 6:(body_index + 1) * 6] = True
        return mask

    def _direct_candidate_norm(
        self,
        x0_hat: torch.Tensor,
        x0_physical: torch.Tensor,
        pitch_deg: torch.Tensor,
        heading_deg: torch.Tensor,
        beta_deg: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace only editable direct pose channels in the model endpoint."""

        body, root, translation, _pose_mask, _ = self._direct_with_root(
            x0_physical, pitch_deg, heading_deg
        )
        body = self._apply_spine_local(body, root, translation, beta_deg)
        direct_physical = x0_physical.clone()
        direct_physical[..., MOTION_LAYOUT.body_pose] = x0_physical[..., MOTION_LAYOUT.body_pose]
        encoded_body = encode_rot6d(body).reshape(*body.shape[:2], 126)
        encoded_root = encode_rot6d(root)
        for body_index in SPINE_BODY_INDICES:
            start = body_index * 6
            direct_physical[..., MOTION_LAYOUT.body_pose.start + start:MOTION_LAYOUT.body_pose.start + start + 6] = encoded_body[..., start:start + 6]
        direct_physical[..., MOTION_LAYOUT.root_rotation] = encoded_root
        direct_norm = self._standardize(direct_physical)
        editable = self._active_direct_mask(x0_hat.device)
        editable = editable.view(1, 1, -1)
        candidate = torch.where(editable, direct_norm, x0_hat)
        return torch.where(frame_mask.unsqueeze(-1), candidate, x0_hat)

    def _iterative_proposal(
        self,
        x0_physical: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Solve root and trunk corrections with a short closed-loop solve."""

        source, _ = _prefix_source(x0_physical, frame_mask)
        body, root, translation = _direct_streams(source)
        f, h, _, _ = __import__("motion_rep.pose_authority", fromlist=["_root_forward"])._root_forward(root)
        target_f = self.targets.target_forward.to(device=f.device, dtype=f.dtype)
        target_h = self.targets.h0.to(device=f.device, dtype=f.dtype)
        r0 = self.targets.r0.to(device=f.device, dtype=f.dtype)
        pitch, pitch_valid = signed_axis_residual_deg(f, target_f, r0, frame_mask, eps=self.config.eps)
        up = UP.to(device=f.device, dtype=f.dtype).expand_as(f)
        heading, heading_valid = signed_axis_residual_deg(h, target_h, up, frame_mask, eps=self.config.eps)
        if not bool((pitch_valid & heading_valid)[frame_mask].all()):
            raise ValueError("degenerate root pitch or heading projection")
        pitch_step = (self.config.residual_gain * pitch).clamp(-self.config.max_step_deg, self.config.max_step_deg)
        heading_step = (self.config.heading_gain * heading).clamp(-self.config.max_heading_step_deg, self.config.max_heading_step_deg)
        beta = torch.zeros((*frame_mask.shape, 3), dtype=x0_physical.dtype, device=x0_physical.device)
        iterations = 0
        sensitivity_min = float("inf")

        for iterations in range(1, self.config.max_solver_iterations + 1):
            body_root, root_now, trans_now, pose_mask, _ = self._direct_with_root(x0_physical, pitch_step, heading_step)
            body_now = self._apply_spine_local(body_root, root_now, trans_now, beta)
            shadow_now = __import__("motion_rep.pose_authority", fromlist=["_authority_from_streams"])._authority_from_streams(
                body_now, root_now, trans_now, pose_mask, skeleton=self.skeleton
            )
            trunk_now = _trunk_direction(shadow_now, frame_mask)
            residual, residual_valid = signed_axis_residual_deg(
                trunk_now,
                self.baseline_trunk.to(device=trunk_now.device, dtype=trunk_now.dtype),
                r0,
                frame_mask,
                eps=self.config.eps,
            )
            if not bool(residual_valid[frame_mask].all()):
                raise ValueError("degenerate trunk projection")
            sensitivity = self._spine_sensitivity(
                body_now, root_now, trans_now, pose_mask, frame_mask, trunk_now
            )
            denom = sensitivity.square().sum(-1)
            valid = torch.isfinite(sensitivity).all(-1) & torch.isfinite(denom) & (denom > self.config.eps)
            if not bool(valid[frame_mask].all()):
                raise ValueError("degenerate spine sensitivity")
            sensitivity_min = min(sensitivity_min, float(torch.linalg.vector_norm(sensitivity[frame_mask], dim=-1).min().detach().cpu()))
            # beta is expressed in degrees, and the task Jacobian is degrees
            # per degree.  Damping avoids unstable updates near a singular
            # sagittal configuration while retaining the minimum-norm solve.
            damping = self.config.jacobian_damping
            beta_step = -self.config.trunk_gain * residual.unsqueeze(-1) * sensitivity / (denom.unsqueeze(-1) + damping * damping)
            beta_step = beta_step * frame_mask.unsqueeze(-1).to(beta_step.dtype)
            cap = min(self.config.max_trunk_step_deg, self.config.spine_budget_factor * abs(self.target_delta_deg))
            beta_next = beta + beta_step
            total = beta_next.abs().sum(-1)
            scale = torch.minimum(torch.ones_like(total), torch.as_tensor(cap, device=total.device, dtype=total.dtype) / total.clamp_min(self.config.eps))
            beta = beta_next * scale.unsqueeze(-1)
            if float(torch.sqrt(residual[frame_mask].square().mean()).detach().cpu()) <= max(0.05, self.config.control_tolerance_deg):
                break

        return pitch_step, heading_step, beta, {
            "pitch_residual_rms_deg": float(torch.sqrt(pitch[frame_mask].square().mean()).detach().cpu()),
            "heading_residual_rms_deg": float(torch.sqrt(heading[frame_mask].square().mean()).detach().cpu()),
            "pitch_proposal_rms_deg": float(torch.sqrt(pitch_step[frame_mask].square().mean()).detach().cpu()),
            "heading_proposal_rms_deg": float(torch.sqrt(heading_step[frame_mask].square().mean()).detach().cpu()),
            "spine_proposal_rms_deg": float(torch.sqrt(beta[frame_mask].square().mean()).detach().cpu()),
            "spine_proposal_total_max_deg": float(beta.abs().sum(-1)[frame_mask].max().detach().cpu()),
            "solver_iterations": int(iterations),
            "spine_sensitivity_min": sensitivity_min,
        }

    def _temporal_project(self, values: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """Project a correction curve onto a bounded first-difference set."""

        result = values.clone()
        limit = float(self.config.temporal_step_limit_deg)
        for _ in range(6):
            for direction in (1, -1):
                indices = range(1, values.shape[1]) if direction == 1 else range(values.shape[1] - 2, -1, -1)
                for t in indices:
                    pair = frame_mask[:, t] & frame_mask[:, t - 1]
                    delta = result[:, t] - result[:, t - 1]
                    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                    scale = torch.minimum(torch.ones_like(norm), torch.as_tensor(limit, device=norm.device, dtype=norm.dtype) / norm.clamp_min(self.config.eps))
                    proposed = result[:, t - 1] + delta * scale
                    result[:, t] = torch.where(pair.unsqueeze(-1), proposed, result[:, t])
        return result * frame_mask.unsqueeze(-1).to(result.dtype) if result.ndim == 3 else result * frame_mask.to(result.dtype)

    def _tail_budget_ok(self, before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor) -> bool:
        """Check only the final eight valid pairs; full-sequence values are audit-only."""

        from motion_rep.pose_authority import _geodesic, _root_forward
        source_before, _ = _prefix_source(before, mask)
        source_after, _ = _prefix_source(after, mask)
        root_before = __import__("motion_rep.phase1", fromlist=["decode_rot6d_safe"]).decode_rot6d_safe(source_before[..., MOTION_LAYOUT.root_rotation])
        root_after = __import__("motion_rep.phase1", fromlist=["decode_rot6d_safe"]).decode_rot6d_safe(source_after[..., MOTION_LAYOUT.root_rotation])
        step_before = _geodesic(root_before[:, 1:], root_before[:, :-1]) * 180.0 / math.pi
        step_after = _geodesic(root_after[:, 1:], root_after[:, :-1]) * 180.0 / math.pi
        pitch_before = _root_forward(root_before)[3]
        pitch_after = _root_forward(root_after)[3]
        delta_so3 = (step_after - step_before).abs()
        delta_pitch = ((pitch_after[:, 1:] - pitch_after[:, :-1]) - (pitch_before[:, 1:] - pitch_before[:, :-1])).abs()
        for index in range(mask.shape[0]):
            pairs = mask[index, 1:] & mask[index, :-1]
            positions = torch.nonzero(pairs, as_tuple=False).flatten()[-8:]
            if positions.numel() == 0 or not bool(torch.isfinite(delta_so3[index, positions]).all() and torch.isfinite(delta_pitch[index, positions]).all()):
                return False
            if float(delta_so3[index, positions].max()) > 2.0 or float(delta_pitch[index, positions].max()) > 2.0:
                return False
        return True

    @staticmethod
    def _channel_delta_summary(delta: torch.Tensor, frame_mask: torch.Tensor) -> dict[str, Any]:
        groups = {
            "body_pose": MOTION_LAYOUT.body_pose,
            "J": MOTION_LAYOUT.joints,
            "dJ": MOTION_LAYOUT.joints_velocity,
            "R": MOTION_LAYOUT.root_rotation,
            "dR": MOTION_LAYOUT.root_rotation_velocity,
            "T": MOTION_LAYOUT.root_translation,
            "dT": MOTION_LAYOUT.root_translation_velocity,
        }
        result: dict[str, Any] = {}
        mask = frame_mask.unsqueeze(-1)
        for name, span in groups.items():
            value = delta[..., span][mask.expand_as(delta[..., span])]
            if value.numel():
                absolute = value.abs()
                result[name] = {
                    "rms": float(torch.sqrt(value.square().mean()).detach().cpu()),
                    "p95": float(torch.quantile(absolute, 0.95).detach().cpu()),
                    "max": float(absolute.max().detach().cpu()),
                }
            else:
                result[name] = {"rms": None, "p95": None, "max": None}
        return result

    def correct_velocity(self, *, x_sigma, velocity, sigma, valid_mask, return_trace=False):
        if x_sigma.shape != velocity.shape or x_sigma.ndim != 3 or x_sigma.shape[-1] != 276:
            raise ValueError("x_sigma and velocity must have shape [B,T,276]")
        if valid_mask.shape != x_sigma.shape[:2] or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool[B,T]")
        if not torch.equal(valid_mask, self.valid_mask.to(valid_mask.device)):
            raise ValueError("sampler valid_mask differs from the frozen guidance mask")
        sigma_value = float(torch.as_tensor(sigma).detach().cpu().item())
        cfg = self.config
        diagnostics: dict[str, Any] = {
            "protocol": self.PROTOCOL,
            "enabled": cfg.enabled,
            "active": False,
            "sigma": sigma_value,
            "target_delta_deg": self.target_delta_deg,
            "accepted": False,
        }
        x0_hat = (x_sigma.float() - sigma_value * velocity.float()).detach()
        inactive = (
            not math.isfinite(sigma_value) or not cfg.enabled or cfg.guidance_strength <= cfg.eps
            or abs(self.target_delta_deg) <= cfg.eps or sigma_value <= cfg.eps
            # Include a schedule endpoint that differs from the configured
            # bound only by floating-point round-off.  The final active
            # sigma is often serialized as 0.066287899... while the config
            # contains 0.0662879; treating that endpoint as inactive would
            # silently drop the last planned correction opportunity.
            or sigma_value + cfg.eps < cfg.sigma_min
            or sigma_value - cfg.eps > cfg.sigma_max
        )
        if inactive:
            if return_trace:
                diagnostics["trace"] = {
                    "velocity_model": velocity.detach().float().clone(),
                    "v_cfg": velocity.detach().float().clone(),
                    "x0_hat": x0_hat.clone(),
                    "x0_guided": x0_hat.clone(),
                    "x0_reconciled": x0_hat.clone(),
                }
            self.last_diagnostics = diagnostics
            return velocity, diagnostics
        if not torch.isfinite(x0_hat).all():
            diagnostics["rejected_reason"] = "non_finite_x0_hat"
            self.last_diagnostics = diagnostics
            return velocity, diagnostics

        with torch.amp.autocast(device_type=x_sigma.device.type, enabled=False):
            x0_physical = self._physical(x0_hat)
            frame_mask = valid_mask
            try:
                base_phys = self._build_candidate(
                    x0_physical,
                    torch.zeros_like(frame_mask, dtype=torch.float32),
                    torch.zeros_like(frame_mask, dtype=torch.float32),
                    torch.zeros((*frame_mask.shape, 3), dtype=torch.float32, device=x_sigma.device),
                )
                base_metrics = self._metrics(base_phys)
                pitch, heading, beta, proposal_diag = self._iterative_proposal(x0_physical, frame_mask)
                stacked = torch.cat((pitch.unsqueeze(-1), heading.unsqueeze(-1), beta), dim=-1)
                stacked = self._temporal_project(stacked, frame_mask)
                pitch, heading, beta = stacked[..., 0], stacked[..., 1], stacked[..., 2:]
            except (ValueError, RuntimeError) as exc:
                diagnostics["rejected_reason"] = str(exc)
                self.last_diagnostics = diagnostics
                return velocity, diagnostics

            base_norm = self._standardize(base_phys)
            projection_delta = base_norm - x0_hat
            accepted = None
            last_check: dict[str, Any] = {}
            for attempt in range(cfg.max_backtracks + 1):
                scale = 0.5 ** attempt
                pitch_try, heading_try, beta_try = pitch * scale, heading * scale, beta * scale
                shadow_candidate = self._build_candidate(x0_physical, pitch_try, heading_try, beta_try)
                shadow_metrics = self._metrics(shadow_candidate)
                direct_candidate = self._direct_candidate_norm(
                    x0_hat, x0_physical, pitch_try, heading_try, beta_try, frame_mask
                )
                finite = bool(torch.isfinite(shadow_candidate).all() and torch.isfinite(direct_candidate).all())
                controls_ok, progress = self._control_constraints_ok(shadow_metrics, base_metrics, frame_mask)
                tail_ok = self._tail_budget_ok(base_phys, shadow_candidate, frame_mask)
                spine_sum_ok = bool(beta_try.abs().sum(-1)[frame_mask].max() <= min(cfg.max_trunk_step_deg, cfg.spine_budget_factor * abs(self.target_delta_deg)) + cfg.eps)
                last_check = {
                    "attempt": int(attempt),
                    "controls_ok": controls_ok,
                    "progress": progress,
                    "tail_ok": tail_ok,
                    "spine_budget_ok": spine_sum_ok,
                    "finite": finite,
                }
                if finite and controls_ok and progress and tail_ok and spine_sum_ok:
                    accepted = (direct_candidate, shadow_candidate, pitch_try, heading_try, beta_try, shadow_metrics, scale, attempt)
                    break

            diagnostics.update({"active": True, **proposal_diag})
            diagnostics["authority_projection_delta_rms_276d"] = float(_masked_rms(projection_delta, frame_mask.unsqueeze(-1).expand_as(projection_delta)).detach().cpu())
            diagnostics["authority_projection_channels"] = self._channel_delta_summary(projection_delta, frame_mask)
            diagnostics["valid_frames"] = int(frame_mask.sum().detach().cpu())
            diagnostics["backtrack_count"] = int(last_check.get("attempt", cfg.max_backtracks))
            if accepted is None:
                diagnostics["rejected_reason"] = "no_feasible_shadow_constraint_candidate"
                diagnostics["last_candidate_check"] = last_check
                if return_trace:
                    diagnostics["trace"] = {
                        "velocity_model": velocity.detach().float().clone(),
                        "v_cfg": velocity.detach().float().clone(),
                        "x0_hat": x0_hat.clone(),
                        "x0_guided": x0_hat.clone(),
                        "x0_reconciled": base_norm.detach().clone(),
                        "x0_physical_shadow": base_norm.detach().clone(),
                        "authority_projection_delta": projection_delta.detach().clone(),
                    }
                self.last_diagnostics = diagnostics
                return velocity, diagnostics

            direct_candidate, shadow_candidate, pitch_try, heading_try, beta_try, shadow_metrics, scale, attempt = accepted
            corrected_velocity = velocity_from_x0(x_sigma.float(), direct_candidate, sigma_value).to(dtype=velocity.dtype)
            corrected_velocity = torch.where(frame_mask.unsqueeze(-1), corrected_velocity, velocity)
            direct_guidance_delta = direct_candidate - base_norm
            actual_injected_delta = direct_candidate - x0_hat
            diagnostics.update({
                "accepted": True,
                "accepted_scale": float(scale),
                "accepted_root_pitch_rms_deg": float(torch.sqrt(pitch_try[frame_mask].square().mean()).detach().cpu()),
                "accepted_heading_rms_deg": float(torch.sqrt(heading_try[frame_mask].square().mean()).detach().cpu()),
                "accepted_spine_rms_deg": float(torch.sqrt(beta_try[frame_mask].square().mean()).detach().cpu()),
                "accepted_alpha_rms_deg": float(torch.sqrt((pitch_try[frame_mask].square() + heading_try[frame_mask].square()).mean()).detach().cpu()),
                "accepted_alpha_max_deg": float(torch.sqrt(pitch_try.square() + heading_try.square())[frame_mask].max().detach().cpu()),
                "actual_injected_rms_276d": float(_masked_rms(actual_injected_delta, frame_mask.unsqueeze(-1).expand_as(actual_injected_delta)).detach().cpu()),
                "direct_guidance_rms_276d": float(_masked_rms(direct_guidance_delta, frame_mask.unsqueeze(-1).expand_as(direct_guidance_delta)).detach().cpu()),
                "authority_projection_channels": self._channel_delta_summary(projection_delta, frame_mask),
                "direct_guidance_channels": self._channel_delta_summary(direct_guidance_delta, frame_mask),
                "actual_injected_channels": self._channel_delta_summary(actual_injected_delta, frame_mask),
                "pitch_rms_before_deg": float(_masked_rms(base_metrics["pitch"], frame_mask).detach().cpu()),
                "pitch_rms_after_deg": float(_masked_rms(shadow_metrics["pitch"], frame_mask).detach().cpu()),
                "forward_rms_before_deg": float(_masked_rms(base_metrics["forward"], frame_mask).detach().cpu()),
                "forward_rms_after_deg": float(_masked_rms(shadow_metrics["forward"], frame_mask).detach().cpu()),
                "heading_rms_before_deg": float(_masked_rms(base_metrics["heading"], frame_mask).detach().cpu()),
                "heading_rms_after_deg": float(_masked_rms(shadow_metrics["heading"], frame_mask).detach().cpu()),
                "trunk_rms_before_deg": float(_masked_rms(base_metrics["trunk"], frame_mask).detach().cpu()),
                "trunk_rms_after_deg": float(_masked_rms(shadow_metrics["trunk"], frame_mask).detach().cpu()),
            })
            if return_trace:
                diagnostics["trace"] = {
                    "velocity_model": velocity.detach().float().clone(),
                    "v_cfg": velocity.detach().float().clone(),
                    "x0_hat": x0_hat.clone(),
                    "x0_guided": direct_candidate.detach().clone(),
                    "x0_reconciled": self._standardize(shadow_candidate).detach().clone(),
                    "x0_physical_shadow": self._standardize(shadow_candidate).detach().clone(),
                    "authority_projection_delta": projection_delta.detach().clone(),
                    "direct_guidance_delta": direct_guidance_delta.detach().clone(),
                    "actual_injected_delta": actual_injected_delta.detach().clone(),
                }
        self.last_diagnostics = diagnostics
        return corrected_velocity, diagnostics


__all__ = [
    "PROTOCOL_NAME",
    "ShadowPoseHierarchicalConfig",
    "ShadowPoseHierarchicalRootForwardGuidance",
]
