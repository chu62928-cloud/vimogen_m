"""Flow-matching sampler for the frozen ViMoGen M0 protocol.

The sampler deliberately requires the initial noise tensor.  Noise creation
belongs to the caller so an old batch can be captured once and replayed
without changing the historical random-number semantics.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch

from utils import smooth_motion_rep


@dataclass
class FlowSampleResult:
    """The tensors needed to audit a sampling run."""

    initial_noise: torch.Tensor
    raw: torch.Tensor
    official_pre_cast: torch.Tensor
    official: torch.Tensor
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    # Optional per-step tensors used only by residual-source diagnostics.
    # ``None`` is the default so the frozen M0/M1 path keeps its historical
    # memory use and output behavior.
    trace: Optional[dict[str, torch.Tensor]] = None
    # Optional explicit control-aware representation reconciliation.  Legacy
    # M0/M1 callers leave this as None and retain bitwise-compatible outputs.
    reconciled: Optional[torch.Tensor] = None
    representation_protocol: Optional[str] = None
    # Opt-in authoritative outputs for absolute-pelvis and relative-root
    # protocols.  They remain None for every historical M0/M1 call.
    g0: Optional[torch.Tensor] = None
    g1: Optional[torch.Tensor] = None
    guidance_summary: Optional[dict] = None


class FlowSampler:
    """Run the frozen 50-step M0 flow-matching sampling procedure."""

    def __init__(
        self,
        scheduler,
        num_inference_steps: int = 50,
        denoising_strength: float = 0.7,
        cfg_scale: float = 5.0,
        smooth_kernel_size: int = 5,
        smooth_sigma: float = 1.0,
    ):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.denoising_strength = denoising_strength
        self.cfg_scale = cfg_scale
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma

    @staticmethod
    def _validate_inputs(
        initial_noise: torch.Tensor,
        valid_mask: torch.Tensor,
        ref_motion: torch.Tensor,
        ref_motion_mask: torch.Tensor,
    ) -> None:
        if initial_noise.ndim != 3:
            raise ValueError(
                f"initial_noise must be [B,T,C], got {tuple(initial_noise.shape)}"
            )
        if valid_mask.shape != initial_noise.shape[:2]:
            raise ValueError(
                "valid_mask must match initial_noise [B,T], "
                f"got {tuple(valid_mask.shape)} vs {tuple(initial_noise.shape[:2])}"
            )
        if ref_motion.ndim != 3 or ref_motion.shape[:2] != initial_noise.shape[:2]:
            raise ValueError(
                "ref_motion must have the same [B,T] as initial_noise, "
                f"got {tuple(ref_motion.shape)} vs {tuple(initial_noise.shape)}"
            )
        if ref_motion_mask.shape != initial_noise.shape[:2]:
            raise ValueError(
                "ref_motion_mask must match initial_noise [B,T], "
                f"got {tuple(ref_motion_mask.shape)} vs {tuple(initial_noise.shape[:2])}"
            )

    @torch.no_grad()
    def generate(
        self,
        *,
        model,
        prompt_emb: torch.Tensor,
        prompt_emb_null: torch.Tensor,
        initial_noise: torch.Tensor,
        valid_mask: torch.Tensor,
        ref_motion: torch.Tensor,
        ref_motion_mask: torch.Tensor,
        condition_on_text: bool = False,
        attend_to_text_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        show_progress: bool = False,
        batch_invariant: bool = False,
        m1_guidance=None,
        absolute_mean_guidance=None,
        relative_root_forward_guidance=None,
        trace_enabled: bool = False,
        reconciliation_config: Optional[dict] = None,
        motion_mean: Optional[torch.Tensor] = None,
        motion_std: Optional[torch.Tensor] = None,
    ) -> FlowSampleResult:
        """Generate M0 output from an externally supplied ``initial_noise``.

        The model is called with the same two-branch CFG batch as the legacy
        validation path.  ``raw`` is cloned immediately before the official
        per-channel smoothing; ``official`` includes that smoothing.
        """
        self._validate_inputs(
            initial_noise, valid_mask, ref_motion, ref_motion_mask
        )
        guidance_count = sum(
            item is not None
            for item in (m1_guidance, absolute_mean_guidance, relative_root_forward_guidance)
        )
        if guidance_count > 1:
            raise ValueError(
                "m1_guidance, absolute_mean_guidance, and "
                "relative_root_forward_guidance are mutually exclusive"
            )
        if guidance_count and reconciliation_config and bool(reconciliation_config.get("enabled", False)):
            raise ValueError(
                "guided protocol owns the final reconciliation boundary; "
                "do not also pass reconciliation_config"
            )
        guidance_hook = (
            absolute_mean_guidance
            if absolute_mean_guidance is not None
            else relative_root_forward_guidance
            if relative_root_forward_guidance is not None
            else m1_guidance
        )
        if batch_invariant and initial_noise.shape[0] > 1:
            """Run each sample through the same single-sample kernel path.

            Some CUDA matrix-multiplication/attention kernels use a different
            reduction path when the batch dimension changes.  The opt-in
            sample-noise protocol promises paired single-sample outputs, so
            this mode trades throughput for that invariant.  The historical
            M0 path leaves it disabled.
            """
            per_sample = []
            for index in range(initial_noise.shape[0]):
                per_sample.append(
                    self.generate(
                        model=model,
                        prompt_emb=prompt_emb[index:index + 1],
                        prompt_emb_null=prompt_emb_null[index:index + 1],
                        initial_noise=initial_noise[index:index + 1],
                        valid_mask=valid_mask[index:index + 1],
                        ref_motion=ref_motion[index:index + 1],
                        ref_motion_mask=ref_motion_mask[index:index + 1],
                        condition_on_text=condition_on_text,
                        attend_to_text_mask=(
                            None if attend_to_text_mask is None
                            else attend_to_text_mask[index:index + 1]
                        ),
                        dtype=dtype,
                        show_progress=show_progress and index == 0,
                        batch_invariant=False,
                        m1_guidance=(
                            None if absolute_mean_guidance is not None
                            else None if m1_guidance is None
                            else m1_guidance.slice(index)
                        ),
                        absolute_mean_guidance=(
                            None if absolute_mean_guidance is None
                            else absolute_mean_guidance.slice(index)
                        ),
                        relative_root_forward_guidance=(
                            None if relative_root_forward_guidance is None
                            else relative_root_forward_guidance.slice(index)
                        ),
                        trace_enabled=trace_enabled,
                        reconciliation_config=reconciliation_config,
                        motion_mean=(None if motion_mean is None else motion_mean[index:index + 1]),
                        motion_std=(None if motion_std is None else motion_std[index:index + 1]),
                    )
                )
            return FlowSampleResult(
                initial_noise=torch.cat(
                    [result.initial_noise for result in per_sample], dim=0
                ),
                raw=torch.cat([result.raw for result in per_sample], dim=0),
                official_pre_cast=torch.cat(
                    [result.official_pre_cast for result in per_sample], dim=0
                ),
                official=torch.cat([result.official for result in per_sample], dim=0),
                sigmas=per_sample[-1].sigmas,
                timesteps=per_sample[-1].timesteps,
                trace=(
                    {
                        key: (
                            torch.stack([result.trace[key] for result in per_sample], dim=1)
                            if per_sample[0].trace[key].ndim == 1
                            else torch.cat([result.trace[key] for result in per_sample], dim=1)
                        )
                        for key in per_sample[0].trace
                    }
                    if trace_enabled and per_sample and per_sample[0].trace is not None
                    else None
                ),
                reconciled=(
                    torch.cat([result.reconciled for result in per_sample], dim=0)
                    if per_sample and per_sample[0].reconciled is not None
                    else None
                ),
                representation_protocol=per_sample[-1].representation_protocol,
                g0=(
                    torch.cat([result.g0 for result in per_sample], dim=0)
                    if per_sample and per_sample[0].g0 is not None
                    else None
                ),
                g1=(
                    torch.cat([result.g1 for result in per_sample], dim=0)
                    if per_sample and per_sample[0].g1 is not None
                    else None
                ),
                guidance_summary=(
                    {
                        "protocol": per_sample[0].guidance_summary.get("protocol"),
                        "samples": [result.guidance_summary for result in per_sample],
                    }
                    if per_sample and per_sample[0].guidance_summary is not None
                    else None
                ),
            )
        device = initial_noise.device
        if prompt_emb.device != device:
            raise ValueError(
                f"prompt_emb must be on {device}, got {prompt_emb.device}"
            )
        if prompt_emb_null.device != device:
            raise ValueError(
                f"prompt_emb_null must be on {device}, got {prompt_emb_null.device}"
            )

        xt = initial_noise.detach().clone()
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            training=False,
            denoising_strength=self.denoising_strength,
        )
        timesteps = self.scheduler.timesteps.to(device)

        if prompt_emb_null.size(1) < prompt_emb.size(1):
            padding = torch.zeros(
                prompt_emb.size(0),
                prompt_emb.size(1) - prompt_emb_null.size(1),
                prompt_emb.size(2),
                device=prompt_emb.device,
                dtype=prompt_emb.dtype,
            )
            prompt_emb_null = torch.cat([prompt_emb_null, padding], dim=1)

        valid_mask_input = torch.cat([valid_mask] * 2, dim=0)
        ref_motion_null = torch.zeros_like(ref_motion)
        ref_motion_input = torch.cat([ref_motion, ref_motion_null], dim=0)
        ref_motion_mask_input = torch.cat([ref_motion_mask] * 2, dim=0)
        attend_to_text_mask_input = None
        if attend_to_text_mask is not None:
            if attend_to_text_mask.shape != (initial_noise.shape[0],):
                raise ValueError(
                    "attend_to_text_mask must have shape [B], "
                    f"got {tuple(attend_to_text_mask.shape)}"
                )
            attend_to_text_mask_input = torch.cat(
                [attend_to_text_mask] * 2, dim=0
            )

        context_input = torch.cat([prompt_emb, prompt_emb_null], dim=0)
        autocast_enabled = device.type == "cuda" and dtype in (
            torch.float16,
            torch.bfloat16,
        )
        autocast_ctx = (
            torch.amp.autocast(dtype=dtype, device_type=device.type)
            if autocast_enabled
            else contextlib.nullcontext()
        )

        iterator = timesteps
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(timesteps, desc="M0 generation")
        trace_records: list[dict[str, torch.Tensor]] = []
        guidance_step_records: list[dict] = []
        for timestep in iterator:
            x_sigma_trace = xt.detach().clone() if trace_enabled else None
            with autocast_ctx:
                latent_model_input = torch.cat([xt] * 2, dim=0)
                timestep_input = timestep.unsqueeze(0)
                velocity = model(
                    x=latent_model_input,
                    timestep=timestep_input,
                    context=context_input,
                    x_mask=valid_mask_input,
                    ref_motion=ref_motion_input,
                    ref_motion_mask=ref_motion_mask_input,
                    use_gradient_checkpointing=False,
                    attend_to_text_mask=attend_to_text_mask_input,
                )
                velocity_cond, velocity_uncond = velocity.chunk(2)
                if condition_on_text:
                    velocity = velocity_uncond + self.cfg_scale * (
                        velocity_cond - velocity_uncond
                    )
                else:
                    velocity = velocity_cond
                sigma = None
                x0_hat = None
                if trace_enabled:
                    timestep_id = torch.argmin(
                        (self.scheduler.timesteps.to(device) - timestep).abs()
                    )
                    sigma = self.scheduler.sigmas[timestep_id].to(device=device)
                    x0_hat = (
                        x_sigma_trace.float()
                        - sigma.float() * velocity.float()
                        ).detach().clone()
                guidance_trace = None
                if guidance_hook is not None:
                    if sigma is None:
                        timestep_id = torch.argmin(
                            (self.scheduler.timesteps.to(device) - timestep).abs()
                        )
                        sigma = self.scheduler.sigmas[timestep_id].to(device=device)
                    guidance_kwargs = {
                        "x_sigma": xt,
                        "velocity": velocity,
                        "sigma": sigma,
                        # The sampler receives a numeric mask from the
                        # dataloader/model path; M1's loss interface is
                        # intentionally strict and consumes a boolean frame
                        # mask.  This conversion is local to the optional M1
                        # hook and does not alter the frozen M0 path.
                        "valid_mask": valid_mask.bool(),
                    }
                    if trace_enabled:
                        guidance_kwargs["return_trace"] = True
                    velocity, guidance_diagnostics = guidance_hook.correct_velocity(
                        **guidance_kwargs
                    )
                    if relative_root_forward_guidance is not None:
                        guidance_step_records.append({
                            key: value for key, value in guidance_diagnostics.items()
                            if key != "trace" and not isinstance(value, torch.Tensor)
                        })
                    if trace_enabled:
                        guidance_trace = guidance_diagnostics.get("trace")
                x_next = self.scheduler.step(velocity, timestep, xt)
                if trace_enabled:
                    trace_records.append(
                        {
                            "sigma": sigma.detach().float().clone(),
                            "timestep": timestep.detach().float().clone(),
                            "x_sigma": x_sigma_trace,
                            "v_cfg": (
                                velocity.detach().float().clone()
                                if guidance_trace is None
                                else guidance_trace.get("v_cfg", guidance_trace["velocity_model"])
                            ),
                            "velocity_model": (
                                velocity.detach().float().clone()
                                if guidance_trace is None
                                else guidance_trace["velocity_model"]
                            ),
                            "x0_hat": (
                                x0_hat
                                if guidance_trace is None
                                else guidance_trace["x0_hat"]
                            ),
                            "x0_guided": (
                                x0_hat
                                if guidance_trace is None
                                else guidance_trace["x0_guided"]
                            ),
                            "x0_reconciled": (
                                x0_hat
                                if guidance_trace is None
                                else guidance_trace["x0_reconciled"]
                            ),
                            "velocity_corrected": velocity.detach().float().clone(),
                            "v_corrected": velocity.detach().float().clone(),
                            "x_next": x_next.detach().float().clone(),
                        }
                    )
                xt = x_next

        raw = xt.clone()
        official = raw.clone()
        for index in range(official.shape[0]):
            official[index] = smooth_motion_rep(
                official[index],
                kernel_size=self.smooth_kernel_size,
                sigma=self.smooth_sigma,
            )
        reconciled = None
        representation_protocol = None
        g0 = None
        g1 = None
        guidance_summary = None
        if absolute_mean_guidance is not None:
            absolute_outputs = absolute_mean_guidance.finalize_outputs(official.float())
            # Preserve the authoritative FP32 boundary for the 1e-4 degree
            # rotation/velocity audit.  ``reconciled`` retains the caller's
            # historical output dtype, while explicit G0/G1 artifacts do not
            # undergo a second lossy cast.
            g0 = absolute_outputs.g0
            g1 = absolute_outputs.g1
            reconciled = g0.to(dtype=dtype)
            representation_protocol = absolute_outputs.protocol
            guidance_summary = {
                **absolute_mean_guidance.protocol_record(),
                "terminal_records": list(absolute_outputs.terminal_records),
            }
        if relative_root_forward_guidance is not None:
            relative_outputs = relative_root_forward_guidance.finalize_outputs(official.float())
            g0 = relative_outputs.g0
            reconciled = g0.to(dtype=dtype)
            representation_protocol = relative_outputs.protocol
            guidance_summary = {
                **relative_root_forward_guidance.protocol_record(),
                "final_projection_audits": list(relative_outputs.projection_audits),
                "whole_body_audits": list(relative_outputs.whole_body_audits),
                "metrics": relative_outputs.metrics,
                "step_records": guidance_step_records,
            }
        if reconciliation_config and bool(reconciliation_config.get("enabled", False)):
            if absolute_mean_guidance is not None or relative_root_forward_guidance is not None:
                raise ValueError(
                    "guided protocol owns the final reconciliation boundary; "
                    "do not also pass reconciliation_config"
                )
            if motion_mean is None or motion_std is None:
                raise ValueError(
                    "representation reconciliation requires motion_mean and motion_std"
                )
            consistency_mode = str(
                reconciliation_config.get(
                    "consistency_mode",
                    reconciliation_config.get("version", reconciliation_config.get("protocol_version", "legacy_v1")),
                )
            )
            if consistency_mode in {"fk_v2", "v2", "vimogen_276d_consistency_v2"}:
                from motion_rep.consistency_v2 import (
                    load_smplx_neutral_22_skeleton,
                    reconcile_motion_tensor_v2,
                )

                skeleton_path = reconciliation_config.get("skeleton_path")
                skeleton = None if skeleton_path is None else load_smplx_neutral_22_skeleton(skeleton_path)
                reconciled_result = reconcile_motion_tensor_v2(
                    official.float(),
                    fusion_window=int(reconciliation_config.get("correction_window", 9)),
                    anchor_weight=float(reconciliation_config.get("anchor_weight", 1.0)),
                    root_rotation_anchor_weight=float(
                        reconciliation_config.get("root_rotation_anchor_weight", reconciliation_config.get("anchor_weight", 1.0))
                    ),
                    valid_mask=valid_mask.bool(),
                    mean=motion_mean.float(),
                    std=motion_std.float(),
                    input_standardized=True,
                    output_standardized=True,
                    output_dtype=dtype,
                    skeleton=skeleton,
                )
            else:
                from motion_rep.reconciliation import ReconciliationConfig, reconcile_motion_tensor

                reconciler_config = ReconciliationConfig(
                    correction_window=int(reconciliation_config.get("correction_window", 9)),
                    anchor_weight=float(reconciliation_config.get("anchor_weight", 1.0)),
                    root_rotation_anchor_weight=float(
                        reconciliation_config.get("root_rotation_anchor_weight", reconciliation_config.get("anchor_weight", 1.0))
                    ),
                )
                reconciled_result = reconcile_motion_tensor(
                    official.float(),
                    config=reconciler_config,
                    component_weights=reconciliation_config.get("component_weights"),
                    valid_mask=valid_mask.bool(),
                    mean=motion_mean.float(),
                    std=motion_std.float(),
                    input_standardized=True,
                    output_standardized=True,
                    output_dtype=dtype,
                )
            reconciled = reconciled_result.motion
            representation_protocol = reconciled_result.protocol
        trace = None
        if trace_enabled and trace_records:
            trace = {
                key: torch.stack([record[key] for record in trace_records], dim=0)
                for key in trace_records[0]
            }
            # The last step has no subsequent model evaluation.  Keep a
            # fixed-shape tensor for convenient saving and expose validity.
            next_x0 = torch.cat(
                [trace["x0_hat"][1:], torch.zeros_like(trace["x0_hat"][:1])],
                dim=0,
            )
            trace["x0_model_next"] = next_x0
            trace["x0_model_next_valid"] = torch.tensor(
                [True] * max(trace["x0_hat"].shape[0] - 1, 0)
                + [False],
                dtype=torch.bool,
            )
            # Public names used by the M1 residual-source audit.  The next
            # model endpoint is the next iteration's pre-guidance x0_hat;
            # the final step has no next model evaluation.
            trace["next_model_x0"] = trace["x0_model_next"]
            trace["next_model_x0_valid"] = trace["x0_model_next_valid"]
        return FlowSampleResult(
            initial_noise=initial_noise.detach().clone(),
            raw=raw,
            official_pre_cast=official,
            official=official.to(dtype=dtype),
            sigmas=self.scheduler.sigmas.detach().clone(),
            timesteps=self.scheduler.timesteps.detach().clone(),
            trace=trace,
            reconciled=reconciled,
            representation_protocol=representation_protocol,
            g0=g0,
            g1=g1,
            guidance_summary=guidance_summary,
        )
