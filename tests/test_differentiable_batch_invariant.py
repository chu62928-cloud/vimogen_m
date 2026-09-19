from __future__ import annotations

import torch

from sampling.differentiable_flow_sampler import (
    DifferentiableSamplerConfig,
    differentiable_generate,
)


class _Scheduler:
    def set_timesteps(self, steps, *, training, denoising_strength):
        del training, denoising_strength
        self.timesteps = torch.linspace(1.0, 0.1, steps)
        self.sigmas = self.timesteps.clone()

    def step(self, velocity, timestep, state):
        del timestep
        return state - 0.1 * velocity


class _BatchSensitiveModel(torch.nn.Module):
    def forward(self, *, x, timestep, context, x_mask, ref_motion, ref_motion_mask,
                use_gradient_checkpointing, attend_to_text_mask):
        del timestep, context, x_mask, ref_motion, ref_motion_mask
        del use_gradient_checkpointing, attend_to_text_mask
        # Mimic a kernel whose reduction path depends on batch size.
        return x * (1.0 + x.shape[0] * 1.0e-3)


def _run(noise: torch.Tensor, *, batch_invariant: bool):
    batch, frames, _ = noise.shape
    return differentiable_generate(
        model=_BatchSensitiveModel(),
        scheduler=_Scheduler(),
        prompt_emb=torch.zeros(batch, 2, 3),
        prompt_emb_null=torch.zeros(batch, 2, 3),
        initial_noise=noise,
        valid_mask=torch.ones(batch, frames, dtype=torch.bool),
        ref_motion=torch.zeros_like(noise),
        ref_motion_mask=torch.zeros(batch, frames, dtype=torch.bool),
        condition_on_text=False,
        attend_to_text_mask=torch.ones(batch, dtype=torch.bool),
        dtype=torch.float32,
        config=DifferentiableSamplerConfig(
            num_inference_steps=2,
            smooth_kernel_size=1,
            smooth_sigma=1.0,
            use_gradient_checkpointing=False,
        ),
        batch_invariant=batch_invariant,
    )


def test_batch_invariant_path_matches_singletons_and_preserves_gradients() -> None:
    noise = torch.randn(2, 5, 276, requires_grad=True)
    batch = _run(noise, batch_invariant=True)
    singles = torch.cat(
        [_run(noise[index : index + 1], batch_invariant=False).official_pre_cast
         for index in range(2)]
    )
    torch.testing.assert_close(batch.official_pre_cast, singles)
    batch.official_pre_cast.square().mean().backward()
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()


def test_joint_batch_path_can_differ_from_singletons() -> None:
    noise = torch.randn(2, 5, 276)
    joint = _run(noise, batch_invariant=False).official_pre_cast
    invariant = _run(noise, batch_invariant=True).official_pre_cast
    assert not torch.equal(joint, invariant)
