from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation.control_metrics import dose_response_metrics, evaluate_control_metrics
from evaluation.content_metrics import evaluate_content_metrics
from evaluation.physical_metrics import evaluate_physical_metrics
from geometry.contacts import freeze_contact_evidence
from geometry.ground import estimate_ground_height
from geometry.pelvis_angle import pelvis_angle_curve_deg, target_angle_curve_deg
from guidance.base import (
    ConstraintPack,
    GuidanceRequest,
    SharedEvidence,
    slice_request,
    write_run_record,
)
from guidance.m7_paht_edit import M7PAHTGeometricEdit
from guidance.m1_loss_guidance import M1Config, M1LossGuidanceHook
from guidance.m2_dflow_source import M2DFlowSourceOptimization
from guidance.m2_dflow_source_v2 import M2DFlowSourceOptimizationV2
from guidance.m3_projflow_local import M3Config, M3ProjFlowLocalHook
from guidance.m3_projflow_local_v2 import M3ProjFlowLocalHookV2
from guidance.m4_pcfm import M4Config, M4PCFMHook
from guidance.m4_pcfm_v2 import M4PCFMHookV2
from guidance.m5_ldf import M5Config, M5LagrangianDualFlowHook
from guidance.m6_lyaguide import M6Config, M6LyaGuideHook
from guidance.sampling_common import predicted_clean
from motion_rep.phase1 import MOTION_LAYOUT, encode_rot6d
from motion_rep.phase1 import decode_rot6d_safe
from motion_rep.sagittal_pelvis_angle import apply_person_right_axis_rotation
from scripts.freeze_m1_m7_protocol import validate_configs
from scripts.freeze_m1_m7_method_revisions import freeze_revisions, validate_revisions


def _motion(frames: int = 5) -> torch.Tensor:
    motion = torch.zeros((1, frames, MOTION_LAYOUT.total_dim), dtype=torch.float32)
    identity = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, frames, 1, 1)
    # SMPL-X local +z is the forward axis while canonical world +z is up.
    # Use the normal zero-pitch root frame (local +z -> world +y), not the
    # identity matrix whose local forward is vertical and angle-singular.
    horizontal_root = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    ).reshape(1, 1, 3, 3).repeat(1, frames, 1, 1)
    motion[..., MOTION_LAYOUT.body_pose] = encode_rot6d(
        identity.unsqueeze(-3).repeat(1, 1, 21, 1, 1)
    ).reshape(1, frames, 126)
    motion[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(horizontal_root)
    motion[..., MOTION_LAYOUT.root_rotation_velocity] = encode_rot6d(identity)
    return motion


def _request(dose: float = 5.0) -> GuidanceRequest:
    baseline = _motion()
    mask = torch.ones(baseline.shape[:2], dtype=torch.bool)
    target = target_angle_curve_deg(baseline, dose)
    evidence = SharedEvidence(mask, target)
    return GuidanceRequest(
        prompt_id="fixture",
        seed=7,
        target_dose_deg=dose,
        constraint_pack=ConstraintPack.C0,
        base_noise=torch.zeros_like(baseline),
        baseline_motion=baseline,
        shared_evidence=evidence,
    )


def test_request_rejects_non_prefix_masks() -> None:
    baseline = _motion()
    mask = torch.tensor([[True, False, True, False, False]])
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        GuidanceRequest(
            prompt_id="fixture",
            seed=0,
            target_dose_deg=2.0,
            constraint_pack="C0",
            base_noise=torch.zeros_like(baseline),
            baseline_motion=baseline,
            shared_evidence=SharedEvidence(
                mask, target_angle_curve_deg(baseline, 2.0)
            ),
        )


def test_v2_method_revisions_freeze_without_overwrite(tmp_path: Path) -> None:
    configs = validate_revisions()
    assert configs["m2_v2.yaml"]["selection_scope"] == "independent_per_sample"
    assert configs["m4_v2.yaml"]["shooting_sigmas"] == []
    output = tmp_path / "protocol_v2_revisions"
    manifest = freeze_revisions(output, code_commit="abc123")
    assert manifest["status"] == "METHOD_REVISIONS_FROZEN_FOR_S1"
    with pytest.raises(FileExistsError):
        freeze_revisions(output, code_commit="abc123")


@pytest.mark.parametrize("dose", [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0])
def test_m7_c0_hits_signed_dose_and_keeps_direct_non_root_channels(dose: float) -> None:
    request = _request(dose)
    result = M7PAHTGeometricEdit().run(None, request)
    assert result.status == "COMPLETED"
    assert result.diagnostics_dict()["angle_residual_p95_deg"] < 1.0e-4
    torch.testing.assert_close(
        result.motion[..., MOTION_LAYOUT.body_pose],
        request.baseline_motion[..., MOTION_LAYOUT.body_pose],
    )
    torch.testing.assert_close(
        result.motion[..., MOTION_LAYOUT.root_translation],
        request.baseline_motion[..., MOTION_LAYOUT.root_translation],
    )
    actual = pelvis_angle_curve_deg(result.motion)
    torch.testing.assert_close(
        actual,
        request.shared_evidence.target_angle_curve_deg,
        atol=1.0e-4,
        rtol=0,
    )


def test_sequence_first_control_metrics_and_dose_response() -> None:
    request = _request(5.0)
    result = M7PAHTGeometricEdit().run(None, request)
    metrics = evaluate_control_metrics(
        result.motion,
        request.baseline_motion,
        request.shared_evidence.target_angle_curve_deg,
        request.shared_evidence.valid_mask,
        target_dose_deg=5.0,
    )
    assert metrics["summary"]["sequence_angle_pass_rate"] == 1.0
    response = dose_response_metrics(
        [
            {"target_dose_deg": -5.0, "actual_mean_dose_deg": -5.0},
            {"target_dose_deg": 5.0, "actual_mean_dose_deg": 5.0},
        ]
    )
    assert response["dose_response_slope"] == pytest.approx(1.0)
    assert response["sign_symmetry_gap_deg"] == pytest.approx(0.0)


def test_run_record_is_strict_and_complete(tmp_path) -> None:
    path = tmp_path / "run_record.json"
    record = {
        "run_id": "r1",
        "code_commit": "abc",
        "vimogen_checkpoint_hash": "def",
        "evaluator_version": "v1",
        "method_name": "M7",
        "method_config_hash": "cfg",
        "prompt_id": "fixture",
        "seed": 0,
        "target_dose_deg": 2.0,
        "constraint_pack": ConstraintPack.C0,
        "baseline_motion_id": "m0",
        "contact_evidence_version": "contact-v1",
        "ground_version": "ground-v1",
        "status": "COMPLETED",
        "failure_reason": "",
        "all_metrics": {},
        "all_diagnostics": {},
    }
    write_run_record(path, record)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["constraint_pack"] == "C0"
    with pytest.raises(ValueError, match="strict JSON"):
        write_run_record(path, {**record, "all_metrics": {"bad": float("nan")}})


def test_contact_and_ground_evidence_is_frozen_from_m0() -> None:
    frames = 5
    heel = torch.zeros((1, frames, 3))
    toe = torch.zeros((1, frames, 3))
    heel[..., 1] = torch.tensor([0.0, 0.001, 0.002, 0.003, 0.004])
    toe[..., 1] = heel[..., 1]
    markers = {
        "left": {"heel": heel.clone(), "toe": toe.clone()},
        "right": {"heel": heel.clone(), "toe": toe.clone()},
    }
    mask = torch.ones((1, frames), dtype=torch.bool)
    stacked = torch.stack(
        [markers[side][name] for side in ("left", "right") for name in ("heel", "toe")],
        dim=2,
    )
    ground = estimate_ground_height(stacked, mask)
    evidence = freeze_contact_evidence(markers, mask, ground)
    assert not evidence.contact["left"][0, 0]
    assert evidence.contact["left"][0, 1:].all()
    assert evidence.flat_contact["right"][0, 1:].all()
    assert evidence.continuous_pair["left"].shape == (1, frames - 1)


def test_physical_and_content_metrics_are_zero_for_m0_pair() -> None:
    motion = _motion()
    mask = torch.ones(motion.shape[:2], dtype=torch.bool)
    positions = torch.zeros((1, motion.shape[1], 3))
    markers = {
        "left": {"heel": positions.clone(), "toe": positions.clone()},
        "right": {"heel": positions.clone(), "toe": positions.clone()},
    }
    contacts = {side: mask.clone() for side in ("left", "right")}
    physical = evaluate_physical_metrics(
        markers, markers, mask, contacts, torch.zeros(1)
    )
    row = physical["per_sequence"][0]
    assert row["penetration_mm"]["p95"] == pytest.approx(0.0)
    assert row["contact_tangential_speed_mm_per_frame"]["p95"] == pytest.approx(0.0)
    content = evaluate_content_metrics(motion, motion, mask)["per_sequence"][0]
    assert content["mpjpe_vs_m0_mm"] == pytest.approx(0.0)
    assert content["root_translation_deviation_p95_mm"] == pytest.approx(0.0)


def test_m1_state_guidance_reduces_one_step_angle_error() -> None:
    request = _request(2.0)
    mean = torch.zeros(MOTION_LAYOUT.total_dim)
    std = torch.ones(MOTION_LAYOUT.total_dim)
    hook = M1LossGuidanceHook(
        request,
        mean=mean,
        std=std,
        config=M1Config(guidance_scale=0.01, gradient_clip_rms=1.0),
    )
    state = request.baseline_motion.clone()
    velocity = torch.zeros_like(state)
    corrected, record = hook.correct_velocity(
        x_sigma=state,
        velocity=velocity,
        sigma=0.5,
        sigma_next=0.4,
        valid_mask=request.shared_evidence.valid_mask,
    )
    assert record["active"]
    next_state = state + (0.4 - 0.5) * corrected
    before = (pelvis_angle_curve_deg(state) - request.shared_evidence.target_angle_curve_deg).abs().mean()
    after = (pelvis_angle_curve_deg(next_state) - request.shared_evidence.target_angle_curve_deg).abs().mean()
    assert after < before


def test_guidance_hooks_slice_frozen_evidence_for_batch_invariant_sampling() -> None:
    request = _request(2.0)
    request = GuidanceRequest(
        prompt_id=request.prompt_id,
        seed=request.seed,
        target_dose_deg=request.target_dose_deg,
        constraint_pack=request.constraint_pack,
        base_noise=request.base_noise.repeat(2, 1, 1),
        baseline_motion=request.baseline_motion.repeat(2, 1, 1),
        shared_evidence=SharedEvidence(
            request.shared_evidence.valid_mask.repeat(2, 1),
            request.shared_evidence.target_angle_curve_deg.repeat(2, 1),
        ),
    )
    hook = M1LossGuidanceHook(
        request,
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
    ).slice(1)
    assert hook.request.baseline_motion.shape[0] == 1
    assert hook.request.shared_evidence.target_angle_curve_deg.shape[0] == 1


class _DifferentiableRollout:
    nfe_per_rollout = 50

    def rollout(self, source_noise, *, request, differentiable):
        assert differentiable
        motion = request.baseline_motion.clone()
        root = decode_rot6d_safe(motion[..., MOTION_LAYOUT.root_rotation])
        motion[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(
            apply_person_right_axis_rotation(root, source_noise[..., 0] * 10.0)
        )
        return motion


def test_m2_optimizes_only_source_noise_against_frozen_target() -> None:
    request = _request(2.0)
    result = M2DFlowSourceOptimization().run(
        _DifferentiableRollout(), request,
        {"learning_rate": 0.05, "iterations": 5, "source_regularization": 0.0},
    )
    diagnostics = result.diagnostics_dict()
    assert result.status == "COMPLETED"
    assert diagnostics["full_rollout_count"] >= 1
    assert diagnostics["target_redefined_during_optimization"] is False
    torch.testing.assert_close(
        request.shared_evidence.target_angle_curve_deg,
        target_angle_curve_deg(request.baseline_motion, 2.0),
    )


def test_m2_v2_is_single_batch_consistent_and_tracks_per_sample_best() -> None:
    one = _request(2.0)
    batch = GuidanceRequest(
        prompt_id=one.prompt_id,
        seed=one.seed,
        target_dose_deg=one.target_dose_deg,
        constraint_pack=one.constraint_pack,
        base_noise=one.base_noise.repeat(2, 1, 1),
        baseline_motion=one.baseline_motion.repeat(2, 1, 1),
        shared_evidence=SharedEvidence(
            one.shared_evidence.valid_mask.repeat(2, 1),
            one.shared_evidence.target_angle_curve_deg.repeat(2, 1),
        ),
    )
    config = {
        "learning_rate": 0.05,
        "iterations": 5,
        "source_regularization": 0.0,
        "content_weight": 0.0,
        "root_weight": 0.0,
        "source_trust_radius": 10.0,
        "step_trust_radius": 10.0,
    }
    method = M2DFlowSourceOptimizationV2()
    batched = method.run(_DifferentiableRollout(), batch, config)
    singles = [
        method.run(_DifferentiableRollout(), slice_request(batch, index), config)
        for index in range(2)
    ]

    torch.testing.assert_close(
        batched.motion, torch.cat([item.motion for item in singles], dim=0)
    )
    diagnostics = batched.diagnostics_dict()
    assert len(diagnostics["per_sample_best_iteration"]) == 2
    assert diagnostics["optimization_scope"] == "independent_per_sample"


def test_m2_v2_zero_dose_is_an_exact_no_rollout_bypass() -> None:
    request = _request(0.0)

    class _ForbiddenRollout:
        def rollout(self, *args, **kwargs):
            raise AssertionError("zero-dose bypass must not call rollout")

    result = M2DFlowSourceOptimizationV2().run(_ForbiddenRollout(), request)

    assert result.status == "COMPLETED_ZERO_DOSE_BYPASS"
    assert torch.equal(result.motion, request.baseline_motion)
    assert result.diagnostics_dict()["full_rollout_count"] == 0


def test_m3_projects_predicted_endpoint_towards_target() -> None:
    request = _request(2.0)
    hook = M3ProjFlowLocalHook(
        request,
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
        config=M3Config(max_step_deg=2.0),
    )


def test_m3_v2_limits_projection_count_and_bypasses_zero_dose() -> None:
    request = _request(2.0)
    hook = M3ProjFlowLocalHookV2(
        request,
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
        config={
            "sigma_min": 0.0,
            "sigma_max": 1.0,
            "projection_stride": 1,
            "max_projections": 1,
            "max_endpoint_delta_rms": 1.0,
        },
    )
    state = request.baseline_motion.clone()
    velocity = torch.zeros_like(state)
    _, first = hook.correct_velocity(
        x_sigma=state, velocity=velocity, sigma=0.5,
        valid_mask=request.shared_evidence.valid_mask,
    )
    second_velocity, second = hook.correct_velocity(
        x_sigma=state, velocity=velocity, sigma=0.4,
        valid_mask=request.shared_evidence.valid_mask,
    )
    assert first["active"] is True
    assert second["active"] is False
    assert second["reason"] == "PROJECTION_BUDGET_EXHAUSTED"
    torch.testing.assert_close(second_velocity, velocity)

    zero = _request(0.0)
    zero_hook = M3ProjFlowLocalHookV2(
        zero,
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
    )
    zero_velocity, record = zero_hook.correct_velocity(
        x_sigma=zero.baseline_motion,
        velocity=torch.zeros_like(zero.baseline_motion),
        sigma=0.3,
        valid_mask=zero.shared_evidence.valid_mask,
    )
    assert record["reason"] == "ZERO_DOSE_STRICT_BYPASS"
    assert torch.equal(zero_velocity, torch.zeros_like(zero_velocity))
    state = request.baseline_motion.clone()
    corrected, record = hook.correct_velocity(
        x_sigma=state,
        velocity=torch.zeros_like(state),
        sigma=0.5,
        valid_mask=request.shared_evidence.valid_mask,
    )
    projected = predicted_clean(state, corrected, 0.5)
    assert record["active"]
    torch.testing.assert_close(
        pelvis_angle_curve_deg(projected),
        request.shared_evidence.target_angle_curve_deg,
        atol=1.0e-3,
        rtol=0,
    )


class _ShootingRuntime:
    def forward_shoot(self, *, x_sigma, sigma, request):
        del sigma, request
        return x_sigma


def test_m4_forward_shoots_and_terminal_gn_hits_c0() -> None:
    request = _request(2.0)
    hook = M4PCFMHook(
        request, runtime=_ShootingRuntime(),
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
        config=M4Config(shooting_sigmas=(0.5,), trust_radius_deg=2.0),
    )
    corrected, record = hook.correct_velocity(
        x_sigma=request.baseline_motion.clone(),
        velocity=torch.zeros_like(request.baseline_motion),
        sigma=0.5,
        valid_mask=request.shared_evidence.valid_mask,
    )
    assert record["active"] and record["full_rollout_count"] == 1
    projected = predicted_clean(request.baseline_motion, corrected, 0.5)
    torch.testing.assert_close(
        pelvis_angle_curve_deg(projected),
        request.shared_evidence.target_angle_curve_deg,
        atol=1.0e-3, rtol=0,
    )


def test_m4_v2_terminal_only_skips_shooting_and_zero_dose_is_exact() -> None:
    request = _request(2.0)

    class _NoShootRuntime:
        def forward_shoot(self, **kwargs):
            raise AssertionError("terminal-only M4-v2 must not forward shoot")

    hook = M4PCFMHookV2(
        request,
        runtime=_NoShootRuntime(),
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
    )
    velocity = torch.zeros_like(request.baseline_motion)
    unchanged, record = hook.correct_velocity(
        x_sigma=request.baseline_motion,
        velocity=velocity,
        sigma=0.5,
        valid_mask=request.shared_evidence.valid_mask,
    )
    assert record["active"] is False
    assert torch.equal(unchanged, velocity)
    terminal, iterations = hook.finalize_output(
        request.baseline_motion, request.shared_evidence.valid_mask
    )
    assert iterations
    torch.testing.assert_close(
        pelvis_angle_curve_deg(terminal),
        request.shared_evidence.target_angle_curve_deg,
        atol=1.0e-3,
        rtol=0,
    )

    zero = _request(0.0)
    zero_hook = M4PCFMHookV2(
        zero,
        runtime=_NoShootRuntime(),
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
    )
    zero_output, zero_iterations = zero_hook.finalize_output(
        zero.baseline_motion, zero.shared_evidence.valid_mask
    )
    assert zero_iterations == []
    assert torch.equal(zero_output, zero.baseline_motion)


def test_m4_batched_statistics_broadcast() -> None:
    single = _request(2.0)
    baseline = single.baseline_motion.repeat(2, 1, 1)
    valid = single.shared_evidence.valid_mask.repeat(2, 1)
    target = single.shared_evidence.target_angle_curve_deg.repeat(2, 1)
    request = GuidanceRequest(
        prompt_id="fixture_batch",
        seed=7,
        target_dose_deg=2.0,
        constraint_pack=ConstraintPack.C0,
        base_noise=torch.zeros_like(baseline),
        baseline_motion=baseline,
        shared_evidence=SharedEvidence(valid, target),
    )
    hook = M4PCFMHook(
        request,
        runtime=_ShootingRuntime(),
        mean=torch.zeros((2, MOTION_LAYOUT.total_dim)),
        std=torch.ones((2, MOTION_LAYOUT.total_dim)),
        config=M4Config(shooting_sigmas=(0.5,), trust_radius_deg=2.0),
    )
    corrected, record = hook.correct_velocity(
        x_sigma=baseline.clone(),
        velocity=torch.zeros_like(baseline),
        sigma=0.5,
        valid_mask=valid,
    )
    assert record["active"] and corrected.shape == baseline.shape


def test_m5_updates_dual_state_without_projection_or_pseudoinverse() -> None:
    request = _request(2.0)
    hook = M5LagrangianDualFlowHook(
        request,
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
        config=M5Config(primal_gain=0.01, penalty=0.1, dual_gain=1.0),
    )
    corrected, record = hook.correct_velocity(
        x_sigma=request.baseline_motion.clone(),
        velocity=torch.zeros_like(request.baseline_motion),
        sigma=0.5, sigma_next=0.4,
        valid_mask=request.shared_evidence.valid_mask,
    )
    assert record["active"]
    assert record["dual_norm"] > 0
    assert record["explicit_projection_used"] is False
    assert record["pseudoinverse_used"] is False
    assert torch.isfinite(corrected).all()


def test_m6_pseudo_projection_enforces_lyapunov_condition() -> None:
    request = _request(2.0)
    hook = M6LyaGuideHook(
        request,
        mean=torch.zeros(MOTION_LAYOUT.total_dim),
        std=torch.ones(MOTION_LAYOUT.total_dim),
        config=M6Config(candidate_scale=0.01, delta=0.1),
    )
    corrected, record = hook.correct_velocity(
        x_sigma=request.baseline_motion.clone(),
        velocity=torch.zeros_like(request.baseline_motion),
        sigma=0.5,
        valid_mask=request.shared_evidence.valid_mask,
    )
    assert record["active"]
    assert record["lyapunov_condition_after"] <= 1.0e-4
    assert torch.isfinite(corrected).all()


def test_all_algorithm_definitions_are_frozen_for_c0() -> None:
    configs = validate_configs()
    assert configs["common"]["doses_deg"] == [-10, -5, -2, 0, 2, 5, 10]
    assert tuple(configs["methods"]) == tuple(f"M{i}" for i in range(1, 8))
    assert all(
        config["constraint_pack"] == "C0"
        for config in configs["methods"].values()
    )
