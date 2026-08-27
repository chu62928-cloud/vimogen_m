from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("motion_rep.phase1")

from evaluation.representation_consistency import (  # noqa: E402
    bootstrap_cluster_curve,
    compute_sequence_metrics,
    interpolate_curve,
)
from motion_rep.phase1 import MOTION_LAYOUT  # noqa: E402


def _identity_rot6d(frames: int) -> torch.Tensor:
    # ViMoGen flattens the first two matrix columns in row-major order.
    identity = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return identity.repeat(frames, 1)


def _consistent_motion(frames: int = 5) -> torch.Tensor:
    motion = torch.zeros(frames, 276)
    motion[:, MOTION_LAYOUT.body_pose] = _identity_rot6d(frames).repeat(1, 21)
    joints = torch.zeros(frames, 22, 3)
    joints[..., 0] = torch.arange(frames, dtype=torch.float32)[:, None] * 0.02
    joints[..., 1] = torch.arange(22, dtype=torch.float32)[None, :] * 0.01
    motion[:, MOTION_LAYOUT.joints] = joints.reshape(frames, 66)
    motion[:, MOTION_LAYOUT.joints_velocity] = torch.cat(
        (joints[1:] - joints[:-1], joints[-1:] - joints[-2:-1]), dim=0
    ).reshape(frames, 66)
    motion[:, MOTION_LAYOUT.root_rotation] = _identity_rot6d(frames)
    motion[:, MOTION_LAYOUT.root_rotation_velocity] = _identity_rot6d(frames)
    translation = torch.zeros(frames, 3)
    translation[:, 1] = torch.arange(frames) * 0.01
    motion[:, MOTION_LAYOUT.root_translation] = translation
    motion[:, MOTION_LAYOUT.root_translation_velocity] = torch.cat(
        (translation[1:] - translation[:-1], translation[-1:] - translation[-2:-1]), dim=0
    )
    return motion


def test_consistent_stream_has_zero_position_velocity_and_drift_metrics() -> None:
    result = compute_sequence_metrics(_consistent_motion())
    assert result["speed_residual_mean_m_per_frame"] == pytest.approx(0.0, abs=2e-7)
    assert result["trajectory_drift_final_m"] == pytest.approx(0.0, abs=1e-7)
    assert result["trajectory_drift_slope_m_per_frame"] == pytest.approx(0.0, abs=1e-7)
    assert result["root_translation_speed_residual"]["mean"] == pytest.approx(0.0, abs=1e-7)
    assert result["root_rotation_integrated_drift_degrees"]["max"] == pytest.approx(0.0, abs=1e-5)


def test_single_velocity_perturbation_accumulates_after_perturbed_frame() -> None:
    motion = _consistent_motion()
    motion[1, MOTION_LAYOUT.joints_velocity.start] += 0.1
    result = compute_sequence_metrics(motion)
    speed_curve = torch.tensor(result["curves"]["speed_residual_m_per_frame"])
    drift_curve = torch.tensor(result["curves"]["trajectory_drift_m"])
    assert speed_curve[1] > 0.09
    assert torch.all(drift_curve[2:] > 0.09)
    assert drift_curve[0] == pytest.approx(0.0, abs=1e-7)


def test_constant_position_offset_does_not_create_velocity_residual() -> None:
    motion = _consistent_motion()
    motion[:, MOTION_LAYOUT.joints] += 0.5
    result = compute_sequence_metrics(motion)
    assert result["speed_residual_mean_m_per_frame"] == pytest.approx(0.0, abs=2e-7)
    assert result["trajectory_drift_final_m"] == pytest.approx(0.0, abs=1e-7)


def test_variable_length_curve_interpolation_and_cluster_bootstrap() -> None:
    assert interpolate_curve([0.0, 1.0], 5).tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    records = [
        {"sample_id": "a", "curves": {"x": [0.0, 1.0]}},
        {"sample_id": "a", "curves": {"x": [0.0, 1.0]}},
        {"sample_id": "b", "curves": {"x": [0.0, 2.0]}},
    ]
    summary = bootstrap_cluster_curve(records, "x", length=5, repetitions=20, seed=3)
    assert len(summary["median"]) == 5
    assert summary["ci95_low"][0] == pytest.approx(0.0)


def test_last_velocity_row_is_excluded_from_position_metrics() -> None:
    motion = _consistent_motion()
    baseline = compute_sequence_metrics(motion)
    motion[-1, MOTION_LAYOUT.joints_velocity] += 100.0
    changed = compute_sequence_metrics(motion)
    assert changed["speed_residual_mean_m_per_frame"] == pytest.approx(
        baseline["speed_residual_mean_m_per_frame"], abs=2e-7
    )
    assert changed["trajectory_drift_final_m"] == pytest.approx(
        baseline["trajectory_drift_final_m"], abs=2e-7
    )


def test_smplx_fk_roundtrip_and_pelvis_relative_translation_invariance() -> None:
    smplx = pytest.importorskip("smplx")
    model_path = Path(__file__).resolve().parents[1] / "data/body_models/smplx"
    if not model_path.exists():
        pytest.skip("SMPL-X assets are not mounted in this checkout")
    frames = 3
    model = smplx.SMPLX(
        model_path=str(model_path), gender="neutral", use_pca=False,
        num_betas=10, batch_size=1,
    ).eval()
    identity = _identity_rot6d(frames)
    zeros3 = torch.zeros(frames, 3)
    with torch.no_grad():
        output = model(
            global_orient=zeros3,
            body_pose=zeros3.repeat(1, 21),
            left_hand_pose=torch.zeros(frames, 45),
            right_hand_pose=torch.zeros(frames, 45),
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            transl=zeros3,
            betas=torch.zeros(frames, 10),
            expression=torch.zeros(frames, 10),
        )
    motion = _consistent_motion(frames)
    motion[:, MOTION_LAYOUT.body_pose] = identity.repeat(1, 21)
    motion[:, MOTION_LAYOUT.root_rotation] = identity
    motion[:, MOTION_LAYOUT.root_rotation_velocity] = identity
    motion[:, MOTION_LAYOUT.root_translation] = 0.0
    motion[:, MOTION_LAYOUT.root_translation_velocity] = 0.0
    joints = output.joints[:, :22].detach()
    motion[:, MOTION_LAYOUT.joints] = joints.reshape(frames, 66)
    motion[:, MOTION_LAYOUT.joints_velocity] = torch.cat(
        (joints[1:] - joints[:-1], joints[-1:] - joints[-2:-1]), dim=0
    ).reshape(frames, 66)
    result = compute_sequence_metrics(motion, model=model)
    assert result["fk_absolute_mean_m"] == pytest.approx(0.0, abs=1e-5)
    assert result["fk_relative_pelvis_mean_m"] == pytest.approx(0.0, abs=1e-5)

    shifted = motion.clone()
    shifted[:, MOTION_LAYOUT.joints] += 0.5
    shifted_joints = shifted[:, MOTION_LAYOUT.joints].reshape(frames, 22, 3)
    shifted[:, MOTION_LAYOUT.joints_velocity] = torch.cat(
        (shifted_joints[1:] - shifted_joints[:-1], shifted_joints[-1:] - shifted_joints[-2:-1]), dim=0
    ).reshape(frames, 66)
    shifted_result = compute_sequence_metrics(shifted, model=model)
    assert shifted_result["fk_absolute_mean_m"] > 0.4
    assert shifted_result["fk_relative_pelvis_mean_m"] == pytest.approx(0.0, abs=1e-5)
