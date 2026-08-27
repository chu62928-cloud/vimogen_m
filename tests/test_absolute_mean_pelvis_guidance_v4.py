import torch

from motion_rep.anatomical_pelvis import PelvisCalibration
from sampling.absolute_mean_pelvis_guidance_v4 import (
    AbsoluteMeanPelvisConfigV4,
    AbsoluteMeanPelvisGuidanceV4,
)


def _calibration():
    names = ("LASI", "RASI", "LPSI", "RPSI")
    return PelvisCalibration(
        template_sha256="2" * 64,
        model_path="synthetic",
        marker_vertex_groups={name: (i,) for i, name in enumerate(names)},
        marker_local_points={"LASI": (-.15, .35, 0), "RASI": (.15, .35, 0), "LPSI": (-.15, -.35, 0), "RPSI": (.15, -.35, 0)},
    )


def _motion(frames=4):
    motion = torch.zeros(1, frames, 276)
    identity = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
    motion[:, :, :126] = identity.repeat(21).view(1, 1, 126)
    motion[:, :, 258:264] = identity
    return motion


def test_v4_requires_calibration_and_keeps_g1_non_rigid():
    baseline = _motion()
    mean = torch.zeros(276)
    std = torch.ones(276)
    mask = torch.ones(1, baseline.shape[1], dtype=torch.bool)
    cfg = AbsoluteMeanPelvisConfigV4(enabled=False)
    guidance = AbsoluteMeanPelvisGuidanceV4(baseline_motion_norm=baseline, valid_mask=mask, mean=mean, std=std, target_mean_deg=5, config=cfg, calibration=_calibration())
    output = guidance.finalize_outputs(baseline)
    assert torch.equal(output.g0, output.g1)
    assert all(record["terminal_skipped"] for record in output.terminal_records)


def test_v4_active_mask_is_explicit():
    record = AbsoluteMeanPelvisConfigV4().from_mapping({"enabled": True})
    assert record.anti_cheat_weight == 1.0
    assert record.soft_limit_deg == 2.0 and record.p95_limit_deg == 3.0
