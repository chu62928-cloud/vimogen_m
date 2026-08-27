import math

import torch

from motion_rep.anatomical_pelvis import (
    PelvisCalibration,
    anatomical_pelvis_geometry,
    anti_cheat_metrics,
    anti_cheat_penalty,
    apply_anatomical_pelvis_delta,
    local_dominance_penalty,
    trunk_and_thigh_angles,
)


def calibration() -> PelvisCalibration:
    # Synthetic z-up neutral pelvis: posterior is -y, anterior is +y.
    points = {
        "LASI": (-0.15, 0.35, 0.0),
        "RASI": (0.15, 0.35, 0.0),
        "LPSI": (-0.15, -0.35, 0.0),
        "RPSI": (0.15, -0.35, 0.0),
    }
    return PelvisCalibration(
        template_sha256="0" * 64,
        model_path="synthetic",
        marker_vertex_groups={name: (index,) for index, name in enumerate(points)},
        marker_local_points=points,
    )


def rot_x(degrees: float) -> torch.Tensor:
    angle = torch.tensor(math.radians(degrees), dtype=torch.float64)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)


def test_anatomical_sign_and_positive_delta():
    cal = calibration()
    neutral = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(anatomical_pelvis_geometry(neutral, cal).angle_degrees, torch.tensor(0.0, dtype=torch.float64), atol=1e-5)
    # A physical anterior-down tilt is R_x(-angle) under the +y anterior convention.
    tilted = rot_x(-10.0)
    measured = anatomical_pelvis_geometry(tilted, cal).angle_degrees
    assert torch.allclose(measured, torch.tensor(10.0, dtype=torch.float64), atol=1e-4)
    corrected = apply_anatomical_pelvis_delta(neutral, 7.0, cal)
    assert torch.allclose(anatomical_pelvis_geometry(corrected, cal).angle_degrees, torch.tensor(7.0, dtype=torch.float64), atol=1e-4)


def test_yaw_invariance_and_degenerate_heading():
    cal = calibration()
    yaw = torch.tensor(math.radians(73.0), dtype=torch.float64)
    c, s = torch.cos(yaw), torch.sin(yaw)
    yaw_rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    pose = yaw_rot @ rot_x(-12.0)
    result = anatomical_pelvis_geometry(pose, cal)
    assert torch.allclose(result.angle_degrees, torch.tensor(12.0, dtype=torch.float64), atol=1e-4)
    # A marker line parallel to up is invalid but must remain finite.
    vertical = PelvisCalibration(
        template_sha256="1" * 64,
        model_path="synthetic",
        marker_vertex_groups={name: (index,) for index, name in enumerate(("LASI", "RASI", "LPSI", "RPSI"))},
        marker_local_points={"LASI": (0, 0, 1), "RASI": (0, 0, 1), "LPSI": (0, 0, 0), "RPSI": (0, 0, 0)},
    )
    degenerate = anatomical_pelvis_geometry(torch.eye(3, dtype=torch.float64), vertical)
    assert not bool(degenerate.valid)
    assert torch.isfinite(degenerate.angle_degrees)


def test_trunk_thigh_and_anti_cheat_metrics():
    cal = calibration()
    pelvis = anatomical_pelvis_geometry(torch.eye(3, dtype=torch.float64), cal)
    joints = torch.zeros(22, 3, dtype=torch.float64)
    joints[3] = torch.tensor((0.0, 0.0, 1.0), dtype=joints.dtype)  # spine1
    joints[12] = torch.tensor((0.0, 0.2, 2.0), dtype=joints.dtype)  # neck forward by 0.2
    joints[1] = torch.tensor((-0.2, 0.0, 0.0), dtype=joints.dtype)
    joints[2] = torch.tensor((0.2, 0.0, 0.0), dtype=joints.dtype)
    joints[4] = torch.tensor((-0.2, 0.0, -1.0), dtype=joints.dtype)
    joints[5] = torch.tensor((0.2, 0.0, -1.0), dtype=joints.dtype)
    values = trunk_and_thigh_angles(joints, pelvis)
    assert values["trunk_deg"] > 0
    penalty = anti_cheat_penalty(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor([True, True]))
    assert torch.allclose(penalty, torch.tensor(0.0))
    penalty_high = anti_cheat_penalty(torch.tensor([4.0]), torch.tensor([2.0]), torch.tensor([2.0]), torch.tensor([True]))
    assert torch.allclose(penalty_high, torch.tensor(4.0 / 3.0))
    metrics = anti_cheat_metrics(
        torch.zeros(4), torch.tensor([0.0, 4.0, 6.0, 8.0]),
        torch.zeros(4), torch.tensor([0.0, 1.0, 2.0, 2.0]),
        torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.zeros(4),
        torch.ones(4, dtype=torch.bool),
    )
    assert metrics["local_change_same_sign"]
    assert float(metrics["local_change_share"]) > 0.5
    assert float(metrics["trunk_abs_p95_deg"]) <= 2.0
    assert local_dominance_penalty(torch.tensor([4.0]), torch.tensor([0.0]), torch.tensor([True])) == 0
    assert local_dominance_penalty(torch.tensor([4.0]), torch.tensor([4.0]), torch.tensor([True])) > 0
