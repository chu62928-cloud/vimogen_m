"""Regression tests for the shared motion-oriented video display frame."""

import torch

from scripts.render_absolute_mean_triptych import (
    estimate_motion_heading,
    fixed_sagittal_side_camera,
)


def _walking_joints(*, direction: float = 1.0) -> torch.Tensor:
    roots = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.4, 1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    roots[:, 1] *= direction
    joints = roots[:, None, :].repeat(1, 22, 1)
    return joints


def test_display_heading_follows_motion_and_mesh_screen_sign() -> None:
    joints = _walking_joints()
    heading = estimate_motion_heading(joints)
    assert torch.allclose(heading, torch.tensor([0.0, 1.0, 0.0]))

    camera_r, camera_t = fixed_sagittal_side_camera(joints, motion_heading=heading)
    camera_points = torch.matmul(joints[:, :1], camera_r) + camera_t[:, None]
    # PyTorch3D's screen convention maps positive camera-x to the left.  The
    # renderer must therefore choose camera-x=-motion-heading so +motion is
    # displayed to the right in both mesh and skeleton outputs.
    screen_x = -camera_points[:, 0, 0]
    assert float(screen_x[-1] - screen_x[0]) > 0.0


def test_camera_follows_negative_motion_and_keeps_motion_to_the_right() -> None:
    joints = _walking_joints(direction=-1.0)
    heading = estimate_motion_heading(joints)
    assert torch.allclose(heading, torch.tensor([0.0, -1.0, 0.0]))
    camera_r, camera_t = fixed_sagittal_side_camera(joints, motion_heading=heading)
    camera_points = torch.matmul(joints[:, :1], camera_r) + camera_t[:, None]
    screen_x = -camera_points[:, 0, 0]
    assert float(screen_x[-1] - screen_x[0]) > 0.0
