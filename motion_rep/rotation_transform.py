"""Rotation conversions for ViMoGen's interleaved Rot6D layout.

The stored representation is the first two columns of a rotation matrix,
flattened row-major (the six values are interleaved by matrix row).  This is
not the same memory layout as every version of a third-party 6D helper, so the
layout is kept explicit here while PyTorch3D supplies matrix/axis-angle and
quaternion conversions.
"""

import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion as p3d_axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_to_axis_angle as p3d_quaternion_to_axis_angle,
)


def mat3x3_to_rot6d(R):
    """Convert matrices to ViMoGen's interleaved first-two-column Rot6D."""
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rot6d_to_mat3x3(rot6d):
    """Decode ViMoGen's interleaved Rot6D with Gram-Schmidt."""
    batch_shape = rot6d.shape[:-1]
    x = rot6d.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1).reshape(*batch_shape, 3, 3)


def axis_angle_to_rot6d(angle_axis):
    """Convert axis-angle rotations to ViMoGen Rot6D."""
    return mat3x3_to_rot6d(axis_angle_to_matrix(angle_axis))


def rot6d_to_axis_angle(rot6d):
    """Convert ViMoGen Rot6D to axis-angle rotations."""
    axis_angle = matrix_to_axis_angle(rot6d_to_mat3x3(rot6d))
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def axis_angle_to_mat3x3(angle_axis):
    return axis_angle_to_matrix(angle_axis)


def mat3x3_to_axis_angle(rot_mat):
    axis_angle = matrix_to_axis_angle(rot_mat)
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def quaternion_to_axis_angle(quaternion):
    return p3d_quaternion_to_axis_angle(quaternion)


def axis_angle_to_quaternion(angle_axis):
    return p3d_axis_angle_to_quaternion(angle_axis)


def quaternion_to_rot6d(quaternion):
    return axis_angle_to_rot6d(quaternion_to_axis_angle(quaternion))


def rot6d_to_quaternion(rot6d):
    return axis_angle_to_quaternion(rot6d_to_axis_angle(rot6d))