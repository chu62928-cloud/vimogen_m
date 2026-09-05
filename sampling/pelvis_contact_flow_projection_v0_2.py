"""Versioned entry point for the temporal-contact projection protocol.

The numerical implementation remains shared with v0.1 so that the original
protocol can be regression-tested unchanged.  v0.2 is selected by its
configuration protocol and enables the velocity residuals and boundary halo.
"""

from .pelvis_contact_flow_projection_v0_1 import (  # noqa: F401
    ACTIVE_BODY_INDICES,
    ACTIVE_JOINT_NAMES,
    EUCLIDEAN_METRIC,
    KINEMATIC_TEMPORAL_METRIC,
    METHOD_NAME,
    PENETRATION_METHOD,
    PelvisContactFlowProjector,
    ProjectionFinalOutputs,
    ProjectionResult,
    ProjectorConfig,
    TEMPORAL_CONTACT_PROTOCOL,
    autograd_jacobian,
    build_projection_metric,
    finite_difference_jacobian,
    predict_clean_endpoint,
    project_increment_norms,
    recompose_velocity,
    so3_exp,
    so3_log,
    solve_local_projection,
    temporal_contact_residual,
    write_strict_json,
)

PROTOCOL_NAME = TEMPORAL_CONTACT_PROTOCOL

__all__ = [
    "ACTIVE_BODY_INDICES",
    "ACTIVE_JOINT_NAMES",
    "EUCLIDEAN_METRIC",
    "KINEMATIC_TEMPORAL_METRIC",
    "METHOD_NAME",
    "PENETRATION_METHOD",
    "PROTOCOL_NAME",
    "PelvisContactFlowProjector",
    "ProjectionFinalOutputs",
    "ProjectionResult",
    "ProjectorConfig",
    "autograd_jacobian",
    "build_projection_metric",
    "finite_difference_jacobian",
    "predict_clean_endpoint",
    "project_increment_norms",
    "recompose_velocity",
    "so3_exp",
    "so3_log",
    "solve_local_projection",
    "temporal_contact_residual",
    "write_strict_json",
]
