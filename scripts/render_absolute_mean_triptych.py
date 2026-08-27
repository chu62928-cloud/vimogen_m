#!/usr/bin/env python3
"""Render the frozen M0/G0/G1 1920x1080 audit-video layout."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from smplx import SMPLX


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.motion_checker import (  # noqa: E402
    _default_smpl_model_path,
    estimate_focal_length,
    render_and_save,
)
from motion_rep.retarget_motion import motion_rep_to_SMPL  # noqa: E402
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from motion_rep.sagittal_pelvis_angle import (  # noqa: E402
    pelvis_sagittal_tilt_degrees,
    person_forward_horizontal_axis,
)
from motion_rep.anatomical_pelvis import (  # noqa: E402
    PelvisCalibration,
    anatomical_pelvis_geometry,
    trunk_and_thigh_angles,
)


WIDTH, HEIGHT, FPS = 1920, 1080, 20
PANEL_WIDTH, PANEL_HEIGHT, PLOT_HEIGHT = 640, 720, 360
COLORS = {"M0": (60, 60, 60), "G0": (220, 110, 20), "G1": (30, 150, 30)}
SMPLX_22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def estimate_motion_heading(
    joints: torch.Tensor,
    *,
    fallback_heading: torch.Tensor | None = None,
    min_displacement: float = 1e-4,
) -> torch.Tensor:
    """Estimate one horizontal display heading from the M0 root trajectory.

    The heading is a display convention, not a replacement for the
    anatomical angle definition.  Using one robust sequence-level direction
    avoids mirroring mesh and skeleton videos or changing the camera on every
    frame.  A nearly stationary sequence has no observable travel direction;
    in that case callers may provide the mean anatomical heading as a
    deterministic fallback.
    """

    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[1] < 1:
        raise ValueError(f"joints must have shape [T,J,3], got {tuple(joints.shape)}")
    if joints.shape[0] < 1:
        raise ValueError("joints must contain at least one frame")
    displacement = joints[-1, 0, :2] - joints[0, 0, :2]
    norm = torch.linalg.vector_norm(displacement)
    if float(norm) >= float(min_displacement):
        return torch.nn.functional.pad(displacement / norm, (0, 1))
    if fallback_heading is None:
        fallback_heading = torch.tensor(
            [0.0, 1.0, 0.0], dtype=joints.dtype, device=joints.device
        )
    fallback_heading = fallback_heading.to(device=joints.device, dtype=joints.dtype)
    if fallback_heading.shape != (3,):
        raise ValueError("fallback_heading must have shape [3]")
    fallback_heading = fallback_heading.clone()
    fallback_heading[2] = 0.0
    fallback_norm = torch.linalg.vector_norm(fallback_heading)
    if float(fallback_norm) < float(min_displacement):
        raise ValueError("fallback_heading must have a non-zero horizontal component")
    return fallback_heading / fallback_norm


def fixed_sagittal_side_camera(
    joints: torch.Tensor,
    *,
    motion_heading: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one fixed, motion-oriented side camera for all render styles.

    ViMoGen uses a z-up world.  PyTorch3D maps positive camera-x to the left
    on screen, so the camera x-axis is deliberately ``-motion_heading``.
    Consequently the observed travel direction is displayed to the right for
    a positive heading in both the mesh and hand-projected skeleton paths.
    The camera is estimated once from M0 and reused for every method/frame.
    """

    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[1] < 1:
        raise ValueError(f"joints must have shape [T,J,3], got {tuple(joints.shape)}")
    if joints.shape[0] < 1:
        raise ValueError("joints must contain at least one frame")
    roots = joints[:, 0]
    if motion_heading is None:
        motion_heading = estimate_motion_heading(joints)
    motion_heading = motion_heading.to(device=joints.device, dtype=joints.dtype)
    if motion_heading.shape != (3,):
        raise ValueError("motion_heading must have shape [3]")
    if not bool(torch.isfinite(motion_heading).all()):
        raise ValueError("motion_heading must be finite")
    motion_norm = torch.linalg.vector_norm(motion_heading)
    if float(motion_norm) < 1e-8:
        raise ValueError("motion_heading must have a non-zero norm")
    motion_heading = motion_heading / motion_norm
    up = torch.zeros(3, dtype=joints.dtype, device=joints.device)
    up[2] = 1.0
    root_min = roots.amin(dim=0)
    root_max = roots.amax(dim=0)
    root_move = root_max - root_min
    # Columns map row-vectors to camera x/y/z.  The cross product keeps a
    # proper right-handed frame while viewing the subject from the side.
    camera_x = -motion_heading
    camera_z = torch.linalg.cross(camera_x, up)
    R_one = torch.stack((camera_x, up, camera_z), dim=1)
    # Keep the subject in front of the camera for either travel direction.
    root_depth = torch.matmul(roots, camera_z)
    depth_offset = 2.5 - root_depth.amin() + 0.5 * torch.linalg.vector_norm(root_move)
    T_one = torch.zeros(3, dtype=joints.dtype, device=joints.device)
    # Center the walking direction so the fixed view keeps the whole motion.
    root_center = 0.5 * (root_min + root_max)
    T_one[0] = torch.dot(root_center, motion_heading)
    T_one[1] = -roots[0, 2]
    T_one[2] = depth_offset
    seq_len = joints.shape[0]
    return R_one[None].repeat(seq_len, 1, 1), T_one[None].repeat(seq_len, 1)


def _load_motion(path: Path, device: str) -> torch.Tensor:
    motion = torch.load(path, map_location=device, weights_only=True)
    if isinstance(motion, dict):
        motion = motion["motion"]
    if motion.ndim != 2 or motion.shape[-1] != 276:
        raise ValueError(f"{path} must contain physical [T,276] motion")
    return motion.float()


@torch.no_grad()
def render_sources(
    motions: dict[str, torch.Tensor], output_dir: Path, device: str, render_style: str = "mesh",
    calibration: PelvisCalibration | None = None,
) -> dict[str, Path]:
    """Render all methods with the exact same fixed sagittal side camera."""

    output_dir.mkdir(parents=True, exist_ok=True)
    parameters, joints, root_rotations = {}, {}, {}
    for method, motion in motions.items():
        parameters[method], joints[method] = motion_rep_to_SMPL(
            motion.to(device), recover_from_velocity=True
        )
        root_rotations[method] = decode_rot6d_safe(
            motion[:, MOTION_LAYOUT.root_rotation]
        ).float()
    anatomical_headings, _ = person_forward_horizontal_axis(root_rotations["M0"])
    fallback_heading = anatomical_headings.mean(dim=0)
    display_heading = estimate_motion_heading(
        joints["M0"], fallback_heading=fallback_heading
    )
    camera_r, camera_t = fixed_sagittal_side_camera(
        joints["M0"], motion_heading=display_heading
    )
    audit_labels = None
    if calibration is not None:
        audit_labels = _v4_audit_labels(joints, root_rotations, calibration)
    if render_style == "skeleton":
        return _render_skeleton_sources(
            joints, root_rotations, camera_r, camera_t, output_dir,
            display_heading=display_heading,
            calibration=calibration, audit_labels=audit_labels
        )
    if render_style != "mesh":
        raise ValueError(f"unsupported render_style {render_style!r}")
    model = SMPLX(
        model_path=_default_smpl_model_path("smplx"),
        gender="neutral",
        num_betas=10,
        batch_size=motions["M0"].shape[0],
        use_pca=False,
    ).to(device)
    faces = torch.from_numpy(model.faces).long().to(device)
    outputs = {}
    for method in ("M0", "G0", "G1"):
        vertices = model(**parameters[method]).vertices
        path = output_dir / f"{method.lower()}_fixed_sagittal_side.mp4"
        render_and_save(
            verts=vertices[None],
            faces=faces,
            R=camera_r,
            T=camera_t,
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
            focal=estimate_focal_length(PANEL_WIDTH, PANEL_HEIGHT),
            batch_size=24,
            fps=FPS,
            output_path=str(path),
            motion_name=method,
        )
        annotated_path = output_dir / f"{method.lower()}_fixed_sagittal_side_annotated.mp4"
        _annotate_mesh_source(
            path,
            annotated_path,
            joints[method],
            root_rotations[method],
            camera_r,
            camera_t,
            calibration=calibration,
            display_heading=display_heading,
            audit_labels=audit_labels[method] if audit_labels is not None else None,
        )
        outputs[method] = annotated_path
    return outputs


def _render_skeleton_sources(
    joints: dict[str, torch.Tensor],
    root_rotations: dict[str, torch.Tensor],
    camera_r: torch.Tensor,
    camera_t: torch.Tensor,
    output_dir: Path,
    display_heading: torch.Tensor,
    calibration: PelvisCalibration | None = None,
    audit_labels: dict[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    """Render joints plus an explicit, non-exaggerated pelvis-angle marker.

    The red arrow is the measured anatomical P->A line; the cyan arrow is
    the sequence-level motion heading shared with the mesh renderer.  The
    numerical angle remains the v4 anatomical evaluator's angle.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for method in ("M0", "G0", "G1"):
        world = joints[method].float()
        projected, projected_heading, projected_sagittal, projected_motion, tilt_values = _pelvis_marker_projections(
            world, root_rotations[method], camera_r, camera_t,
            display_heading=display_heading, calibration=calibration
        )
        path = output_dir / f"{method.lower()}_fixed_sagittal_side_skeleton.mp4"
        command = [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s",
            f"{PANEL_WIDTH}x{PANEL_HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame_index, frame in enumerate(projected):
                image = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
                points = np.rint(frame).astype(np.int32)
                for child, parent in enumerate(SMPLX_22_PARENTS):
                    if parent < 0:
                        continue
                    p0, p1 = points[parent], points[child]
                    cv2.line(image, tuple(p0), tuple(p1), (210, 210, 210), 4, cv2.LINE_AA)
                _draw_pelvis_marker(
                    image,
                    frame_index,
                    points[0],
                    projected_heading,
                    projected_sagittal,
                    projected_motion,
                    tilt_values,
                    calibration=calibration,
                    audit_values=(None if audit_labels is None else {key: value[frame_index] for key, value in audit_labels[method].items()}),
                )
                for index, point in enumerate(points):
                    color = (40, 210, 255) if index in (0, 1, 2, 3) else (245, 245, 245)
                    cv2.circle(image, tuple(point), 7 if index == 0 else 5, color, -1, cv2.LINE_AA)
                process.stdin.write(image.tobytes())
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed while rendering {path}")
        outputs[method] = path
    return outputs


def _pelvis_marker_projections(
    world: torch.Tensor,
    root_rotation: torch.Tensor,
    camera_r: torch.Tensor,
    camera_t: torch.Tensor,
    display_heading: torch.Tensor | None = None,
    calibration: PelvisCalibration | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project pelvis endpoints and the shared motion direction.

    The explicit minus sign in screen-x matches PyTorch3D's camera convention
    used by the mesh renderer.  Without it, the hand-rendered skeleton is a
    horizontal mirror of the shaded mesh.
    """

    focal = estimate_focal_length(PANEL_WIDTH, PANEL_HEIGHT)

    def project(points: torch.Tensor) -> np.ndarray:
        camera = torch.matmul(points, camera_r) + camera_t[:, None]
        depth = camera[..., 2].clamp_min(1e-3)
        return torch.stack(
            (
                -focal * camera[..., 0] / depth + PANEL_WIDTH / 2.0,
                PANEL_HEIGHT / 2.0 - focal * camera[..., 1] / depth,
            ),
            dim=-1,
        ).detach().cpu().numpy()

    projected = project(world)
    root = world[:, :1]
    if calibration is not None:
        geometry = anatomical_pelvis_geometry(root_rotation, calibration, root_translation=root[:, 0])
        projected_posterior = project(geometry.posterior_point[:, None])[:, 0]
        projected_anterior = project(geometry.anterior_point[:, None])[:, 0]
        if display_heading is None:
            display_heading = estimate_motion_heading(world)
        marker_length = 0.75
        projected_motion = project(
            geometry.posterior_point[:, None]
            + marker_length * display_heading[None, None, :]
        )[:, 0]
        return (
            projected,
            projected_posterior,
            projected_anterior,
            projected_motion,
            geometry.angle_degrees.detach().cpu().numpy(),
        )
    heading, _ = person_forward_horizontal_axis(root_rotation)
    tilt = pelvis_sagittal_tilt_degrees(root_rotation)
    up = torch.zeros_like(heading)
    up[..., 2] = 1.0
    tilt_radians = tilt * (np.pi / 180.0)
    sagittal_forward = heading * torch.cos(tilt_radians)[..., None] + up * torch.sin(tilt_radians)[..., None]
    # The longer marker improves visual legibility without changing the angle.
    marker_length = 0.75
    if display_heading is None:
        display_heading = heading.mean(dim=0)
        display_heading = display_heading / torch.linalg.vector_norm(display_heading)
    projected_heading = project(root + marker_length * heading[:, None])[:, 0]
    projected_sagittal = project(root + marker_length * sagittal_forward[:, None])[:, 0]
    projected_motion = project(root + marker_length * display_heading[None, None, :])[:, 0]
    return projected, projected_heading, projected_sagittal, projected_motion, tilt.detach().cpu().numpy()


def _v4_audit_labels(
    joints: dict[str, torch.Tensor],
    root_rotations: dict[str, torch.Tensor],
    calibration: PelvisCalibration,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-frame torso/local-change labels shared by mesh and skeleton."""

    curves = {}
    for method in ("M0", "G0", "G1"):
        pelvis = anatomical_pelvis_geometry(root_rotations[method], calibration)
        segments = trunk_and_thigh_angles(joints[method].float(), pelvis)
        curves[method] = {
            "pelvis": pelvis.angle_degrees.detach().cpu().numpy(),
            "trunk": segments["trunk_deg"].detach().cpu().numpy(),
        }
    result = {}
    for method in ("M0", "G0", "G1"):
        delta_trunk = curves[method]["trunk"] - curves["M0"]["trunk"]
        delta_pelvis = curves[method]["pelvis"] - curves["M0"]["pelvis"]
        result[method] = {
            "delta_trunk": delta_trunk,
            "delta_local": delta_pelvis - delta_trunk,
            "anti_pass": ((np.abs(delta_trunk) <= 3.0) & np.isfinite(delta_trunk)),
        }
    return result


def _draw_pelvis_marker(
    image: np.ndarray,
    frame_index: int,
    root_point: np.ndarray | tuple[int, int],
    projected_heading: np.ndarray,
    projected_sagittal: np.ndarray,
    projected_motion: np.ndarray,
    tilt_values: np.ndarray,
    calibration: PelvisCalibration | None = None,
    audit_values: dict[str, float | bool] | None = None,
) -> None:
    """Draw the angle marker and label on one source frame."""

    heading_xy = tuple(np.rint(projected_heading[frame_index]).astype(np.int32))
    sagittal_xy = tuple(np.rint(projected_sagittal[frame_index]).astype(np.int32))
    motion_xy = tuple(np.rint(projected_motion[frame_index]).astype(np.int32))
    if calibration is not None:
        # For v4 the exact anatomical line is P -> A; do not substitute the
        # pelvis joint as its origin.  The cyan arrow is the sequence-level
        # motion direction used by the shared camera, so its screen direction
        # is identical in mesh and skeleton videos.
        root_xy = heading_xy
        cv2.circle(image, heading_xy, 8, (255, 210, 40), -1, cv2.LINE_AA)
        cv2.circle(image, sagittal_xy, 8, (30, 70, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(image, root_xy, sagittal_xy, (30, 70, 255), 6, cv2.LINE_AA, tipLength=0.16)
        cv2.arrowedLine(image, root_xy, motion_xy, (255, 210, 40), 4, cv2.LINE_AA, tipLength=0.16)
        angle = float(tilt_values[frame_index])
        label = f"Pelvis {'anterior' if angle >= 0 else 'posterior'} tilt {angle:+.1f} deg"
        cv2.putText(image, label, (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 70, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "P posterior | A anterior | cyan motion ->", (18, PANEL_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (235, 235, 235), 1, cv2.LINE_AA)
        # The inset is oriented by the observed motion: right is the same
        # sequence-level heading used by the camera.  The red line is the
        # actual projected P->A segment, rather than a hard-coded front arrow.
        x0, y0, iw, ih = PANEL_WIDTH - 190, 58, 170, 150
        cv2.rectangle(image, (x0, y0), (x0 + iw, y0 + ih), (245, 245, 245), -1)
        cv2.rectangle(image, (x0, y0), (x0 + iw, y0 + ih), (60, 60, 60), 1)
        center = (x0 + iw // 2, y0 + ih // 2)
        cv2.arrowedLine(image, center, (x0 + iw - 18, center[1]), (100, 100, 100), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.putText(image, "motion ->", (x0 + 76, y0 + ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)
        delta = np.asarray(sagittal_xy, dtype=np.float64) - np.asarray(heading_xy, dtype=np.float64)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            delta = np.array([1.0, 0.0], dtype=np.float64)
            delta_norm = 1.0
        endpoint = tuple(
            np.rint(np.asarray(center, dtype=np.float64) + 58.0 * delta / delta_norm).astype(np.int32)
        )
        cv2.arrowedLine(image, center, endpoint, (30, 70, 255), 4, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(image, "local sagittal", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(image, "red P -> A", (x0 + 8, y0 + ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)
        if audit_values is not None:
            dt = float(audit_values["delta_trunk"])
            dl = float(audit_values["delta_local"])
            status = "PASS" if bool(audit_values["anti_pass"]) else "REVIEW"
            cv2.putText(image, f"trunk vs M0 {dt:+.1f} deg", (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(image, f"pelvis-trunk {dl:+.1f} deg | anti {status}", (18, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 245), 1, cv2.LINE_AA)
        return
    root_xy = tuple(np.rint(root_point).astype(np.int32))
    # Cyan is the shared motion direction; red is the exact
    # heading-removed pelvis-forward direction used for theta.
    cv2.arrowedLine(image, root_xy, motion_xy, (255, 210, 40), 5, cv2.LINE_AA, tipLength=0.16)
    cv2.arrowedLine(image, root_xy, sagittal_xy, (30, 70, 255), 6, cv2.LINE_AA, tipLength=0.16)
    cv2.putText(
        image,
        f"pelvis tilt {tilt_values[frame_index]:+.1f} deg",
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 70, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "red: pelvis forward | cyan: motion heading",
        (18, PANEL_HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def _annotate_mesh_source(
    source_path: Path,
    output_path: Path,
    world: torch.Tensor,
    root_rotation: torch.Tensor,
    camera_r: torch.Tensor,
    camera_t: torch.Tensor,
    display_heading: torch.Tensor,
    calibration: PelvisCalibration | None = None,
    audit_labels: dict[str, np.ndarray] | None = None,
) -> None:
    """Overlay the same pelvis marker on a shaded mesh source video."""

    projected, projected_heading, projected_sagittal, projected_motion, tilt_values = _pelvis_marker_projections(
        world.float(), root_rotation, camera_r, camera_t,
        display_heading=display_heading, calibration=calibration
    )
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open mesh source {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{PANEL_WIDTH}x{PANEL_HEIGHT}", "-r", str(FPS),
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_index in range(len(projected)):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"mesh source ended at frame {frame_index}")
            image = cv2.resize(image, (PANEL_WIDTH, PANEL_HEIGHT))
            _draw_pelvis_marker(
                image,
                frame_index,
                projected[frame_index, 0],
                projected_heading,
                projected_sagittal,
                projected_motion,
                tilt_values,
                calibration=calibration,
                audit_values=(None if audit_labels is None else {key: value[frame_index] for key, value in audit_labels.items()}),
            )
            process.stdin.write(image.tobytes())
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while annotating {output_path}")
def _read_angles(path: Path, sample_id: str) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["sample_id"]) == str(sample_id):
                rows.append(row)
    if not rows:
        raise ValueError(f"sample {sample_id} not found in {path}")
    rows.sort(key=lambda row: int(row["frame"]))
    return {
        "M0": np.array([float(row["m0_angle_deg"]) for row in rows]),
        "G0": np.array([float(row["g0_angle_deg"]) for row in rows]),
        "G1": np.array([float(row["g1_angle_deg"]) for row in rows]),
    }


def _plot_panel(
    curves: dict[str, np.ndarray], frame: int, target: float, correction: float
) -> np.ndarray:
    canvas = np.full((PLOT_HEIGHT, WIDTH, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 90, WIDTH - 40, 35, PLOT_HEIGHT - 65
    all_values = np.concatenate([*curves.values(), np.array([target])])
    low = float(np.floor(all_values.min() - 2.0))
    high = float(np.ceil(all_values.max() + 2.0))
    if high - low < 5:
        high = low + 5

    def point(index: int, value: float) -> tuple[int, int]:
        x = left + int(index * (right - left) / max(len(curves["M0"]) - 1, 1))
        y = bottom - int((value - low) * (bottom - top) / (high - low))
        return x, y

    cv2.rectangle(canvas, (left, top), (right, bottom), (40, 40, 40), 1)
    target_y = point(0, target)[1]
    cv2.line(canvas, (left, target_y), (right, target_y), (30, 30, 210), 2)
    cv2.putText(canvas, f"target {target:+.1f} deg", (right - 220, target_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 210), 2)
    for method, curve in curves.items():
        color = COLORS[method]
        points = np.array([point(i, float(curve[i])) for i in range(frame + 1)], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, point(frame, float(curve[frame])), 5, color, -1)
    x_text = 110
    for method in ("M0", "G0", "G1"):
        current = float(curves[method][frame])
        running = float(curves[method][: frame + 1].mean())
        text = f"{method}: current {current:+.2f} deg, running mean {running:+.2f} deg"
        cv2.putText(canvas, text, (x_text, PLOT_HEIGHT - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.63, COLORS[method], 2, cv2.LINE_AA)
        x_text += 570
    cv2.putText(canvas, f"G1 terminal correction: {correction:+.3f} deg", (110, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def compose(
    *,
    videos: dict[str, Path],
    curves: dict[str, np.ndarray],
    target: float,
    correction: float,
    output: Path,
) -> dict:
    captures = {method: cv2.VideoCapture(str(path)) for method, path in videos.items()}
    if not all(capture.isOpened() for capture in captures.values()):
        raise RuntimeError("failed to open one or more source videos")
    frame_count = min(
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures.values()
    )
    frame_count = min(frame_count, *(len(curve) for curve in curves.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-vcodec", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_index in range(frame_count):
            panels = []
            for method in ("M0", "G0", "G1"):
                ok, image = captures[method].read()
                if not ok:
                    raise RuntimeError(f"source {method} ended at frame {frame_index}")
                image = cv2.resize(image, (PANEL_WIDTH, PANEL_HEIGHT))
                cv2.rectangle(image, (0, 0), (PANEL_WIDTH, 48), (250, 250, 250), -1)
                cv2.putText(image, method, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[method], 2, cv2.LINE_AA)
                panels.append(image)
            top = np.concatenate(panels, axis=1)
            plot = _plot_panel(curves, frame_index, target, correction)
            process.stdin.write(np.concatenate((top, plot), axis=0).tobytes())
    finally:
        for capture in captures.values():
            capture.release()
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate",
            "-of", "json", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    expected = {"codec_name": "h264", "width": WIDTH, "height": HEIGHT, "r_frame_rate": "20/1"}
    for key, value in expected.items():
        if stream.get(key) != value:
            raise RuntimeError(f"video verification failed for {key}: {stream.get(key)} != {value}")
    return {"output": str(output), "frames": frame_count, **stream}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-motion", type=Path, required=True)
    parser.add_argument("--g0-motion", type=Path, required=True)
    parser.add_argument("--g1-motion", type=Path, required=True)
    parser.add_argument("--angles-csv", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--target-mean-deg", type=float, choices=[5.0, 10.0], required=True)
    parser.add_argument("--terminal-correction-deg", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--render-style",
        choices=["mesh", "skeleton"],
        default="mesh",
        help="Render shaded SMPL-X meshes or a joint-and-bone skeleton.",
    )
    parser.add_argument(
        "--camera-view",
        choices=["sagittal_side"],
        default="sagittal_side",
        help="Frozen display camera; v2 uses a true z-up sagittal side view.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration", type=Path, default=None, help="v4 frozen LASI/RASI/LPSI/RPSI calibration JSON")
    args = parser.parse_args()
    motions = {
        "M0": _load_motion(args.m0_motion, args.device),
        "G0": _load_motion(args.g0_motion, args.device),
        "G1": _load_motion(args.g1_motion, args.device),
    }
    calibration = None
    if args.calibration is not None:
        from motion_rep.anatomical_pelvis import load_pelvis_calibration
        calibration = load_pelvis_calibration(args.calibration)
    videos = render_sources(motions, args.source_dir, args.device, args.render_style, calibration=calibration)
    curves = _read_angles(args.angles_csv, args.sample_id)
    print(json.dumps(compose(
        videos=videos,
        curves=curves,
        target=args.target_mean_deg,
        correction=args.terminal_correction_deg,
        output=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
