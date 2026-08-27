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


WIDTH, HEIGHT, FPS = 1920, 1080, 20
PANEL_WIDTH, PANEL_HEIGHT, PLOT_HEIGHT = 640, 720, 360
COLORS = {"M0": (60, 60, 60), "G0": (220, 110, 20), "G1": (30, 150, 30)}
SMPLX_22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def fixed_sagittal_side_camera(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one fixed camera looking from the person's side.

    ViMoGen uses a z-up world.  The former renderer used ``x`` as image
    horizontal, which is a frontal view for this convention.  Here image
    horizontal is world ``y`` (the walking/forward direction), image vertical
    is world ``z``, and camera depth is world ``x``.  The camera is fixed for
    all three methods and all frames; only its framing is derived from M0.
    """

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joints must have shape [T,J,3], got {tuple(joints.shape)}")
    roots = joints[:, 0]
    root_min = roots.amin(dim=0)
    root_max = roots.amax(dim=0)
    root_move = root_max - root_min
    # ``render_depth_maps`` multiplies row-vectors on the left, so the
    # columns below implement x_cam <- y_world, y_cam <- z_world,
    # z_cam <- x_world.  This is a true sagittal side projection.
    R_one = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=joints.dtype,
        device=joints.device,
    )
    # Keep the same framing scale as the legacy camera: use root-motion
    # ranges, not the subject's full body height, for the depth offset.
    depth_offset = 2.5 + 1.0 * root_move[1] + 2.0 * root_move[2] + root_max[0]
    T_one = torch.zeros(3, dtype=joints.dtype, device=joints.device)
    # Center the walking direction so the fixed view keeps the whole motion.
    T_one[0] = -0.5 * (root_min[1] + root_max[1])
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
    motions: dict[str, torch.Tensor], output_dir: Path, device: str, render_style: str = "mesh"
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
    camera_r, camera_t = fixed_sagittal_side_camera(joints["M0"])
    if render_style == "skeleton":
        return _render_skeleton_sources(
            joints, root_rotations, camera_r, camera_t, output_dir
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
        )
        outputs[method] = annotated_path
    return outputs


def _render_skeleton_sources(
    joints: dict[str, torch.Tensor],
    root_rotations: dict[str, torch.Tensor],
    camera_r: torch.Tensor,
    camera_t: torch.Tensor,
    output_dir: Path,
) -> dict[str, Path]:
    """Render joints plus an explicit, non-exaggerated pelvis-angle marker.

    The red arrow is the pelvis local forward axis after removing the current
    person heading; the cyan arrow is that frame's horizontal heading.  The
    angle between them is exactly the angle used by the v3 evaluator.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for method in ("M0", "G0", "G1"):
        world = joints[method].float()
        projected, projected_heading, projected_sagittal, tilt_values = _pelvis_marker_projections(
            world, root_rotations[method], camera_r, camera_t
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
                    tilt_values,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the exact measured pelvis direction and its local horizontal reference."""

    focal = estimate_focal_length(PANEL_WIDTH, PANEL_HEIGHT)

    def project(points: torch.Tensor) -> np.ndarray:
        camera = torch.matmul(points, camera_r) + camera_t[:, None]
        depth = camera[..., 2].clamp_min(1e-3)
        return torch.stack(
            (
                focal * camera[..., 0] / depth + PANEL_WIDTH / 2.0,
                PANEL_HEIGHT / 2.0 - focal * camera[..., 1] / depth,
            ),
            dim=-1,
        ).detach().cpu().numpy()

    projected = project(world)
    root = world[:, :1]
    heading, _ = person_forward_horizontal_axis(root_rotation)
    tilt = pelvis_sagittal_tilt_degrees(root_rotation)
    up = torch.zeros_like(heading)
    up[..., 2] = 1.0
    tilt_radians = tilt * (np.pi / 180.0)
    sagittal_forward = heading * torch.cos(tilt_radians)[..., None] + up * torch.sin(tilt_radians)[..., None]
    # The longer marker improves visual legibility without changing the angle.
    marker_length = 0.75
    projected_heading = project(root + marker_length * heading[:, None])[:, 0]
    projected_sagittal = project(root + marker_length * sagittal_forward[:, None])[:, 0]
    return projected, projected_heading, projected_sagittal, tilt.detach().cpu().numpy()


def _draw_pelvis_marker(
    image: np.ndarray,
    frame_index: int,
    root_point: np.ndarray | tuple[int, int],
    projected_heading: np.ndarray,
    projected_sagittal: np.ndarray,
    tilt_values: np.ndarray,
) -> None:
    """Draw the angle marker and label on one source frame."""

    root_xy = tuple(np.rint(root_point).astype(np.int32))
    heading_xy = tuple(np.rint(projected_heading[frame_index]).astype(np.int32))
    sagittal_xy = tuple(np.rint(projected_sagittal[frame_index]).astype(np.int32))
    # Cyan is the local horizontal heading; red is the exact heading-removed
    # pelvis-forward direction used for theta.
    cv2.arrowedLine(image, root_xy, heading_xy, (255, 210, 40), 5, cv2.LINE_AA, tipLength=0.16)
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
        "red: pelvis forward | cyan: horizontal heading",
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
) -> None:
    """Overlay the same pelvis marker on a shaded mesh source video."""

    projected, projected_heading, projected_sagittal, tilt_values = _pelvis_marker_projections(
        world.float(), root_rotation, camera_r, camera_t
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
                tilt_values,
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
    args = parser.parse_args()
    motions = {
        "M0": _load_motion(args.m0_motion, args.device),
        "G0": _load_motion(args.g0_motion, args.device),
        "G1": _load_motion(args.g1_motion, args.device),
    }
    videos = render_sources(motions, args.source_dir, args.device, args.render_style)
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
