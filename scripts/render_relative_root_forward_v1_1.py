#!/usr/bin/env python3
"""Render v1.1 root-forward audit videos from frozen server artifacts.

The renderer is deliberately independent of the v4 anatomical-pelvis video
path.  It uses one camera estimated from M0 and overlays the actual root
forward, the frozen-M0 target forward, and the spine1-to-neck trunk direction
on both shaded meshes and the hand-projected skeleton.
"""

from __future__ import annotations

import argparse
import math
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
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from motion_rep.retarget_motion import motion_rep_to_SMPL  # noqa: E402
from motion_rep.rotation_transform import axis_angle_to_mat3x3  # noqa: E402
from motion_rep.pose_authority import _root_forward  # noqa: E402
from scripts.render_absolute_mean_triptych import (  # noqa: E402
    estimate_motion_heading,
    fixed_sagittal_side_camera,
)


WIDTH, HEIGHT, FPS = 1920, 1080, 20
PANEL_WIDTH, PANEL_HEIGHT, PLOT_HEIGHT = 640, 720, 360
COLORS = {
    "M0": (100, 100, 100),
    "-": (255, 150, 30),
    "+": (30, 170, 255),
}
SMPLX_22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
SPINE1_INDEX, NECK_INDEX = 9, 12


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


def _latest_attempt(config_dir: Path, delta: int) -> Path:
    sign = "+" if delta >= 0 else ""
    parent = config_dir / f"delta_{sign}{delta}deg"
    attempts = sorted(
        (
            p
            for p in parent.glob("attempt_*")
            if (p / "guided_artifacts" / "batch_000" / "g0_norm_batch.pt").is_file()
        ),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not attempts:
        raise FileNotFoundError(f"no completed attempt under {parent}")
    return attempts[-1]


def _physical_batch(path: Path, index: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    value = _load(path)
    if value.ndim != 3 or value.shape[-1] != 276:
        raise ValueError(f"{path} must have shape [B,T,276]")
    return value[index] * std + mean


def _root_geometry(root: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _root_forward(root)


def _target_forward(
    f0: torch.Tensor, r0: torch.Tensor, delta_deg: float
) -> torch.Tensor:
    correction = axis_angle_to_mat3x3(
        r0 * (-float(delta_deg) * math.pi / 180.0)
    )
    return (correction @ f0.unsqueeze(-1)).squeeze(-1)


def _project(points: torch.Tensor, camera_r: torch.Tensor, camera_t: torch.Tensor) -> np.ndarray:
    focal = estimate_focal_length(PANEL_WIDTH, PANEL_HEIGHT)
    camera = torch.matmul(points, camera_r) + camera_t[:, None]
    depth = camera[..., 2].clamp_min(1e-3)
    projected = torch.stack(
        (
            -focal * camera[..., 0] / depth + PANEL_WIDTH / 2.0,
            PANEL_HEIGHT / 2.0 - focal * camera[..., 1] / depth,
        ),
        dim=-1,
    )
    return projected.detach().cpu().numpy()


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-7)


def _draw_arrow(
    image: np.ndarray,
    origin: np.ndarray,
    endpoint: np.ndarray,
    color: tuple[int, int, int],
    width: int = 5,
) -> None:
    p0 = tuple(np.rint(origin).astype(np.int32))
    p1 = tuple(np.rint(endpoint).astype(np.int32))
    cv2.arrowedLine(image, p0, p1, color, width, cv2.LINE_AA, tipLength=0.16)


def _panel_vectors(
    joints: torch.Tensor,
    root: torch.Tensor,
    target: torch.Tensor,
    camera_r: torch.Tensor,
    camera_t: torch.Tensor,
    *,
    length: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actual = root @ torch.tensor([0.0, 0.0, 1.0], dtype=root.dtype, device=root.device)
    trunk = _unit(joints[:, NECK_INDEX] - joints[:, SPINE1_INDEX])
    points = torch.stack(
        (
            joints[:, 0],
            joints[:, 0] + length * _unit(actual),
            joints[:, 0] + length * _unit(target),
            joints[:, SPINE1_INDEX],
            joints[:, SPINE1_INDEX] + length * trunk,
        ),
        dim=1,
    )
    projected = _project(points, camera_r, camera_t)
    return projected[:, 0], projected[:, 1], projected[:, 2], projected[:, 3:]


def _draw_direction_overlay(
    image: np.ndarray,
    frame: int,
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    label: str,
    root_change_deg: float,
    target_delta_deg: float,
) -> None:
    root_xy, actual_xy, target_xy, trunk_pair = vectors
    _draw_arrow(image, root_xy[frame], actual_xy[frame], (255, 210, 40), 5)
    _draw_arrow(image, root_xy[frame], target_xy[frame], (40, 70, 255), 5)
    _draw_arrow(image, trunk_pair[frame, 0], trunk_pair[frame, 1], (210, 70, 220), 5)
    cv2.rectangle(image, (0, 0), (PANEL_WIDTH, 55), (245, 245, 245), -1)
    cv2.putText(image, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"root change {root_change_deg:+.2f} deg | target {target_delta_deg:+.1f} deg",
        (18, PANEL_HEIGHT - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "yellow actual root | red target root | magenta spine1->neck",
        (18, PANEL_HEIGHT - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _skeleton_panel(
    joints: torch.Tensor,
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    frame: int,
    *,
    label: str,
    root_change_deg: float,
    target_delta_deg: float,
    camera_r: torch.Tensor,
    camera_t: torch.Tensor,
) -> np.ndarray:
    projected = _project(joints, camera_r, camera_t)
    image = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 25, dtype=np.uint8)
    points = np.rint(projected[frame]).astype(np.int32)
    for child, parent in enumerate(SMPLX_22_PARENTS):
        if parent >= 0:
            cv2.line(image, tuple(points[parent]), tuple(points[child]), (200, 200, 200), 4, cv2.LINE_AA)
    for index, point in enumerate(points):
        color = (40, 210, 255) if index in (0, 1, 2, 3) else (245, 245, 245)
        cv2.circle(image, tuple(point), 7 if index == 0 else 5, color, -1, cv2.LINE_AA)
    _draw_direction_overlay(
        image, frame, vectors, label=label, root_change_deg=root_change_deg,
        target_delta_deg=target_delta_deg,
    )
    return image


def _plot(
    curves: dict[str, np.ndarray],
    frame: int,
    target_pair: tuple[float, float],
) -> np.ndarray:
    image = np.full((PLOT_HEIGHT, WIDTH, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 90, WIDTH - 40, 35, PLOT_HEIGHT - 65
    all_values = np.concatenate([*curves.values(), np.asarray(target_pair, dtype=np.float32), np.zeros(1)])
    low = float(np.floor(all_values.min() - 2.0))
    high = float(np.ceil(all_values.max() + 2.0))
    if high - low < 5:
        high = low + 5

    def point(index: int, value: float) -> tuple[int, int]:
        x = left + int(index * (right - left) / max(len(curves["M0"]) - 1, 1))
        y = bottom - int((value - low) * (bottom - top) / (high - low))
        return x, y

    cv2.rectangle(image, (left, top), (right, bottom), (40, 40, 40), 1)
    for target, color in zip(target_pair, ((255, 150, 30), (30, 170, 255))):
        y = point(0, float(target))[1]
        cv2.line(image, (left, y), (right, y), color, 2, cv2.LINE_AA)
        cv2.putText(image, f"target {target:+.1f} deg", (right - 215, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    for method, curve in curves.items():
        color = COLORS[method]
        points = np.asarray([point(i, float(curve[i])) for i in range(frame + 1)], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)
        cv2.circle(image, point(frame, float(curve[frame])), 6, color, -1, cv2.LINE_AA)
    x = 105
    for method in ("M0", "-", "+"):
        current = float(curves[method][frame])
        cv2.putText(image, f"{method}: {current:+.2f} deg", (x, PLOT_HEIGHT - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[method], 2, cv2.LINE_AA)
        x += 600
    cv2.putText(image, "root-forward change relative to M0 (downward positive)", (90, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
    return image


def _encoder(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
    )


def render_sample(
    motions: dict[str, torch.Tensor],
    deltas: tuple[int, int],
    output_dir: Path,
    sample_label: str,
    device: str,
) -> list[Path]:
    parameters, joints, roots = {}, {}, {}
    for method, motion in motions.items():
        parameters[method], joints[method] = motion_rep_to_SMPL(motion.to(device), recover_from_velocity=True)
        roots[method] = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation]).float().to(joints[method].device)
    display_heading = estimate_motion_heading(joints["M0"])
    camera_r, camera_t = fixed_sagittal_side_camera(joints["M0"], motion_heading=display_heading)
    f0, _, r0, phi0 = _root_geometry(roots["M0"])
    root_change = {
        method: (phi0 - _root_geometry(roots[method])[3]).detach().cpu().numpy()
        for method in motions
    }
    vectors = {}
    for method, motion in motions.items():
        method_delta = 0.0 if method == "M0" else float(deltas[0] if method == "-" else deltas[1])
        target = _target_forward(f0, r0, method_delta)
        vectors[method] = _panel_vectors(joints[method], roots[method], target, camera_r, camera_t)

    model = SMPLX(
        model_path=_default_smpl_model_path("smplx"), gender="neutral", num_betas=10,
        batch_size=motions["M0"].shape[0], use_pca=False,
    ).to(device)
    faces = torch.from_numpy(model.faces).long().to(device)
    mesh_sources: dict[str, Path] = {}
    source_dir = output_dir / "mesh_sources" / sample_label
    for method in ("M0", "-", "+"):
        vertices = model(**parameters[method]).vertices
        path = source_dir / f"{method.replace('-', 'minus').replace('+', 'plus')}.mp4"
        render_and_save(
            verts=vertices[None], faces=faces, R=camera_r, T=camera_t,
            width=PANEL_WIDTH, height=PANEL_HEIGHT,
            focal=estimate_focal_length(PANEL_WIDTH, PANEL_HEIGHT), batch_size=24,
            fps=FPS, output_path=str(path), motion_name=method,
        )
        mesh_sources[method] = path

    outputs: list[Path] = []
    for style in ("mesh", "skeleton"):
        name = f"{sample_label}_{style}_M0_{deltas[0]:+g}_{deltas[1]:+g}.mp4"
        output = output_dir / name
        process = _encoder(output)
        captures = {method: cv2.VideoCapture(str(path)) for method, path in mesh_sources.items()} if style == "mesh" else {}
        try:
            frame_count = joints["M0"].shape[0]
            for frame in range(frame_count):
                panels = []
                for method in ("M0", "-", "+"):
                    target_delta = 0.0 if method == "M0" else float(deltas[0] if method == "-" else deltas[1])
                    label = "M0" if method == "M0" else f"G0 {target_delta:+.0f} deg"
                    if style == "mesh":
                        ok, panel = captures[method].read()
                        if not ok:
                            raise RuntimeError(f"mesh source ended early: {mesh_sources[method]}")
                        panel = cv2.resize(panel, (PANEL_WIDTH, PANEL_HEIGHT))
                    else:
                        panel = _skeleton_panel(
                            joints[method].float().cpu(), vectors[method], frame,
                            label=label, root_change_deg=float(root_change[method][frame]),
                            target_delta_deg=target_delta,
                            camera_r=camera_r.cpu(), camera_t=camera_t.cpu(),
                        )
                    if style == "mesh":
                        _draw_direction_overlay(
                            panel, frame, vectors[method], label=label,
                            root_change_deg=float(root_change[method][frame]),
                            target_delta_deg=target_delta,
                        )
                    panels.append(panel)
                curves = {"M0": root_change["M0"], "-": root_change["-"], "+": root_change["+"]}
                bottom = _plot(curves, frame, (float(deltas[0]), float(deltas[1])))
                process.stdin.write(np.concatenate((np.concatenate(panels, axis=1), bottom), axis=0).tobytes())
        finally:
            for capture in captures.values():
                capture.release()
            if process.stdin is not None:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed while rendering {output}")
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--sample-label", required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()
    m0_attempt = _latest_attempt(args.config_dir, 5)
    motions = {
        "M0": _physical_batch(m0_attempt / "guided_artifacts" / "batch_000" / "m0_consistent_norm_batch.pt", args.sample_index, mean, std),
    }
    for method, delta in (("-", -5), ("+", 5), ("minus10", -10), ("plus10", 10)):
        attempt = _latest_attempt(args.config_dir, delta)
        motions[method if abs(delta) == 5 else ("-10" if delta < 0 else "+10")] = _physical_batch(
            attempt / "guided_artifacts" / "batch_000" / "g0_norm_batch.pt", args.sample_index, mean, std
        )
    outputs = []
    for pair in ((-5, 5), (-10, 10)):
        pair_motions = {"M0": motions["M0"], "-": motions["-" if pair[0] == -5 else "-10"], "+": motions["+" if pair[1] == 5 else "+10"]}
        outputs.extend(render_sample(pair_motions, pair, args.output_dir, args.sample_label, args.device))
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
