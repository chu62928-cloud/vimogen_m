#!/usr/bin/env python3
"""Render M0/candidate mesh and foot-local audits for v3."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import pyrender

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import pelvis_pitch_delta_deg
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion, direct_smpl_parameters
from motion_rep.smplx_utils import default_smpl_model_path
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from scripts.render_absolute_mean_triptych import estimate_motion_heading, fixed_sagittal_side_camera
from scripts.render_relative_root_forward_v2_single_mesh import _draw_direction_overlay, _render_panel
from scripts.render_relative_root_forward_v1_1 import _panel_vectors, _root_geometry


WIDTH, HEIGHT, FPS = 1280, 1080, 20
PANEL_WIDTH, PANEL_HEIGHT, PLOT_HEIGHT = 640, 720, 360


def _encoder(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )


def _plot(curve: np.ndarray, target: float, frame: int) -> np.ndarray:
    image = np.full((PLOT_HEIGHT, WIDTH, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 90, WIDTH - 40, 35, PLOT_HEIGHT - 65
    values = np.concatenate((curve[np.isfinite(curve)], np.asarray([0.0, target], dtype=np.float32)))
    low, high = float(np.floor(values.min() - 2.0)), float(np.ceil(values.max() + 2.0))
    if high - low < 5:
        high = low + 5
    cv2.rectangle(image, (left, top), (right, bottom), (40, 40, 40), 1)
    y_target = bottom - int((target - low) * (bottom - top) / (high - low))
    cv2.line(image, (left, y_target), (right, y_target), (30, 170, 255), 2)
    points = []
    for index in range(frame + 1):
        if np.isfinite(curve[index]):
            x = left + int(index * (right - left) / max(len(curve) - 1, 1))
            y = bottom - int((curve[index] - low) * (bottom - top) / (high - low))
            points.append((x, y))
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (30, 170, 255), 3, cv2.LINE_AA)
    if points:
        cv2.circle(image, points[-1], 6, (30, 170, 255), -1, cv2.LINE_AA)
    cv2.putText(image, "pelvis dose relative to M0 (v1.3 sign; downward positive)", (90, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, f"target {target:+.1f} deg    current {curve[frame]:+.2f} deg", (90, PLOT_HEIGHT - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    m0 = torch.load(args.run_root / "m0_physical.pt", map_location="cpu", weights_only=True).float()
    candidate = torch.load(args.run_root / "selected_motion.pt", map_location="cpu", weights_only=True).float()
    if m0.ndim == 3:
        m0 = m0[0]
    if candidate.ndim == 3:
        candidate = candidate[0]
    if m0.shape != candidate.shape:
        raise ValueError("M0 and candidate shapes differ")
    device = torch.device(args.device)
    model_path = args.model_path or default_smpl_model_path("smplx", ROOT)
    model = __import__("smplx").SMPLX(model_path=str(model_path), gender="neutral", num_betas=10, batch_size=int(m0.shape[0]), use_pca=False).to(device)
    motions = {"M0": m0, "candidate": candidate}
    vertices, joints, roots = {}, {}, {}
    with torch.inference_mode():
        for name, motion in motions.items():
            params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
            params = {key: value[0] for key, value in params.items()}
            vertices[name] = model(**params, return_verts=True).vertices.detach().cpu().numpy()
            joints[name] = direct_joints_from_motion(motion.unsqueeze(0))[0].cpu()
            roots[name] = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation]).float()
    faces = np.asarray(model.faces, dtype=np.int32)
    camera_heading = estimate_motion_heading(joints["M0"])
    camera_r, camera_t = fixed_sagittal_side_camera(joints["M0"], motion_heading=camera_heading)
    camera_r_np, camera_t_np = camera_r.numpy(), camera_t.numpy()
    output_dir = args.output_dir or (args.run_root / "videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "pelvis_contact_v3_M0_candidate.mp4"
    foot_path = output_dir / "pelvis_contact_v3_foot_local.mp4"
    grid = _encoder(grid_path)
    foot = _encoder(foot_path)
    renderer = pyrender.OffscreenRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    dose_curve = pelvis_pitch_delta_deg(roots["M0"], roots["candidate"]).numpy()
    target_vectors = _panel_vectors(joints["M0"], roots["M0"], _root_geometry(roots["M0"])[0], camera_r, camera_t)
    try:
        for frame in range(m0.shape[0]):
            panels = []
            for name, label, material_color in (("M0", "M0 baseline", [0.55, 0.58, 0.62, 1.0]), ("candidate", f"v3 candidate {args.target_delta_deg:+g} deg", [0.10, 0.55, 0.92, 1.0])):
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=material_color, metallicFactor=0.0, roughnessFactor=0.82)
                panel = _render_panel(vertices[name], faces, camera_r_np[frame], camera_t_np[frame], frame, renderer, material)
                cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 54), (245, 245, 245), -1)
                cv2.putText(panel, label, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)
                panels.append(panel)
            top = np.concatenate(panels, axis=1)
            bottom = _plot(dose_curve, float(args.target_delta_deg), frame)
            frame_image = np.concatenate((top, bottom), axis=0)
            assert grid.stdin is not None and foot.stdin is not None
            grid.stdin.write(frame_image.tobytes())
            foot_image = cv2.resize(top[PANEL_HEIGHT // 2:, :], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            foot.stdin.write(foot_image.tobytes())
    finally:
        renderer.delete()
        for process in (grid, foot):
            if process.stdin is not None:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed while rendering v3 video")
    print(json.dumps({"grid": str(grid_path), "foot": str(foot_path)}, ensure_ascii=False))


if __name__ == "__main__":
    import json
    main()
