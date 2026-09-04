#!/usr/bin/env python3
"""Render the full sample94 walk as M0/root-only/compensated panels."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import pelvis_pitch_delta_deg, target_root_rotation
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion, direct_smpl_parameters
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe, encode_rot6d
from motion_rep.pose_authority import authority_project
from motion_rep.smplx_utils import default_smpl_model_path
from scripts.render_pelvis_contact_v3_mesh import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    _render_panel,
    estimate_motion_heading,
    fixed_sagittal_side_camera,
)


WIDTH, HEIGHT, FPS = 1920, 1080, 20
PLOT_HEIGHT = HEIGHT - PANEL_HEIGHT


def _encoder(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
    )


def _root_only_motion(m0: torch.Tensor, target_delta_deg: float) -> torch.Tensor:
    root = decode_rot6d_safe(m0[..., MOTION_LAYOUT.root_rotation])
    direct = m0.clone()
    direct[..., MOTION_LAYOUT.root_rotation] = encode_rot6d(target_root_rotation(root, target_delta_deg))
    valid = torch.ones((1, m0.shape[0]), dtype=torch.bool)
    return authority_project(direct.unsqueeze(0), valid_mask=valid, output_dtype=torch.float32).physical_motion[0]


def _axis_change_deg(m0_joints: torch.Tensor, candidate_joints: torch.Tensor, end_name: str) -> np.ndarray:
    pelvis = SMPLX_22_JOINT_INDEX["pelvis"]
    end = SMPLX_22_JOINT_INDEX[end_name]
    base = torch.nn.functional.normalize(m0_joints[:, end] - m0_joints[:, pelvis], dim=-1)
    changed = torch.nn.functional.normalize(candidate_joints[:, end] - candidate_joints[:, pelvis], dim=-1)
    angle = torch.acos((base * changed).sum(-1).clamp(-1.0, 1.0)) * 180.0 / math.pi
    return angle.cpu().numpy()


def _x(index: int, frames: int, left: int, right: int) -> int:
    return left + int(index * (right - left) / max(frames - 1, 1))


def _curve_points(curve: np.ndarray, frame: int, low: float, high: float, left: int, right: int, top: int, bottom: int) -> np.ndarray:
    points = []
    for index in range(frame + 1):
        if np.isfinite(curve[index]):
            y = bottom - int((float(curve[index]) - low) * (bottom - top) / max(high - low, 1e-6))
            points.append((_x(index, len(curve), left, right), y))
    return np.asarray(points, dtype=np.int32)


def _plot(dose_root_only: np.ndarray, dose_candidate: np.ndarray, neck: np.ndarray, head: np.ndarray, target: float, frame: int) -> np.ndarray:
    image = np.full((PLOT_HEIGHT, WIDTH, 3), 248, dtype=np.uint8)
    left, right = 95, WIDTH - 45
    dose_top, dose_bottom = 38, 150
    posture_top, posture_bottom = 205, PLOT_HEIGHT - 45
    cv2.putText(image, "pelvis dose (deg): orange=root-only, blue=compensated", (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.rectangle(image, (left, dose_top), (right, dose_bottom), (55, 55, 55), 1)
    dose_values = np.concatenate((dose_root_only, dose_candidate, np.asarray([0.0, target], dtype=np.float32)))
    dose_low = float(np.floor(np.nanmin(dose_values) - 1.0))
    dose_high = float(np.ceil(np.nanmax(dose_values) + 1.0))
    target_y = dose_bottom - int((target - dose_low) * (dose_bottom - dose_top) / max(dose_high - dose_low, 1e-6))
    cv2.line(image, (left, target_y), (right, target_y), (70, 70, 70), 1)
    for curve, colour in ((dose_root_only, (0, 165, 255)), (dose_candidate, (235, 110, 20))):
        points = _curve_points(curve, frame, dose_low, dose_high, left, right, dose_top, dose_bottom)
        if len(points) > 1:
            cv2.polylines(image, [points], False, colour, 3, cv2.LINE_AA)
    cv2.putText(image, "whole-body change (deg): magenta=pelvis-neck, green=pelvis-head", (left, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.rectangle(image, (left, posture_top), (right, posture_bottom), (55, 55, 55), 1)
    posture_high = max(float(np.nanmax(np.concatenate((neck, head)))) + 1.0, 5.0)
    for curve, colour in ((neck, (190, 35, 190)), (head, (40, 150, 40))):
        points = _curve_points(curve, frame, 0.0, posture_high, left, right, posture_top, posture_bottom)
        if len(points) > 1:
            cv2.polylines(image, [points], False, colour, 3, cv2.LINE_AA)
    cv2.putText(
        image,
        f"frame {frame + 1}/{len(dose_candidate)}   target {target:+.1f}   compensated {dose_candidate[frame]:+.2f}   neck {neck[frame]:.2f}   head {head[frame]:.2f}",
        (left, PLOT_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 2, cv2.LINE_AA,
    )
    return image


def _slow_copy(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", "setpts=2.0*PTS", "-an", "-r", str(FPS),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    m0 = torch.load(args.run_root / "m0_physical.pt", map_location="cpu", weights_only=True).float()
    candidate_path = args.candidate_path or (args.run_root / "diagnostic_motion.pt")
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True).float()
    if m0.ndim == 3:
        m0 = m0[0]
    if candidate.ndim == 3:
        candidate = candidate[0]
    if m0.shape != candidate.shape:
        raise ValueError("M0 and candidate shapes differ")
    root_only = _root_only_motion(m0, float(args.target_delta_deg))

    device = torch.device(args.device)
    model_path = args.model_path or default_smpl_model_path("smplx", ROOT)
    model = __import__("smplx").SMPLX(
        model_path=str(model_path), gender="neutral", num_betas=10,
        batch_size=int(m0.shape[0]), use_pca=False,
    ).to(device)
    motions = {"M0": m0, "root-only": root_only, "compensated": candidate}
    vertices: dict[str, np.ndarray] = {}
    joints: dict[str, torch.Tensor] = {}
    roots: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, motion in motions.items():
            params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
            params = {key: value[0] for key, value in params.items()}
            vertices[name] = model(**params, return_verts=True).vertices.detach().cpu().numpy()
            joints[name] = direct_joints_from_motion(motion.unsqueeze(0))[0].cpu()
            roots[name] = decode_rot6d_safe(motion[..., MOTION_LAYOUT.root_rotation]).float()

    camera_heading = estimate_motion_heading(joints["M0"])
    camera_r, camera_t = fixed_sagittal_side_camera(joints["M0"], motion_heading=camera_heading)
    camera_r_np, camera_t_np = camera_r.numpy(), camera_t.numpy()
    dose_root_only = pelvis_pitch_delta_deg(roots["M0"], roots["root-only"]).numpy()
    dose_candidate = pelvis_pitch_delta_deg(roots["M0"], roots["compensated"]).numpy()
    neck = _axis_change_deg(joints["M0"], joints["compensated"], "neck")
    head = _axis_change_deg(joints["M0"], joints["compensated"], "head")

    output_dir = args.output_dir or (args.run_root / "videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_path = output_dir / "sample94_walk_M0_root_only_compensated.mp4"
    slow_path = output_dir / "sample94_walk_M0_root_only_compensated_slow.mp4"
    foot_path = output_dir / "sample94_walk_foot_local.mp4"
    normal = _encoder(normal_path)
    foot = _encoder(foot_path)
    renderer = pyrender.OffscreenRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    panels_spec = (
        ("M0", "M0 baseline", [0.55, 0.58, 0.62, 1.0]),
        ("root-only", "root-only +10 deg", [0.95, 0.55, 0.12, 1.0]),
        ("compensated", "diagnostic compensation", [0.10, 0.55, 0.92, 1.0]),
    )
    try:
        for frame in range(m0.shape[0]):
            panels = []
            for name, label, colour in panels_spec:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=colour, metallicFactor=0.0, roughnessFactor=0.82,
                )
                panel = _render_panel(
                    vertices[name], np.asarray(model.faces, dtype=np.int32),
                    camera_r_np[frame], camera_t_np[frame], frame, renderer, material,
                )
                cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 54), (245, 245, 245), -1)
                cv2.putText(panel, label, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
                panels.append(panel)
            top = np.concatenate(panels, axis=1)
            plot = _plot(dose_root_only, dose_candidate, neck, head, float(args.target_delta_deg), frame)
            frame_image = np.concatenate((top, plot), axis=0)
            assert normal.stdin is not None and foot.stdin is not None
            normal.stdin.write(frame_image.tobytes())
            foot_crop = cv2.resize(top[PANEL_HEIGHT // 2:, :], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            foot.stdin.write(foot_crop.tobytes())
    finally:
        renderer.delete()
        for process in (normal, foot):
            if process.stdin is not None:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed while rendering walk diagnostic")
    _slow_copy(normal_path, slow_path)
    print(json.dumps({"normal": str(normal_path), "slow": str(slow_path), "foot": str(foot_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
