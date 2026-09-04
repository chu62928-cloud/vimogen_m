#!/usr/bin/env python3
"""Render M0, root-only, v1.3 and v2 sample94 candidates side by side."""

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

from evaluation.pelvis_contact_compensation_v3 import pelvis_pitch_delta_deg
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion, direct_smpl_parameters
from motion_rep.phase1 import MOTION_LAYOUT, SMPLX_22_JOINT_INDEX, decode_rot6d_safe
from scripts.render_pelvis_contact_v3_mesh import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    _render_panel,
    estimate_motion_heading,
    fixed_sagittal_side_camera,
)


WIDTH, HEIGHT, FPS = 1920, 1080, 20
DISPLAY_PANEL = 480
PLOT_HEIGHT = HEIGHT - DISPLAY_PANEL


def _load(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True).float()
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2 or value.shape[-1] != 276:
        raise ValueError(f"expected [T,276] at {path}")
    return value


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


def _slow_copy(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", "setpts=2.0*PTS", "-an", "-r", str(FPS),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ],
        check=True,
    )


def _axis_change(m0_joints: torch.Tensor, candidate_joints: torch.Tensor) -> np.ndarray:
    spine1 = SMPLX_22_JOINT_INDEX["spine1"]
    neck = SMPLX_22_JOINT_INDEX["neck"]
    base = torch.nn.functional.normalize(m0_joints[:, neck] - m0_joints[:, spine1], dim=-1)
    candidate = torch.nn.functional.normalize(candidate_joints[:, neck] - candidate_joints[:, spine1], dim=-1)
    return (torch.acos((base * candidate).sum(-1).clamp(-1.0, 1.0)) * 180.0 / math.pi).numpy()


def _curve_points(curve: np.ndarray, frame: int, high: float, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    points = []
    for index in range(frame + 1):
        x = left + int(index * (right - left) / max(len(curve) - 1, 1))
        y = bottom - int(float(curve[index]) * (bottom - top) / max(high, 1.0e-6))
        points.append((x, y))
    return np.asarray(points, dtype=np.int32)


def _plot(doses: dict[str, np.ndarray], trunks: dict[str, np.ndarray], frame: int) -> np.ndarray:
    image = np.full((PLOT_HEIGHT, WIDTH, 3), 248, dtype=np.uint8)
    colours = {
        "root-only": (0, 145, 245),
        "v1.3": (45, 155, 55),
        "v2": (220, 95, 45),
    }
    dose_box = (95, 70, WIDTH - 50, 245)
    trunk_box = (95, 335, WIDTH - 50, PLOT_HEIGHT - 55)
    dose_high = max(12.0, max(float(np.max(value)) for value in doses.values()) + 1.0)
    trunk_high = max(12.0, max(float(np.max(value)) for value in trunks.values()) + 1.0)
    cv2.putText(image, "pelvis dose relative to each archived M0 (deg)", (95, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(image, "trunk direction change relative to each archived M0 (deg)", (95, 307), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
    for box in (dose_box, trunk_box):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (70, 70, 70), 1)
    for index, name in enumerate(("root-only", "v1.3", "v2")):
        colour = colours[name]
        dose_points = _curve_points(doses[name], frame, dose_high, dose_box)
        trunk_points = _curve_points(trunks[name], frame, trunk_high, trunk_box)
        if len(dose_points) > 1:
            cv2.polylines(image, [dose_points], False, colour, 3, cv2.LINE_AA)
            cv2.polylines(image, [trunk_points], False, colour, 3, cv2.LINE_AA)
        x = 720 + index * 300
        cv2.line(image, (x, 30), (x + 35, 30), colour, 5)
        cv2.putText(image, name, (x + 45, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2, cv2.LINE_AA)
    current = "   ".join(
        f"{name}: dose {doses[name][frame]:.2f}, trunk {trunks[name][frame]:.2f}"
        for name in ("root-only", "v1.3", "v2")
    )
    cv2.putText(image, f"frame {frame + 1}/{len(next(iter(doses.values())))}   {current}", (95, PLOT_HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 2, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    motions = {
        "M0": _load(args.comparison_dir / "M0_current_physical.pt"),
        "root-only": _load(args.comparison_dir / "root_only_physical.pt"),
        "v1.3": _load(args.comparison_dir / "v1_3_guided_physical.pt"),
        "v2": _load(args.comparison_dir / "v2_source_noise_physical.pt"),
    }
    baselines = {
        "root-only": motions["M0"],
        "v1.3": _load(args.comparison_dir / "M0_v1_3_physical.pt"),
        "v2": _load(args.comparison_dir / "M0_v2_physical.pt"),
    }
    if len({motion.shape for motion in motions.values()}) != 1:
        raise ValueError("all comparison motions must have equal shape")
    device = torch.device(args.device)
    model = __import__("smplx").SMPLX(
        model_path=str(args.model_path), gender="neutral", num_betas=10,
        batch_size=int(motions["M0"].shape[0]), use_pca=False,
    ).to(device)
    vertices: dict[str, np.ndarray] = {}
    joints: dict[str, torch.Tensor] = {}
    roots: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, motion in {**motions, **{f"baseline-{key}": value for key, value in baselines.items()}}.items():
            params = direct_smpl_parameters(motion.unsqueeze(0).to(device))
            params = {key: value[0] for key, value in params.items()}
            if name in motions:
                vertices[name] = model(**params, return_verts=True).vertices.detach().cpu().numpy()
            joints[name] = direct_joints_from_motion(motion.unsqueeze(0))[0].cpu()
            roots[name] = decode_rot6d_safe(motion[..., MOTION_LAYOUT.root_rotation]).float()

    doses = {
        name: pelvis_pitch_delta_deg(roots[f"baseline-{name}"], roots[name]).numpy()
        for name in ("root-only", "v1.3", "v2")
    }
    trunks = {
        name: _axis_change(joints[f"baseline-{name}"], joints[name])
        for name in ("root-only", "v1.3", "v2")
    }
    heading = estimate_motion_heading(joints["M0"])
    camera_r, camera_t = fixed_sagittal_side_camera(joints["M0"], motion_heading=heading)
    camera_r_np, camera_t_np = camera_r.numpy(), camera_t.numpy()
    output_dir = args.output_dir or (args.comparison_dir / "videos")
    normal_path = output_dir / "sample94_M0_root_only_v1_3_v2.mp4"
    slow_path = output_dir / "sample94_M0_root_only_v1_3_v2_slow.mp4"
    encoder = _encoder(normal_path)
    renderer = pyrender.OffscreenRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    specs = (
        ("M0", "M0 (current / v1.3)", [0.55, 0.58, 0.62, 1.0]),
        ("root-only", "direct root-only +10 deg", [0.95, 0.55, 0.12, 1.0]),
        ("v1.3", "v1.3 guided +10 deg", [0.20, 0.68, 0.28, 1.0]),
        ("v2", "v2 source-noise +10 deg", [0.18, 0.48, 0.90, 1.0]),
    )
    try:
        for frame in range(motions["M0"].shape[0]):
            panels = []
            for name, label, colour in specs:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=colour, metallicFactor=0.0, roughnessFactor=0.82,
                )
                panel = _render_panel(
                    vertices[name], np.asarray(model.faces, dtype=np.int32),
                    camera_r_np[frame], camera_t_np[frame], frame, renderer, material,
                )
                panel = cv2.resize(panel, (DISPLAY_PANEL, DISPLAY_PANEL), interpolation=cv2.INTER_AREA)
                cv2.rectangle(panel, (0, 0), (DISPLAY_PANEL, 46), (245, 245, 245), -1)
                cv2.putText(panel, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 2, cv2.LINE_AA)
                panels.append(panel)
            image = np.concatenate((np.concatenate(panels, axis=1), _plot(doses, trunks, frame)), axis=0)
            assert encoder.stdin is not None
            encoder.stdin.write(image.tobytes())
    finally:
        renderer.delete()
        if encoder.stdin is not None:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    _slow_copy(normal_path, slow_path)
    print(json.dumps({"normal": str(normal_path), "slow": str(slow_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
