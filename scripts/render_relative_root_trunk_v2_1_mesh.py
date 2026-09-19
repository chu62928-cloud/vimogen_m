#!/usr/bin/env python3
"""Render the frozen M0, pure-v2, and v2.1 motions with one mesh camera."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
import pyrender

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion, direct_smpl_parameters, root_trunk_relative_angle_deg
from motion_rep.motion_checker import _default_smpl_model_path
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from motion_rep.pose_authority import _root_forward, authority_project
from scripts.render_relative_root_forward_v2_single_mesh import (
    _draw_direction_overlay,
    _encoder,
    _plot,
    _render_panel,
)
from scripts.render_relative_root_forward_v1_1 import _panel_vectors, _root_geometry, _target_forward
from scripts.render_absolute_mean_triptych import estimate_motion_heading, fixed_sagittal_side_camera


PANEL_WIDTH, PANEL_HEIGHT, PLOT_HEIGHT = 640, 720, 360
FPS = 20


def _archive_motion(run_root: Path, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m0_path = next(run_root.glob("m0_artifacts/batch_*/m0_official_norm_batch.pt"))
    archive_path = next(run_root.glob("trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt"))
    m0 = torch.load(m0_path, map_location="cpu", weights_only=True).float() * std.view(1, 1, -1) + mean.view(1, 1, -1)
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    candidate = archive["motion_norm"].float() * archive["motion_std"].float()[:, None, :] + archive["motion_mean"].float()[:, None, :]
    mask = archive["motion_mask"].bool()
    return authority_project(m0, valid_mask=mask).physical_motion[0], authority_project(candidate, valid_mask=mask).physical_motion[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-run-root", type=Path, required=True)
    parser.add_argument("--v2-1-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()
    m0, v2 = _archive_motion(args.v2_run_root, mean, std)
    m0_v21, v21 = _archive_motion(args.v2_1_run_root, mean, std)
    if m0.shape != m0_v21.shape or not torch.allclose(m0, m0_v21, atol=1e-5, rtol=0.0):
        raise ValueError("v2 and v2.1 do not share the same canonical M0")
    motions = {"M0": m0, "v2": v2, "v2.1": v21}
    model = __import__("smplx").SMPLX(model_path=_default_smpl_model_path("smplx"), gender="neutral", num_betas=10, batch_size=m0.shape[0], use_pca=False).to(args.device)
    parameters, vertices, joints, roots = {}, {}, {}, {}
    with torch.inference_mode():
        for name, motion in motions.items():
            parameters[name] = direct_smpl_parameters(motion.unsqueeze(0).to(args.device))
            parameters[name] = {key: value[0] for key, value in parameters[name].items()}
            vertices[name] = model(**parameters[name]).vertices.detach().cpu().numpy()
            joints[name] = direct_joints_from_motion(motion.unsqueeze(0))[0].cpu()
            roots[name] = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation]).float()
    faces = np.asarray(model.faces, dtype=np.int32)
    camera_heading = estimate_motion_heading(joints["M0"])
    camera_r, camera_t = fixed_sagittal_side_camera(joints["M0"], motion_heading=camera_heading)
    f0, _, r0, phi0 = _root_geometry(roots["M0"])
    vectors = {name: _panel_vectors(joints[name], roots[name], _target_forward(f0, r0, 10.0 if name != "M0" else 0.0), camera_r, camera_t) for name in motions}
    root_change = {name: (phi0 - _root_geometry(roots[name])[3]).cpu().numpy() for name in motions}
    heading = _root_forward(roots["M0"])[1]
    relative = {name: root_trunk_relative_angle_deg(roots[name].unsqueeze(0), joints[name].unsqueeze(0), m0_heading=heading.unsqueeze(0))[0].numpy() for name in motions}
    materials = {
        "M0": pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.55, 0.58, 0.62, 1.0], metallicFactor=0.0, roughnessFactor=0.82),
        "v2": pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.95, 0.52, 0.10, 1.0], metallicFactor=0.0, roughnessFactor=0.82),
        "v2.1": pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.10, 0.55, 0.92, 1.0], metallicFactor=0.0, roughnessFactor=0.82),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "sample94_v2_1_mesh_M0_v2_v2_1.mp4"
    foot_output = args.output_dir / "sample94_v2_1_mesh_left_foot_local.mp4"
    process = _encoder(output)
    foot_process = _encoder(foot_output)
    renderer = pyrender.OffscreenRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    camera_r_np, camera_t_np = camera_r.numpy(), camera_t.numpy()
    try:
        for frame in range(m0.shape[0]):
            panels = []
            for name in ("M0", "v2", "v2.1"):
                label = {"M0": "M0", "v2": "pure v2 +10 deg", "v2.1": "root-trunk v2.1 +10 deg"}[name]
                panel = _render_panel(vertices[name], faces, camera_r_np, camera_t_np, frame, renderer, materials[name])
                _draw_direction_overlay(panel, frame, vectors[name], label=label, root_change_deg=float(root_change[name][frame]), target_delta_deg=0.0 if name == "M0" else 10.0)
                panels.append(panel)
            curves = {"M0": relative["M0"], "-": relative["v2"], "+": relative["v2.1"]}
            bottom = _plot(curves, frame, (-100.0, -60.0))
            process.stdin.write(np.concatenate((np.concatenate(panels, axis=1), bottom), axis=0).tobytes())
            foot = np.concatenate([panel[PANEL_HEIGHT // 2 :, :] for panel in panels], axis=1)
            foot_process.stdin.write(cv2.resize(foot, (PANEL_WIDTH * 3, 1080), interpolation=cv2.INTER_CUBIC).tobytes())
    finally:
        renderer.delete()
        for item in (process, foot_process):
            if item.stdin is not None:
                item.stdin.close()
            if item.wait() != 0:
                raise RuntimeError("ffmpeg failed while rendering v2.1 video")
    print(json.dumps({"grid": str(output), "left_foot": str(foot_output)}, ensure_ascii=False))


if __name__ == "__main__":
    import json
    main()
