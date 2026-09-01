#!/usr/bin/env python3
"""Render M0/candidate mesh and foot-local audits for v3."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
import trimesh
import pyrender

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import pelvis_pitch_delta_deg
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion, direct_smpl_parameters
from motion_rep.smplx_utils import default_smpl_model_path
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
def estimate_motion_heading(joints: torch.Tensor, min_displacement: float = 1e-4) -> torch.Tensor:
    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[1] < 1:
        raise ValueError("joints must have shape [T,J,3]")
    displacement = joints[-1, 0, :2] - joints[0, 0, :2]
    norm = torch.linalg.vector_norm(displacement)
    if float(norm) < min_displacement:
        return torch.tensor([0.0, 1.0, 0.0], dtype=joints.dtype, device=joints.device)
    return torch.nn.functional.pad(displacement / norm, (0, 1))


def fixed_sagittal_side_camera(joints: torch.Tensor, *, motion_heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    roots = joints[:, 0]
    heading = motion_heading.to(device=joints.device, dtype=joints.dtype)
    heading = heading / torch.linalg.vector_norm(heading).clamp_min(1e-8)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=joints.dtype, device=joints.device)
    camera_x = -heading
    camera_z = torch.linalg.cross(camera_x, up)
    R_one = torch.stack((camera_x, up, camera_z), dim=1)
    root_min, root_max = roots.amin(dim=0), roots.amax(dim=0)
    root_move = root_max - root_min
    root_depth = torch.matmul(roots, camera_z)
    depth_offset = 2.5 - root_depth.amin() + 0.5 * torch.linalg.vector_norm(root_move)
    T_one = torch.zeros(3, dtype=joints.dtype, device=joints.device)
    root_center = 0.5 * (root_min + root_max)
    T_one[0] = torch.dot(root_center, heading)
    T_one[1] = -roots[0, 2]
    T_one[2] = depth_offset
    return R_one[None].repeat(joints.shape[0], 1, 1), T_one[None].repeat(joints.shape[0], 1)


def _render_panel(vertices: np.ndarray, faces: np.ndarray, camera_r: np.ndarray, camera_t: np.ndarray, frame: int, renderer: pyrender.OffscreenRenderer, material: pyrender.MetallicRoughnessMaterial) -> np.ndarray:
    """Render one frame using the already-selected camera transform.

    The caller passes a single frame's rotation/translation.  Keeping the
    frame index only for the mesh prevents accidentally indexing a 3x3 camera
    matrix as if it were a time sequence.
    """
    camera_points = vertices[frame] @ camera_r + camera_t
    mesh_points = np.stack((-camera_points[:, 0], camera_points[:, 1], -camera_points[:, 2]), axis=-1)
    tri = trimesh.Trimesh(vertices=mesh_points, faces=faces, process=False)
    scene = pyrender.Scene(bg_color=np.array([245, 245, 245, 255], dtype=np.uint8), ambient_light=np.array([0.35, 0.35, 0.35]))
    focal = 0.85 * max(PANEL_WIDTH, PANEL_HEIGHT)
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=PANEL_WIDTH / 2.0, cy=PANEL_HEIGHT / 2.0, znear=0.01, zfar=100.0)
    scene.add(camera, pose=np.eye(4, dtype=np.float32))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=np.eye(4, dtype=np.float32))
    light_pose = np.eye(4, dtype=np.float32)
    light_pose[:3, :3] = np.array([[0.82, 0.0, 0.57], [0.0, 1.0, 0.0], [-0.57, 0.0, 0.82]], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=light_pose)
    scene.add(pyrender.Mesh.from_trimesh(tri, material=material, smooth=False))
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
    return cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)


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
