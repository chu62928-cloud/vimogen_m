"""Render a v2 single-case M0/v2 comparison with shaded SMPL-X meshes.

This is a software/OpenGL-EGL fallback for hosts where the repository's
PyTorch3D rasterizer extension has no kernel for the installed GPU.  The
motion and camera remain the same as the v2 skeleton audit; only the mesh
renderer is replaced.
"""

from __future__ import annotations

import argparse
import os
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

from motion_rep.motion_checker import (  # noqa: E402
    _default_smpl_model_path,
    estimate_focal_length,
)
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from motion_rep.retarget_motion import motion_rep_to_SMPL  # noqa: E402
from scripts.render_absolute_mean_triptych import (  # noqa: E402
    estimate_motion_heading,
    fixed_sagittal_side_camera,
)
from scripts.render_relative_root_forward_v1_1 import (  # noqa: E402
    FPS,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    _draw_direction_overlay,
    _encoder,
    _load,
    _panel_vectors,
    _plot,
    _root_geometry,
    _target_forward,
)


def _load_motions(
    run_root: Path, sample_index: int, mean: torch.Tensor, std: torch.Tensor
) -> dict[str, torch.Tensor]:
    m0_path = next(run_root.glob("m0_artifacts/batch_*/m0_official_norm_batch.pt"))
    candidate_path = next(
        run_root.glob("trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt")
    )
    m0 = _load(m0_path)[sample_index] * std + mean
    archive = torch.load(candidate_path, map_location="cpu", weights_only=True)
    candidate = archive["motion_norm"].float()[sample_index]
    candidate = (
        candidate * archive["motion_std"].float()[0]
        + archive["motion_mean"].float()[0]
    )
    return {"M0": m0, "-": m0.clone(), "+": candidate}


def _render_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_r: np.ndarray,
    camera_t: np.ndarray,
    frame: int,
    renderer: pyrender.OffscreenRenderer,
    material: pyrender.MetallicRoughnessMaterial,
) -> np.ndarray:
    """Render one mesh frame in the same camera convention as the audit code."""

    # Existing audit projections use p_cam = p_world @ R + T and positive z
    # depth. Pyrender looks along negative z; keep the camera-up coordinate
    # positive so its screen y direction agrees with the audit projection.
    camera_points = vertices[frame] @ camera_r[frame] + camera_t[frame]
    mesh_points = np.stack(
        (-camera_points[:, 0], camera_points[:, 1], -camera_points[:, 2]), axis=-1
    )
    tri = trimesh.Trimesh(vertices=mesh_points, faces=faces, process=False)
    scene = pyrender.Scene(
        bg_color=np.array([245, 245, 245, 255], dtype=np.uint8),
        ambient_light=np.array([0.35, 0.35, 0.35]),
    )
    focal = estimate_focal_length(PANEL_WIDTH, PANEL_HEIGHT)
    camera = pyrender.IntrinsicsCamera(
        fx=float(focal),
        fy=float(focal),
        cx=PANEL_WIDTH / 2.0,
        cy=PANEL_HEIGHT / 2.0,
        znear=0.01,
        zfar=100.0,
    )
    scene.add(camera, pose=np.eye(4, dtype=np.float32))
    # Two camera-side lights make the silhouette and feet readable without
    # introducing a method-dependent lighting change.
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        pose=np.eye(4, dtype=np.float32),
    )
    light_pose = np.eye(4, dtype=np.float32)
    light_pose[:3, :3] = np.array(
        [[0.82, 0.0, 0.57], [0.0, 1.0, 0.0], [-0.57, 0.0, 0.82]],
        dtype=np.float32,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
        pose=light_pose,
    )
    scene.add(pyrender.Mesh.from_trimesh(tri, material=material, smooth=False))
    color, _ = renderer.render(
        scene,
        flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
    )
    return cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-label", default="sample94_v2_single")
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    mean = torch.from_numpy(np.load(args.mean)).float()
    std = torch.from_numpy(np.load(args.std)).float()
    motions = _load_motions(args.run_root, args.sample_index, mean, std)
    parameters, joints, roots = {}, {}, {}
    with torch.inference_mode():
        for method, motion in motions.items():
            parameters[method], joints[method] = motion_rep_to_SMPL(
                motion.to(args.device), recover_from_velocity=True
            )
            roots[method] = decode_rot6d_safe(
                motion[:, MOTION_LAYOUT.root_rotation]
            ).float().to(joints[method].device)
        model = __import__("smplx").SMPLX(
            model_path=_default_smpl_model_path("smplx"),
            gender="neutral",
            num_betas=10,
            batch_size=motions["M0"].shape[0],
            use_pca=False,
        ).to(args.device)
        vertices = {
            method: model(**parameters[method]).vertices.detach().cpu().numpy()
            for method in motions
        }
    faces = np.asarray(model.faces, dtype=np.int32)

    display_heading = estimate_motion_heading(joints["M0"])
    camera_r, camera_t = fixed_sagittal_side_camera(
        joints["M0"], motion_heading=display_heading
    )
    f0, _, r0, phi0 = _root_geometry(roots["M0"])
    vectors = {
        method: _panel_vectors(
            joints[method],
            roots[method],
            _target_forward(f0, r0, 0.0 if method != "+" else 10.0),
            camera_r,
            camera_t,
        )
        for method in motions
    }
    root_change = {
        method: (phi0 - _root_geometry(roots[method])[3]).detach().cpu().numpy()
        for method in motions
    }
    materials = {
        "M0": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.55, 0.58, 0.62, 1.0], metallicFactor=0.0, roughnessFactor=0.82
        ),
        "-": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.95, 0.52, 0.10, 1.0], metallicFactor=0.0, roughnessFactor=0.82
        ),
        "+": pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.10, 0.55, 0.92, 1.0], metallicFactor=0.0, roughnessFactor=0.82
        ),
    }
    camera_r_np = camera_r.detach().cpu().numpy()
    camera_t_np = camera_t.detach().cpu().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.sample_label}_mesh_M0_+10.mp4"
    process = _encoder(output)
    renderer = pyrender.OffscreenRenderer(PANEL_WIDTH, PANEL_HEIGHT)
    try:
        frame_count = joints["M0"].shape[0]
        for frame in range(frame_count):
            panels = []
            for method in ("M0", "-", "+"):
                target_delta = 0.0 if method != "+" else 10.0
                label = "M0" if method != "+" else "v2 +10 deg"
                panel = _render_panel(
                    vertices[method], faces, camera_r_np, camera_t_np,
                    frame, renderer, materials[method],
                )
                _draw_direction_overlay(
                    panel, frame, vectors[method], label=label,
                    root_change_deg=float(root_change[method][frame]),
                    target_delta_deg=target_delta,
                )
                panels.append(panel)
            curves = {
                "M0": root_change["M0"],
                "-": root_change["-"],
                "+": root_change["+"],
            }
            bottom = _plot(curves, frame, (0.0, 10.0))
            process.stdin.write(
                np.concatenate((np.concatenate(panels, axis=1), bottom), axis=0).tobytes()
            )
    finally:
        renderer.delete()
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while rendering {output}")
    print(str(output))


if __name__ == "__main__":
    main()
