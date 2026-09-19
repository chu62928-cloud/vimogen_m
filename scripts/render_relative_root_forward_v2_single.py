"""Render the completed v2 single-case result for visual inspection.

The shared v1.1 renderer expects three panels.  The middle panel is therefore
an explicit M0 copy; the right panel is the v2 +10 degree candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_absolute_mean_triptych import (  # noqa: E402
    estimate_motion_heading,
    fixed_sagittal_side_camera,
)
from scripts.render_relative_root_forward_v1_1 import (  # noqa: E402
    FPS,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    _encoder,
    _load,
    _panel_vectors,
    _plot,
    _root_geometry,
    _skeleton_panel,
    _target_forward,
)
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe  # noqa: E402
from motion_rep.retarget_motion import motion_rep_to_SMPL  # noqa: E402


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
    m0_path = next(args.run_root.glob("m0_artifacts/batch_*/m0_official_norm_batch.pt"))
    candidate_path = next(args.run_root.glob("trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt"))
    m0 = _load(m0_path)[args.sample_index] * std + mean
    archive = torch.load(candidate_path, map_location="cpu", weights_only=True)
    candidate = archive["motion_norm"].float()[args.sample_index]
    candidate = candidate * archive["motion_std"].float()[0] + archive["motion_mean"].float()[0]
    motions = {"M0": m0, "-": m0.clone(), "+": candidate}
    joints = {}
    roots = {}
    for method, motion in motions.items():
        _, joints[method] = motion_rep_to_SMPL(
            motion.to(args.device), recover_from_velocity=True
        )
        roots[method] = decode_rot6d_safe(
            motion[:, MOTION_LAYOUT.root_rotation]
        ).float().to(joints[method].device)
    display_heading = estimate_motion_heading(joints["M0"])
    camera_r, camera_t = fixed_sagittal_side_camera(
        joints["M0"], motion_heading=display_heading
    )
    f0, _, r0, phi0 = _root_geometry(roots["M0"])
    vectors = {
        method: _panel_vectors(
            joints[method], roots[method],
            _target_forward(f0, r0, 0.0 if method != "+" else 10.0),
            camera_r, camera_t,
        )
        for method in motions
    }
    root_change = {
        method: (phi0 - _root_geometry(roots[method])[3]).detach().cpu().numpy()
        for method in motions
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.sample_label}_skeleton_M0_+10.mp4"
    process = _encoder(output)
    try:
        for frame in range(joints["M0"].shape[0]):
            panels = []
            for method in ("M0", "-", "+"):
                label = "M0" if method != "+" else "v2 +10 deg"
                panels.append(
                    _skeleton_panel(
                        joints[method].float().cpu(), vectors[method], frame,
                        label=label,
                        root_change_deg=float(root_change[method][frame]),
                        target_delta_deg=0.0 if method != "+" else 10.0,
                        camera_r=camera_r.cpu(), camera_t=camera_t.cpu(),
                    )
                )
            curves = {"M0": root_change["M0"], "-": root_change["-"], "+": root_change["+"]}
            bottom = _plot(curves, frame, (0.0, 10.0))
            process.stdin.write(
                np.concatenate((np.concatenate(panels, axis=1), bottom), axis=0).tobytes()
            )
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while rendering {output}")
    print(str(output))


if __name__ == "__main__":
    main()
