#!/usr/bin/env python3
"""Render clean and diagnostic fixed-side M0/candidate videos for one S1 record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(path: Path) -> dict:
    value = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(value.stdout)["streams"][0]
    if stream.get("codec_name") != "h264":
        raise RuntimeError(f"{path} is not H.264")
    return stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--m0-motion", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.record = args.record.resolve()
    args.m0_motion = args.m0_motion.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    candidate_path = args.record.parent / "motion_physical.pt"
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True).float()
    baseline = torch.load(args.m0_motion, map_location="cpu", weights_only=True).float()
    if candidate.shape != baseline.shape or candidate.ndim != 3 or candidate.shape[0] != 1:
        raise ValueError("candidate and M0 must share shape [1,T,276]")
    from guidance.base import tensor_sha256
    if tensor_sha256(baseline) != record["source_m0_sha256"]:
        raise ValueError("M0 tensor hash does not match the run record")

    runtime = args.runtime_root.resolve()
    # The checkout supplies the frozen side-camera implementation.  The
    # runtime root only fills in historical modules/assets absent here.
    if str(runtime) not in sys.path:
        sys.path.append(str(runtime))
    os.chdir(runtime)
    renderer_path = ROOT / "scripts/render_absolute_mean_triptych.py"
    spec = importlib.util.spec_from_file_location("local_pelvis_side_renderer", renderer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load checkout renderer {renderer_path}")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    render_sources = renderer.render_sources

    args.output.mkdir(parents=True)
    rendered = render_sources(
        {"M0": baseline[0], "candidate": candidate[0]},
        args.output / "sources",
        args.device,
        "mesh",
        fps=20.0,
        annotate=False,
        mesh_backend="pyrender_egl",
    )
    clean = args.output / "clean_candidate.mp4"
    shutil.copy2(rendered["candidate"], clean)
    diagnostic = args.output / "diagnostic_m0_candidate.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(rendered["M0"]), "-i", str(rendered["candidate"]),
         "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "20", str(diagnostic)],
        check=True, capture_output=True, text=True,
    )
    summary = {
        "protocol": "vimogen_local_pelvis_s1_side_pair_v1",
        "status": "RENDERED_PENDING_USER_VISUAL_REVIEW",
        "run_id": record["run_id"],
        "source_m0_sha256": record["source_m0_sha256"],
        "candidate_sha256": _sha256(candidate_path),
        "shared_m0_camera": True,
        "clean_video": str(clean),
        "diagnostic_video": str(diagnostic),
        "clean_video_sha256": _sha256(clean),
        "diagnostic_video_sha256": _sha256(diagnostic),
        "clean_probe": _probe(clean),
        "diagnostic_probe": _probe(diagnostic),
        "visual_review_status": "PENDING_USER_FINAL_REVIEW",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
