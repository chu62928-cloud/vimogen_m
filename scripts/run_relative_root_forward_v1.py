#!/usr/bin/env python3
"""Run the server-side smoke matrix for relative root-forward v1.

The runner is deliberately non-overwriting.  It freezes a two-item smoke
manifest (MBench sample 94 and turning sample 34122), reuses the existing
sample-noise-v1 cache when available, and writes every run below the new
phase-7 protocol directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_NAME = "vimogen_relative_root_forward_v1_pose_authoritative"
PROTOCOL_ROOT = ROOT / "results/phase7/relative_root_forward_v1"
SMOKE_MANIFEST = PROTOCOL_ROOT / "data/smoke_sample94_34122.json"
PROTOCOL_FILE = PROTOCOL_ROOT / "protocol.json"
DEFAULT_NOISE_CACHE = ROOT / "results/phase6/absolute_mean_pelvis_v2/noise_cache"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_one(rows: list[dict], predicate) -> dict:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one smoke row, found {len(matches)}")
    return dict(matches[0])


def ensure_smoke_manifest() -> Path:
    """Freeze the exact two real-smoke prompts without touching old manifests."""

    if SMOKE_MANIFEST.is_file():
        return SMOKE_MANIFEST
    mbench_path = ROOT / "data/meta_info/MBench_final.json"
    dev_path = ROOT / "results/phase6/absolute_mean_pelvis_v2/data/development20.json"
    mbench = json.loads(mbench_path.read_text(encoding="utf-8"))
    development = json.loads(dev_path.read_text(encoding="utf-8"))
    sample94 = _find_one(
        mbench,
        lambda row: str(row.get("global_id")) == "94"
        and row.get("dimension") == "Motion_Quality",
    )
    sample34122 = _find_one(
        development, lambda row: str(row.get("sample_id")) == "34122"
    )
    # The MBench loader always creates a 100-frame text-to-motion sequence and
    # only needs the prompt embedding.  Reuse the frozen prompt embedding from
    # the development manifest for the turning sample.
    sample34122["global_id"] = "34122"
    sample34122["sample_id"] = "34122"
    sample34122["prompt_motion_detailed_wanvideot5_embed_path"] = sample34122.pop(
        "prompt_wanvideot5_embed_path"
    )
    sample34122["use_ref_motion"] = False
    rows = [sample94, sample34122]
    SMOKE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_MANIFEST.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return SMOKE_MANIFEST


def ensure_protocol(manifest: Path) -> Path:
    if PROTOCOL_FILE.is_file():
        return PROTOCOL_FILE
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol": PROTOCOL_NAME,
        "authority": ["body_pose", "root_rotation", "root_translation"],
        "derived": ["J", "dJ", "dR", "dT"],
        "guidance": "per-frame scalar left root rotation about frozen M0 right axis",
        "target_delta_deg_range": [-10.0, 10.0],
        "downward_positive": True,
        "m0_projection": "once_and_frozen",
        "smoke_manifest": str(manifest),
        "smoke_manifest_sha256": sha256_file(manifest),
        "consistency_thresholds": {
            "J_FK_m": 1e-5,
            "dJ_m": 1e-6,
            "dR_deg": 1e-4,
            "dT_m": 1e-6,
        },
    }
    PROTOCOL_FILE.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return PROTOCOL_FILE


def build_config(
    *,
    target_delta_deg: float,
    seed: int,
    run_root: Path,
    noise_cache: Path,
    manifest: Path,
    batch_size: int,
):
    if not -10.0 <= float(target_delta_deg) <= 10.0:
        raise ValueError("target_delta_deg must lie in [-10,10]")
    config = OmegaConf.load(ROOT / "configs/tm2m_infer.yaml")
    config.mode = "eval"
    config.mbench_name = (
        f"relative_root_forward_v1_delta{target_delta_deg:g}_seed{seed}"
    )
    config.experiment.global_seed = int(seed)
    config.experiment.auto_resume = False
    config.experiment.eval_steps = 1
    config.experiment.result_dir = str(run_root / "trainer")
    config.dataloader.test_local_batch = int(batch_size)
    config.dataloader.num_workers = 4
    config.dataset.test_json_file_list = [str(manifest)]
    config.dataset.text_key = "prompt_motion_detailed"
    config.save_motion_visualizations = False
    config.m0 = {
        "noise_protocol": "sample_v1",
        "sample_noise_cache_dir": str(noise_cache),
        "artifact_dir": str(run_root / "m0_artifacts"),
        "initial_noise_path": None,
        "batch_invariant": True,
    }
    config.m1 = {"enabled": False}
    config.absolute_mean_pelvis = {"enabled": False}
    config.relative_root_forward = {
        "enabled": True,
        "protocol": PROTOCOL_NAME,
        "target_delta_deg": float(target_delta_deg),
        "guidance_strength": 1.0,
        "sigma_min": 0.25,
        "sigma_max": 0.65,
        "motion_weight": 0.1,
        "base_step_deg": 1.0,
        "max_correction_rms": 0.05,
        "max_backtracks": 11,
        "trace_enabled": True,
        "artifact_dir": str(run_root / "guided_artifacts"),
    }
    config.representation = {"reconciliation": {"enabled": False}}
    return config


def run(args) -> dict:
    manifest = ensure_smoke_manifest()
    protocol = ensure_protocol(manifest)
    noise_cache = args.noise_cache
    if not noise_cache.is_dir():
        raise FileNotFoundError(f"sample-noise cache is missing: {noise_cache}")
    run_parent = (
        PROTOCOL_ROOT
        / "runs"
        / "smoke"
        / f"seed_{args.seed:03d}"
        / f"delta_{args.target_delta_deg:+g}deg"
    )
    attempt = 1
    run_base = run_parent / f"attempt_{attempt:02d}"
    while run_base.exists():
        attempt += 1
        run_base = run_parent / f"attempt_{attempt:02d}"
    run_base.mkdir(parents=True)
    config = build_config(
        target_delta_deg=args.target_delta_deg,
        seed=args.seed,
        run_root=run_base,
        noise_cache=noise_cache,
        manifest=manifest,
        batch_size=args.batch_size,
    )
    record = {
        "status": "RUNNING",
        "protocol": PROTOCOL_NAME,
        "protocol_path": str(protocol),
        "protocol_sha256": sha256_file(protocol),
        "input_manifest": str(manifest),
        "input_manifest_sha256": sha256_file(manifest),
        "target_delta_deg": float(args.target_delta_deg),
        "seed": int(args.seed),
        "noise_cache": str(noise_cache),
        "run_root": str(run_base),
        "consistency_boundary": "authoritative_pose_to_fk_to_J_to_dJ_dR_dT_to_276D",
    }
    (run_base / "run_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OmegaConf.save(config, run_base / "resolved_config.yaml")
    started = time.perf_counter()
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        os.environ.setdefault("GROUP_RANK", "0")
        os.environ.setdefault("GROUP_WORLD_SIZE", "1")
        from train_eval_vimogen import main as train_eval_main

        train_eval_main(config)
        record["status"] = "COMPLETED_GENERATION_PENDING_EVALUATION"
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        (run_base / "run_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-delta-deg", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-cache", type=Path, default=DEFAULT_NOISE_CACHE)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
