#!/usr/bin/env python3
"""Run one server-side minimal source-noise root-forward v2 experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_ROOT = ROOT / "results/phase7/relative_root_forward_v2"
DEFAULT_NOISE_CACHE = ROOT / "results/phase6/absolute_mean_pelvis_v2/noise_cache"
PROTOCOL_NAME = "vimogen_relative_root_forward_v2_minimal_source_noise"


def _manifest(run_root: Path) -> Path:
    source = ROOT / "results/phase7/relative_root_forward_v1/data/smoke_sample94_34122.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    matches = [row for row in rows if str(row.get("global_id")) == "94"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one sample94 row, found {len(matches)}")
    path = run_root / "sample94_manifest.json"
    path.write_text(json.dumps(matches, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _next_attempt(seed: int, target_delta_deg: float) -> Path:
    parent = PROTOCOL_ROOT / "minimal_source_noise" / f"seed_{seed:03d}" / f"delta_{target_delta_deg:+g}deg"
    attempt = 1
    path = parent / f"attempt_{attempt:02d}"
    while path.exists():
        attempt += 1
        path = parent / f"attempt_{attempt:02d}"
    path.mkdir(parents=True)
    return path


def build_config(args, run_root: Path, manifest: Path):
    config = OmegaConf.load(ROOT / "configs/tm2m_infer.yaml")
    config.mode = "eval"
    config.mbench_name = f"relative_root_forward_v2_minimal_delta{args.target_delta_deg:+g}_seed{args.seed}"
    config.experiment.global_seed = int(args.seed)
    config.experiment.auto_resume = False
    config.experiment.eval_steps = 1
    config.experiment.validation_steps = 50
    config.experiment.result_dir = str(run_root / "trainer")
    config.dataloader.test_local_batch = 1
    config.dataloader.num_workers = 2
    config.dataset.test_json_file_list = [str(manifest)]
    config.dataset.text_key = "prompt_motion_detailed"
    config.save_motion_visualizations = False
    config.m0 = {
        "noise_protocol": "sample_v1",
        "sample_noise_cache_dir": str(args.noise_cache),
        "artifact_dir": str(run_root / "m0_artifacts"),
        "initial_noise_path": None,
        "batch_invariant": True,
    }
    config.m1 = {"enabled": False}
    config.absolute_mean_pelvis = {"enabled": False}
    config.relative_root_forward = {
        "enabled": True,
        "protocol": PROTOCOL_NAME,
        "target_delta_deg": float(args.target_delta_deg),
        "artifact_dir": str(run_root / "source_noise_artifacts"),
        "iterations": int(args.iterations),
        "step_rms": float(args.step_rms),
        "max_delta_rms": float(args.max_delta_rms),
        "line_search_steps": int(args.line_search_steps),
        "feasible_pitch_mae_deg": float(args.feasible_pitch_mae_deg),
        "feasible_forward_p95_deg": float(args.feasible_forward_p95_deg),
        "forward_loss_temperature": float(args.forward_loss_temperature),
        "max_runtime_seconds": float(args.max_runtime_seconds),
        "use_gradient_checkpointing": True,
    }
    config.representation = {"reconciliation": {"enabled": False}}
    return config


def run(args) -> dict:
    if not args.noise_cache.is_dir():
        raise FileNotFoundError(f"sample-noise cache is missing: {args.noise_cache}")
    run_root = _next_attempt(args.seed, args.target_delta_deg)
    manifest = _manifest(run_root)
    config = build_config(args, run_root, manifest)
    OmegaConf.save(config, run_root / "resolved_config.yaml")
    record = {
        "status": "RUNNING",
        "protocol": PROTOCOL_NAME,
        "run_root": str(run_root),
        "seed": int(args.seed),
        "target_delta_deg": float(args.target_delta_deg),
        "iterations": int(args.iterations),
        "max_runtime_seconds": float(args.max_runtime_seconds),
    }
    record_path = run_root / "run_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29528")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        os.environ.setdefault("GROUP_RANK", "0")
        os.environ.setdefault("GROUP_WORLD_SIZE", "1")
        from train_eval_vimogen import main as train_eval_main

        train_eval_main(config)
        summaries = list(
            (run_root / "source_noise_artifacts").glob(
                "batch_*/text/guidance_summary.json"
            )
        )
        if len(summaries) != 1:
            raise RuntimeError(f"expected one source-noise summary, found {len(summaries)}")
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        record["summary_path"] = str(summaries[0])
        record["optimization_status"] = summary.get("status")
        record["feasible"] = bool(summary.get("feasible", False))
        record["status"] = "COMPLETED"
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--step-rms", type=float, default=0.01)
    parser.add_argument("--max-delta-rms", type=float, default=1.0)
    parser.add_argument("--line-search-steps", type=int, default=8)
    parser.add_argument("--feasible-pitch-mae-deg", type=float, default=1.0)
    parser.add_argument("--feasible-forward-p95-deg", type=float, default=2.0)
    parser.add_argument("--forward-loss-temperature", type=float, default=5.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--noise-cache", type=Path, default=DEFAULT_NOISE_CACHE)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
