#!/usr/bin/env python3
"""Run the differentiable 50-step source-noise reproduction stop gate."""

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


def _sample94_manifest(run_root: Path) -> Path:
    source = ROOT / "results/phase7/relative_root_forward_v1/data/smoke_sample94_34122.json"
    if not source.is_file():
        raise FileNotFoundError(f"frozen v1 smoke manifest is missing: {source}")
    rows = json.loads(source.read_text(encoding="utf-8"))
    matches = [row for row in rows if str(row.get("global_id")) == "94"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one sample94 row, found {len(matches)}")
    path = run_root / "sample94_gate_manifest.json"
    path.write_text(
        json.dumps(matches, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _next_attempt(seed: int) -> Path:
    parent = PROTOCOL_ROOT / "gates/differentiable_50step" / f"seed_{seed:03d}"
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
    config.mbench_name = f"relative_root_forward_v2_stop_gate_seed{args.seed}"
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
    config.relative_root_forward = {"enabled": False}
    config.representation = {"reconciliation": {"enabled": False}}
    config.source_noise_gate = {
        "enabled": True,
        "artifact_dir": str(run_root / "gate_artifacts"),
        "target_delta_deg": float(args.target_delta_deg),
        "use_gradient_checkpointing": True,
        "max_reserved_mib": float(args.max_reserved_mib),
    }
    if args.probe:
        config.source_noise_probe = {
            "enabled": True,
            "artifact_dir": str(run_root / "subspace_probe_artifacts"),
            "historical_delta_path": str(args.historical_delta_path) if args.historical_delta_path else None,
            "direction_seed": int(args.direction_seed),
            "rms_values": [0.005, 0.01],
            "use_gradient_checkpointing": True,
        }
    return config


def run(args) -> dict:
    if not args.noise_cache.is_dir():
        raise FileNotFoundError(f"sample-noise cache is missing: {args.noise_cache}")
    run_root = _next_attempt(args.seed)
    manifest = _sample94_manifest(run_root)
    config = build_config(args, run_root, manifest)
    OmegaConf.save(config, run_root / "resolved_config.yaml")
    record = {
        "status": "RUNNING",
        "run_root": str(run_root),
        "seed": int(args.seed),
        "target_delta_deg": float(args.target_delta_deg),
        "max_reserved_mib": float(args.max_reserved_mib),
    }
    record_path = run_root / "run_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29527")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        os.environ.setdefault("GROUP_RANK", "0")
        os.environ.setdefault("GROUP_WORLD_SIZE", "1")
        from train_eval_vimogen import main as train_eval_main

        train_eval_main(config)
        gate_paths = list(
            (run_root / "gate_artifacts").glob(
                "batch_*/text/differentiable_50step_gate.json"
            )
        )
        if len(gate_paths) != 1:
            raise RuntimeError(f"expected one gate result, found {len(gate_paths)}")
        gate = json.loads(gate_paths[0].read_text(encoding="utf-8"))
        record["gate_result"] = str(gate_paths[0])
        record["gate_passed"] = bool(gate.get("passed", False))
        record["status"] = "PASSED" if record["gate_passed"] else "STOPPED"
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
    parser.add_argument("--max-reserved-mib", type=float, default=28672.0)
    parser.add_argument("--noise-cache", type=Path, default=DEFAULT_NOISE_CACHE)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--historical-delta-path", type=Path, default=None)
    parser.add_argument("--direction-seed", type=int, default=314159)
    args = parser.parse_args()
    record = run(args)
    print(json.dumps(record, indent=2))
    if not record.get("gate_passed", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
