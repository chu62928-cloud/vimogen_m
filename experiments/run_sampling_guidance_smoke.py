#!/usr/bin/env python3
"""Launch one real ViMoGen S0 batch for M1/M3/M5/M6."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "results/phase9/pelvis_m1_m7/protocol_v1/manifest.json"
DEFAULT_OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/s0_sampling"
DEFAULT_MANIFEST = Path(
    "/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v1/"
    "data/smoke_sample94_34122.json"
)
DEFAULT_NOISE = Path(
    "/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v2/noise_cache"
)

DEFAULT_SETTINGS = {
    "M1": {
        "guidance_scale": 0.01,
        "gradient_clip_rms": 1.0,
        "sigma_min": 0.0662879,
        "sigma_max": 0.65,
    },
    "M2": {
        "learning_rate": 0.01,
        "iterations": 2,
        "source_regularization": 0.001,
        "gradient_clip_norm": 10.0,
    },
    "M3": {
        "damping": 1.0e-6,
        "max_step_deg": 2.0,
        "sigma_min": 0.0662879,
        "sigma_max": 0.65,
    },
    "M4": {
        "shooting_sigmas": [0.55, 0.35, 0.15],
        "gn_iterations": 3,
        "damping": 1.0e-5,
        "trust_radius_deg": 4.0,
        "propagation_gain": 1.0,
        "terminal_tolerance_deg": 1.0e-4,
    },
    "M5": {
        "primal_gain": 0.05,
        "penalty": 0.1,
        "dual_gain": 1.0,
        "max_dual_norm": 50.0,
        "gradient_clip_rms": 1.0,
    },
    "M6": {
        "candidate_scale": 0.05,
        "delta": 0.1,
        "gradient_clip_rms": 1.0,
        "sigma_min": 0.0662879,
        "sigma_max": 0.65,
    },
}


@contextmanager
def runtime_environment(runtime_root: Path | None):
    """Expose unversioned runtime assets while keeping this checkout's code first."""
    if runtime_root is None:
        yield
        return

    resolved = runtime_root.resolve()
    runtime_text = str(resolved)
    added_to_path = runtime_text not in sys.path
    previous_cwd = Path.cwd()
    if added_to_path:
        sys.path.append(runtime_text)
    os.chdir(resolved)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if added_to_path:
            sys.path.remove(runtime_text)

M2_V2_SETTINGS = {
    "learning_rate": 0.005,
    "iterations": 8,
    "source_regularization": 0.001,
    "content_weight": 0.01,
    "root_weight": 0.05,
    "gradient_clip_norm": 10.0,
    "source_trust_radius": 25.0,
    "step_trust_radius": 5.0,
    "early_stop_mae_deg": 1.0,
    "early_stop_patience": 2,
}

M3_V2_SETTINGS = {
    "sigma_min": 0.10,
    "sigma_max": 0.45,
    "damping": 1.0e-6,
    "max_step_deg": 1.0,
    "projection_stride": 2,
    "max_projections": 4,
    "max_endpoint_delta_rms": 0.05,
}

M4_V2_SETTINGS = {
    "shooting_sigmas": [],
    "gn_iterations": 3,
    "damping": 1.0e-5,
    "trust_radius_deg": 4.0,
    "propagation_gain": 0.5,
    "terminal_tolerance_deg": 1.0e-4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(args: argparse.Namespace, run_root: Path, settings: dict):
    config = OmegaConf.load(args.base_config)
    config.mode = "eval"
    config.mbench_name = f"m1_m7_{args.method.lower()}_d{args.dose:+g}_s{args.seed}"
    config.experiment.global_seed = int(args.seed)
    config.experiment.auto_resume = False
    config.experiment.eval_steps = 1
    config.experiment.result_dir = str(run_root / "trainer")
    config.dataloader.test_local_batch = 2
    config.dataloader.num_workers = 4
    config.dataset.test_json_file_list = [str(args.manifest)]
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
    config.pelvis_contact_projection = {"enabled": False}
    config.source_noise = {"enabled": False}
    config.m1_m7_guidance = {
        "enabled": True,
        "method": args.method,
        "method_version": args.method_version,
        "target_delta_deg": float(args.dose),
        "trace_enabled": bool(args.trace),
        "artifact_dir": str(run_root / "guided_artifacts"),
        "settings": settings,
    }
    config.representation = {"reconciliation": {"enabled": False}}
    return config


def run(args: argparse.Namespace) -> dict:
    for required in (args.base_config, args.manifest, args.noise_cache, args.protocol):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.runtime_root is not None and not args.runtime_root.is_dir():
        raise FileNotFoundError(args.runtime_root)
    versioned_settings = {
        ("M2", "v2"): M2_V2_SETTINGS,
        ("M3", "v2"): M3_V2_SETTINGS,
        ("M4", "v2"): M4_V2_SETTINGS,
    }
    settings = dict(
        versioned_settings.get((args.method, args.method_version), DEFAULT_SETTINGS[args.method])
    )
    if args.settings_json:
        settings.update(json.loads(args.settings_json))
    parent = args.output / args.method / f"seed_{args.seed:03d}" / f"dose_{args.dose:+g}"
    attempt = 1
    run_root = parent / f"attempt_{attempt:02d}"
    while run_root.exists():
        attempt += 1
        run_root = parent / f"attempt_{attempt:02d}"
    run_root.mkdir(parents=True)
    config = build_config(args, run_root, settings)
    OmegaConf.save(config, run_root / "resolved_config.yaml")
    record = {
        "status": "RUNNING",
        "scope": "S0_REAL_VIMOGEN_BATCH",
        "method": args.method,
        "method_version": args.method_version,
        "seed": args.seed,
        "target_dose_deg": args.dose,
        "sample_ids": ["94", "34122"],
        "settings": settings,
        "code_commit": args.code_commit,
        "checkpoint_hash": sha256(ROOT / "checkpoints/model.pt") if (ROOT / "checkpoints/model.pt").is_file() else "not_available_on_runner_host",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "runtime_root": None if args.runtime_root is None else str(args.runtime_root),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "noise_cache": str(args.noise_cache),
        "run_root": str(run_root),
    }
    (run_root / "run_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29519")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        os.environ.setdefault("GROUP_RANK", "0")
        os.environ.setdefault("GROUP_WORLD_SIZE", "1")
        # The clean Git checkout owns all versioned code. Large, historically
        # untracked runtime packages and model resources remain in the server
        # project root. Keep that root last on sys.path, and temporarily make it
        # the working directory so legacy relative resource paths resolve there.
        with runtime_environment(args.runtime_root):
            from train_eval_vimogen import main as train_eval_main

            train_eval_main(config)
        record["status"] = "COMPLETED_GENERATION_PENDING_EVALUATION"
    except Exception as error:
        record["status"] = "FAILED"
        record["error"] = repr(error)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        (run_root / "run_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(DEFAULT_SETTINGS), required=True)
    parser.add_argument("--method-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--dose", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/tm2m_infer.yaml")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--noise-cache", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settings-json", default="")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.method_version == "v2" and args.method not in {"M2", "M3", "M4"}:
        parser.error("--method-version v2 is currently valid only for M2/M3/M4")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
