#!/usr/bin/env python3
"""Run one frozen sample34122 pelvis/contact sampling-projection pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.pelvis_contact_flow_projection_v0_1 import (  # noqa: E402
    EUCLIDEAN_METRIC,
    KINEMATIC_TEMPORAL_METRIC,
    PROTOCOL_NAME,
    ProjectorConfig,
    write_strict_json,
)


DEFAULT_PROTOCOL_ROOT = Path(
    "/root/autodl-tmp/vimogen_pelvis_contact_v3_0_1_results/"
    "protocol_v3_0_1_final"
)
DEFAULT_NOISE_CACHE = ROOT / "results/phase6/absolute_mean_pelvis_v2/noise_cache"
DEFAULT_OUTPUT = ROOT / "results/phase8/pelvis_contact_flow_projection_v0_1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for child in files:
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    else:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def ensure_manifest(output_root: Path) -> Path:
    destination = output_root / "data/sample34122.json"
    if destination.is_file():
        return destination
    source = ROOT / "results/phase7/relative_root_forward_v1/data/smoke_sample94_34122.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    selected = [row for row in rows if str(row.get("sample_id", row.get("global_id"))) == "34122"]
    if len(selected) != 1:
        raise RuntimeError(f"expected one sample34122 row, found {len(selected)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(destination, selected)
    return destination


def find_noise_record(cache: Path, sample_id: str, seed: int) -> dict[str, Any]:
    matches = []
    for metadata_path in cache.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = metadata.get("key", {})
        if str(key.get("sample_id")) == str(sample_id) and int(key.get("seed", -1)) == int(seed):
            tensor_name = metadata.get("tensor_file", metadata_path.with_suffix(".pt").name)
            tensor_path = metadata_path.parent / str(tensor_name)
            if tensor_path.is_file():
                matches.append((metadata_path, tensor_path, metadata))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one noise cache record for sample={sample_id}, seed={seed}; "
            f"found {len(matches)}"
        )
    metadata_path, tensor_path, metadata = matches[0]
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_path(metadata_path),
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha256_path(tensor_path),
        "metadata": metadata,
    }


def next_attempt(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while (parent / f"attempt_{index:02d}").exists():
        index += 1
    result = parent / f"attempt_{index:02d}"
    result.mkdir()
    return result


def build_config(args: argparse.Namespace, run_root: Path, manifest: Path) -> Any:
    config = OmegaConf.load(ROOT / "configs/tm2m_infer.yaml")
    config.mode = "eval"
    config.mbench_name = (
        f"pelvis_contact_projection_{args.metric}_{args.side}_dose"
        f"{args.target_delta_deg:g}_seed0"
    )
    config.experiment.global_seed = 0
    config.experiment.auto_resume = False
    config.experiment.eval_steps = 1
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
    config.source_noise = {"enabled": False}
    projector = ProjectorConfig(
        metric=args.metric,
        sigma_min=0.0662879,
        sigma_max=0.65,
        lambda_root=args.lambda_root,
        lambda_skel=args.lambda_skel,
        lambda_vel=args.lambda_vel,
        lambda_acc=args.lambda_acc,
        epsilon=1.0e-6,
        contact_weight=args.contact_weight,
        max_relinearization_iters=5,
        pelvis_tolerance_deg=0.25,
        contact_tolerance_m=0.001,
        penetration_epsilon_m=0.0005,
        max_joint_increment_deg=5.0,
        max_root_translation_m=0.010,
    )
    projector.validate()
    config.pelvis_contact_projection = {
        **asdict(projector),
        "enabled": True,
        "protocol_root": str(args.protocol_root),
        "artifact_dir": str(run_root / "projection_artifacts"),
        "sample_id": "34122",
        "side": args.side,
        "target_delta_deg": float(args.target_delta_deg),
        "model_path": None if args.model_path is None else str(args.model_path),
    }
    config.representation = {"reconciliation": {"enabled": False}}
    return config


def freeze_inputs(
    args: argparse.Namespace,
    run_root: Path,
    config_path: Path,
    manifest: Path,
) -> dict[str, Any]:
    import torch
    from trainer.scheduler import FlowMatchScheduler

    protocol = json.loads((args.protocol_root / "protocol.json").read_text(encoding="utf-8"))
    case = next(
        item for item in protocol["cases"] if str(item["sample_id"]) == "34122"
    )
    contact_payload = {
        side: case["sides"][side]["evidence"]["valid_masks"]
        for side in ("left", "right")
    }
    contact_hash = hashlib.sha256(
        json.dumps(contact_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    scheduler = FlowMatchScheduler()
    scheduler.set_timesteps(50, training=False, denoising_strength=0.7)
    sigmas = [float(value) for value in scheduler.sigmas.detach().cpu().tolist()]
    guidance_mask = [
        0.0662879 - 1.0e-8 <= value <= 0.65 + 1.0e-8 and value > 0.0
        for value in sigmas[:50]
    ]
    checkpoint = ROOT / "checkpoints/model.pt"
    smplx_path = args.model_path
    if smplx_path is None:
        smplx_path = Path(protocol["inputs"]["smplx_model"]["path"])
    snapshot = {
        "protocol": PROTOCOL_NAME,
        "status": "INPUTS_FROZEN_BEFORE_GENERATION",
        "source_revision": git_value("rev-parse", "HEAD"),
        "source_branch": git_value("branch", "--show-current"),
        "dirty_state": git_value("status", "--porcelain"),
        "manifest": {"path": str(manifest), "sha256": sha256_path(manifest)},
        "resolved_config": {
            "path": str(config_path),
            "sha256": sha256_path(config_path),
        },
        "vimogen_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_path(checkpoint),
        },
        "smplx_model": {
            "path": str(smplx_path),
            "sha256": sha256_path(smplx_path),
        },
        "frozen_v3_protocol": {
            "path": str(args.protocol_root / "protocol.json"),
            "sha256": sha256_path(args.protocol_root / "protocol.json"),
        },
        "contact_mask_sha256": contact_hash,
        "heel_toe_marker_index_sha256": sha256_path(
            args.protocol_root / "foot_patches.json"
        ),
        "m0_motion_sha256": sha256_path(args.protocol_root / "m0_physical.pt"),
        "initial_noise": find_noise_record(args.noise_cache, "34122", 0),
        "sampling_schedule": {
            "num_inference_steps": 50,
            "denoising_strength": 0.7,
            "sigmas": sigmas,
        },
        "guidance_step_mask": guidance_mask,
        "projection_window": case["sides"][args.side]["stable_window"],
        "metric": args.metric,
        "target_delta_deg": float(args.target_delta_deg),
        "seed": 0,
        "sample_id": "34122",
        "side": args.side,
        "torch_version": torch.__version__,
    }
    write_strict_json(run_root / "input_snapshot.json", snapshot)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=(EUCLIDEAN_METRIC, KINEMATIC_TEMPORAL_METRIC), required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--target-delta-deg", type=float, choices=(2.0, 5.0, 10.0), required=True)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--noise-cache", type=Path, default=DEFAULT_NOISE_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--lambda-root", type=float, default=10.0)
    parser.add_argument("--lambda-skel", type=float, default=1.0)
    parser.add_argument("--lambda-vel", type=float, default=1.0)
    parser.add_argument("--lambda-acc", type=float, default=5.0)
    parser.add_argument("--contact-weight", type=float, default=1.0e6)
    args = parser.parse_args()
    manifest = ensure_manifest(args.output_root)
    parent = (
        args.output_root
        / "pilot_sample34122"
        / args.side
        / args.metric
        / f"dose_{args.target_delta_deg:+g}deg"
    )
    run_root = next_attempt(parent)
    config = build_config(args, run_root, manifest)
    config_path = run_root / "resolved_config.yaml"
    OmegaConf.save(config, config_path)
    snapshot = freeze_inputs(args, run_root, config_path, manifest)
    record = {
        "protocol": PROTOCOL_NAME,
        "status": "RUNNING",
        "run_root": str(run_root),
        "input_snapshot": str(run_root / "input_snapshot.json"),
        "input_snapshot_sha256": sha256_path(run_root / "input_snapshot.json"),
        "source_revision": snapshot["source_revision"],
        "sample_id": "34122",
        "seed": 0,
        "side": args.side,
        "metric": args.metric,
        "target_delta_deg": float(args.target_delta_deg),
    }
    write_strict_json(run_root / "run_record.json", record)
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
        record["status"] = "COMPLETED_GENERATION_PENDING_EVALUATION"
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write_strict_json(run_root / "run_record.json", record)
    print(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
