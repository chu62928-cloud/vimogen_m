"""Lightweight provenance helpers shared by candidate runners and schedulers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def execution_code_files(method: str) -> dict[str, Path]:
    method_module = {
        "M1": "m1_loss_guidance.py",
        "M2": "m2_dflow_source_v2.py",
        "M3": "m3_projflow_local_v2.py",
        "M4": "m4_pcfm_v2.py",
        "M5": "m5_ldf.py",
        "M6": "m6_lyaguide.py",
    }[method]
    return {
        "runner": ROOT / "experiments/run_sampling_guidance_smoke.py",
        "train_eval": ROOT / "train_eval_vimogen.py",
        "flow_sampler": ROOT / "sampling/flow_sampler.py",
        "guidance_base": ROOT / "guidance/base.py",
        "guidance_method": ROOT / "guidance" / method_module,
        "pose_authority": ROOT / "motion_rep/pose_authority.py",
        "pelvis_angle": ROOT / "geometry/pelvis_angle.py",
        "local_pelvis": ROOT / "geometry/local_pelvis.py",
        "sampling_common": ROOT / "guidance/sampling_common.py",
        "control_metrics": ROOT / "evaluation/control_metrics.py",
        "local_pelvis_metrics": ROOT / "evaluation/local_pelvis_metrics.py",
        "content_metrics": ROOT / "evaluation/content_metrics.py",
    }


def execution_code_fingerprint(method: str) -> tuple[str, dict[str, str]]:
    hashes = {
        name: sha256_file(path) for name, path in execution_code_files(method).items()
    }
    return mapping_sha256(hashes), hashes


def attempt_parent(
    output: Path,
    method: str,
    config_id: str,
    seed: int,
    dose: float,
) -> Path:
    method_root = output / method
    if config_id != "unlabeled":
        method_root = method_root / config_id
    return method_root / f"seed_{seed:03d}" / f"dose_{dose:+g}"


__all__ = [
    "attempt_parent",
    "execution_code_files",
    "execution_code_fingerprint",
    "mapping_sha256",
    "sha256_file",
]
