#!/usr/bin/env python3
"""Run the real S1 M2 repeat/batch/singleton gate without conflating failure with method quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
FIELDS = {
    "m0": "guided_artifacts/batch_000/m0_authority_norm_batch.pt",
    "candidate": "guided_artifacts/batch_000/g0_norm_batch.pt",
    "selected_noise": "guided_artifacts/batch_000/selected_source_noise_batch.pt",
    "source_noise": "m0_artifacts/batch_000/z0_replayed.pt",
}
LIMITS = {"m0": 1e-6, "candidate": 1e-5, "selected_noise": 1e-5, "source_noise": 0.0}


def _attempt(root: Path, config_id: str, dose: float) -> Path | None:
    parent = root / "generation/M2" / config_id / "seed_000" / f"dose_{dose:+g}"
    for path in reversed(sorted(parent.glob("attempt_*/run_record.json"))):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("status") == "COMPLETED_GENERATION_PENDING_EVALUATION":
            return path.parent
    return None


def _load(run: Path) -> dict:
    archive = next(run.glob("trainer/**/batch_000/mbench_raw_norm_batch.pt"), None)
    if archive is None:
        raise FileNotFoundError(f"missing sample archive below {run}")
    data = torch.load(archive, map_location="cpu", weights_only=True)
    sample_ids = [str(item) for item in data["sample_ids"]]
    tensors = {
        key: torch.load(run / path, map_location="cpu", weights_only=True).float()
        for key, path in FIELDS.items()
    }
    if any(value.shape[0] != len(sample_ids) for value in tensors.values()):
        raise ValueError(f"batch tensor/sample ID mismatch below {run}")
    return {"sample_ids": sample_ids, "tensors": tensors}


def _compare(left: dict, right: dict, sample_id: str, label: str) -> dict:
    li = left["sample_ids"].index(sample_id)
    ri = right["sample_ids"].index(sample_id)
    metrics = {}
    for field, limit in LIMITS.items():
        a = left["tensors"][field][li]
        b = right["tensors"][field][ri]
        if a.shape != b.shape:
            raise ValueError(f"{label} {sample_id} {field} shape mismatch")
        delta = a - b
        metrics[field] = {
            "max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()),
            "bitwise_equal": bool(torch.equal(a, b)),
            "limit": limit,
            "pass": bool(torch.isfinite(delta).all() and delta.abs().max() <= limit),
        }
    return {
        "comparison": label,
        "sample_id": sample_id,
        "fields": metrics,
        "pass": all(value["pass"] for value in metrics.values()),
    }


def run(args: argparse.Namespace) -> dict:
    manifest_items = json.loads(args.manifest_source.read_text(encoding="utf-8"))
    by_id = {
        str(item.get("sample_id", item.get("global_id", item.get("id")))): item
        for item in manifest_items
    }
    if not {"94", "34122"}.issubset(by_id):
        raise ValueError("source manifest must contain samples 94 and 34122")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    arms = {
        "batch_a": [by_id["94"], by_id["34122"]],
        "batch_repeat": [by_id["94"], by_id["34122"]],
        "single_94": [by_id["94"]],
        "single_34122": [by_id["34122"]],
    }
    runs = {}
    for arm, items in arms.items():
        manifest = manifest_dir / f"{arm}.json"
        encoded = json.dumps(items, indent=2, ensure_ascii=False) + "\n"
        if manifest.exists() and manifest.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to change existing manifest {manifest}")
        if not manifest.exists():
            manifest.write_text(encoded, encoding="utf-8")
        config_id = f"s1_m2_invariance_{arm}"
        existing = _attempt(args.output, config_id, args.dose)
        if existing is None:
            command = [
                str(PYTHON), str(ROOT / "experiments/run_sampling_guidance_smoke.py"),
                "--method", "M2", "--method-version", "v2",
                "--task", "relative_pelvis_s1", "--dose", str(args.dose),
                "--seed", "0", "--code-commit", args.code_revision,
                "--base-config", str(args.runtime_root / "configs/tm2m_infer.yaml"),
                "--protocol", str(ROOT / "configs/m1_m7/local_pelvis_pilot_v1.json"),
                "--runtime-root", str(args.runtime_root),
                "--manifest", str(manifest),
                "--noise-cache", str(args.runtime_root / "results/phase6/absolute_mean_pelvis_v2/noise_cache"),
                "--output", str(args.output / "generation"),
                "--settings-json", json.dumps({"learning_rate": 0.005, "iterations": 8}),
                "--config-id", config_id, "--experiment-stage", "preflight",
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            existing = _attempt(args.output, config_id, args.dose)
            if existing is None:
                raise RuntimeError(f"generation left no completed attempt for {arm}")
        runs[arm] = existing
        (args.output / "progress.json").write_text(
            json.dumps({"status": "RUNNING", "completed_arms": list(runs),
                        "run_roots": {key: str(value) for key, value in runs.items()}}, indent=2) + "\n",
            encoding="utf-8",
        )
    loaded = {arm: _load(path) for arm, path in runs.items()}
    comparisons = []
    for sample_id in ("94", "34122"):
        comparisons.append(_compare(
            loaded["batch_a"], loaded["batch_repeat"], sample_id, "repeated_batch"
        ))
        comparisons.append(_compare(
            loaded["batch_a"], loaded[f"single_{sample_id}"], sample_id, "batch_vs_single"
        ))
    result = {
        "status": "PASS" if all(item["pass"] for item in comparisons) else "FAIL_INVARIANCE",
        "protocol": "vimogen_local_pelvis_s1_m2_invariance_v1",
        "dose_deg": args.dose,
        "run_roots": {key: str(value) for key, value in runs.items()},
        "comparisons": comparisons,
    }
    output = args.output / "invariance_audit.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-source", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dose", type=float, default=5.0)
    parser.add_argument("--code-revision", default="scale-local-pelvis-s1-20260918")
    args = parser.parse_args()
    # The generation runner switches cwd to the model runtime while loading
    # assets; manifests and output roots must therefore be absolute.
    args.manifest_source = args.manifest_source.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.output = args.output.resolve()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
