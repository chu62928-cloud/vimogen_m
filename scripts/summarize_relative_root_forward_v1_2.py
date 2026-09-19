"""Summarise the multi-seed constraint-first v1.2 smoke matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.relative_root_forward_v1 import (  # noqa: E402
    compute_relative_root_forward_metrics,
    dose_monotonicity,
)
from sampling.relative_root_forward_guidance_v1_2 import PROTOCOL_NAME  # noqa: E402


DELTAS = (-10, -5, 5, 10)
DEFAULT_SEEDS = (0, 42, 464229750, 1057660199, 1386772747)


def _latest_artifact(root: Path, seed: int, delta: int, parameter_key: str | None = None) -> Path:
    sign = "+" if delta >= 0 else ""
    candidates = []
    parameter_dirs = (
        [(root / "runs" / "smoke" / f"seed_{seed:03d}" / parameter_key)]
        if parameter_key is not None
        else (root / "runs" / "smoke" / f"seed_{seed:03d}").glob("*")
    )
    for parameter in parameter_dirs:
        parent = parameter / f"delta_{sign}{delta}deg"
        for attempt in parent.glob("attempt_*"):
            artifact = attempt / "guided_artifacts" / "batch_000"
            if (artifact / "g0_norm_batch.pt").is_file():
                candidates.append((int(attempt.name.split("_")[-1]), artifact))
    if not candidates:
        raise FileNotFoundError(f"no completed delta={delta} seed={seed} below {root}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _choose_parameter_key(root: Path, seeds: list[int], parameter_key: str | None) -> str | None:
    """Choose one common parameter directory; never mix parameters per dose."""
    if parameter_key is not None:
        return parameter_key
    if not seeds:
        return None
    base = root / "runs" / "smoke" / f"seed_{seeds[0]:03d}"
    candidates = []
    for directory in base.glob("*"):
        if not directory.is_dir():
            continue
        complete = True
        for delta in DELTAS:
            sign = "+" if delta >= 0 else ""
            if not list((directory / f"delta_{sign}{delta}deg").glob("attempt_*/guided_artifacts/batch_000/g0_norm_batch.pt")):
                complete = False
                break
        if complete:
            candidates.append(directory.name)
    if not candidates:
        return None
    # Prefer the explicitly versioned pitch/sigma key when several historical
    # calibration directories are present, then use lexical order for a stable
    # fallback.  The selected key is written to the summary manifest.
    candidates.sort(key=lambda value: ("pitch_" not in value, value))
    return candidates[0]


def _load(path: Path, name: str) -> torch.Tensor:
    return torch.load(path / name, map_location="cpu", weights_only=True).float()


def _gate(metrics: dict, monotonicity: dict) -> bool:
    for delta in DELTAS:
        rows = metrics.get(str(delta), {}).get("per_sample", [])
        tails = metrics.get(str(delta), {}).get("tail_safety", {}).get("per_sample", [])
        whole = metrics.get(str(delta), {}).get("whole_body", {})
        if not rows or not all(row["dose_sign_correct"] for row in rows):
            return False
        if any(
            row["mean_absolute_error_deg"] > 1.0
            or row["p95_absolute_error_deg"] > 2.0
            or row["forward_vector_error_p95_deg"] > 2.0
            or row["horizontal_heading_drift_p95_deg"] > 2.0
            for row in rows
        ):
            return False
        if any(not row["tail_pass"] for row in tails):
            return False
        if float(whole.get("q_rigid", float("inf"))) > 0.2:
            return False
        trunk = whole.get("trunk_change_deg", {})
        if trunk.get("p95") is None or float(trunk["p95"]) > 2.0:
            return False
    return bool(
        monotonicity.get("positive", {}).get("absolute_monotonic", False)
        and monotonicity.get("negative", {}).get("absolute_monotonic", False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/phase7/relative_root_forward_v1_2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--parameter-key",
        default=None,
        help="One parameter directory name; prevents mixing calibration settings across doses.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    parameter_key = _choose_parameter_key(root, list(args.seeds), args.parameter_key)
    repo = root.parents[2]
    mean = torch.from_numpy(np.load(repo / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(repo / "data/meta_info/std.npy")).float()
    seed_summaries = []
    for seed in args.seeds:
        artifacts = {delta: _latest_artifact(root, seed, delta, parameter_key) for delta in DELTAS}
        first = artifacts[DELTAS[0]]
        m0 = _load(first, "m0_consistent_norm_batch.pt")
        mask = torch.ones(m0.shape[:2], dtype=torch.bool)
        metrics = {}
        g0 = {}
        for delta, artifact in artifacts.items():
            g0[delta] = _load(artifact, "g0_norm_batch.pt")
            metrics[str(delta)] = compute_relative_root_forward_metrics(
                m0 * std + mean,
                g0[delta] * std + mean,
                mask,
                float(delta),
                protocol_name=PROTOCOL_NAME,
            )
        monotonicity = {
            "positive": dose_monotonicity(m0 * std + mean, g0[5] * std + mean, g0[10] * std + mean, mask),
            "negative": dose_monotonicity(m0 * std + mean, g0[-5] * std + mean, g0[-10] * std + mean, mask),
        }
        seed_summaries.append({"seed": seed, "passed": _gate(metrics, monotonicity), "metrics": metrics, "dose_monotonicity": monotonicity})
    result = {
        "protocol": PROTOCOL_NAME,
        "seeds": list(args.seeds),
        "parameter_key": parameter_key,
        "all_seeds_passed": all(row["passed"] for row in seed_summaries),
        "per_seed": seed_summaries,
    }
    # Cross-seed aggregates are descriptive only.  Gate decisions remain
    # per-seed and per-sample so an average cannot hide a failed seed.
    aggregate: dict[str, dict[str, float | None]] = {}
    metric_names = (
        "mean_absolute_error_deg",
        "p95_absolute_error_deg",
        "forward_vector_error_p95_deg",
        "horizontal_heading_drift_p95_deg",
    )
    for delta in DELTAS:
        for metric_name in metric_names:
            values = [
                row["metrics"][str(delta)]["per_sample"][sample][metric_name]
                for row in seed_summaries
                for sample in range(len(row["metrics"][str(delta)]["per_sample"]))
            ]
            values_np = np.asarray(values, dtype=float)
            aggregate[f"{delta}:{metric_name}"] = {
                "mean": float(values_np.mean()) if values_np.size else None,
                "std": float(values_np.std(ddof=1)) if values_np.size > 1 else None,
                "cross_seed_p95": float(np.quantile(values_np, 0.95)) if values_np.size else None,
                "worst": float(values_np.max()) if values_np.size else None,
            }
    result["cross_seed_aggregates"] = aggregate
    output = args.output or root / "summaries" / "multi_seed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in seed_summaries:
        print(f"seed={row['seed']} passed={row['passed']}")
    print(f"all_seeds_passed={result['all_seeds_passed']}")


if __name__ == "__main__":
    main()
