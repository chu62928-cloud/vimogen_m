"""Summarise residual-adaptive root-forward calibration runs."""

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
from sampling.relative_root_forward_guidance_v1_1 import PROTOCOL_NAME  # noqa: E402


DELTAS = (0, 5, -5, 10, -10)


def _attempt(root: Path, delta: int) -> Path:
    sign = "+" if delta >= 0 else ""
    parent = root / f"delta_{sign}{delta}deg"
    # Failed/incomplete retries remain as evidence but must not hide the
    # latest completed artifact from the summary.
    attempts = sorted(
        (
            p
            for p in parent.glob("attempt_*")
            if p.is_dir()
            and (p / "guided_artifacts" / "batch_000" / "g0_norm_batch.pt").is_file()
        ),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not attempts:
        raise FileNotFoundError(parent)
    return attempts[-1] / "guided_artifacts" / "batch_000"


def _config_key(path: Path) -> tuple[float, float, float, float]:
    match = re.fullmatch(
        r"gain_([-+0-9.e]+)_step_([-+0-9.e]+)"
        r"(?:_sigma_([-+0-9.e]+)_to_([-+0-9.e]+))?",
        path.name,
    )
    if not match:
        raise ValueError(f"unexpected calibration directory: {path}")
    return (
        float(match.group(1)),
        float(match.group(2)),
        0.25 if match.group(3) is None else float(match.group(3)),
        0.65 if match.group(4) is None else float(match.group(4)),
    )


def _load(path: Path, name: str) -> torch.Tensor:
    return torch.load(path / name, map_location="cpu", weights_only=True).float()


def _gate(metrics: dict, monotonicity: dict) -> bool:
    for delta in (5, -5, 10, -10):
        if str(delta) not in metrics:
            return False
        for row in metrics[str(delta)]["per_sample"]:
            if (
                row["mean_absolute_error_deg"] > 1.0
                or row["p95_absolute_error_deg"] > 2.0
                or row["forward_vector_error_p95_deg"] > 2.0
                or row["horizontal_heading_drift_p95_deg"] > 2.0
                or not row["dose_sign_correct"]
            ):
                return False
        if not all(row["tail_pass"] for row in metrics[str(delta)]["tail_safety"]["per_sample"]):
            return False
    return bool(
        monotonicity.get("positive", {}).get("absolute_monotonic", False)
        and monotonicity.get("negative", {}).get("absolute_monotonic", False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("results/phase7/relative_root_forward_v1_1")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = root.parents[2]
    mean = torch.from_numpy(np.load(repo / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(np.load(repo / "data/meta_info/std.npy")).float()
    config_parent = root / "runs" / "smoke" / "seed_000"
    configs = sorted(
        (p for p in config_parent.glob("gain_*_step_*") if p.is_dir()),
        key=_config_key,
    )
    summaries = []
    for config_root in configs:
        gain, step, sigma_min, sigma_max = _config_key(config_root)
        guided_deltas = (5, -5, 10, -10)
        g0 = {
            delta: _load(_attempt(config_root, delta), "g0_norm_batch.pt")
            for delta in guided_deltas
            if (config_root / f"delta_{'+' if delta >= 0 else ''}{delta}deg").is_dir()
        }
        baseline_delta = next(iter(g0))
        m0_baseline = _load(
            _attempt(config_root, baseline_delta), "m0_consistent_norm_batch.pt"
        )
        m0 = {delta: m0_baseline for delta in DELTAS}
        mask = torch.ones(m0[0].shape[:2], dtype=torch.bool)
        metrics = {}
        for delta in guided_deltas:
            if delta not in g0:
                continue
            metrics[str(delta)] = compute_relative_root_forward_metrics(
                m0[delta] * std + mean,
                g0[delta] * std + mean,
                mask,
                float(delta),
                protocol_name=PROTOCOL_NAME,
            )
        monotonicity = {}
        if 5 in g0 and 10 in g0:
            monotonicity["positive"] = dose_monotonicity(
                m0[0] * std + mean,
                g0[5] * std + mean,
                g0[10] * std + mean,
                mask,
            )
        if -5 in g0 and -10 in g0:
            monotonicity["negative"] = dose_monotonicity(
                m0[0] * std + mean,
                g0[-5] * std + mean,
                g0[-10] * std + mean,
                mask,
            )
        summaries.append(
            {
                "residual_gain": gain,
                "max_step_deg": step,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "protocol": PROTOCOL_NAME,
                "passed": _gate(metrics, monotonicity),
                "metrics": metrics,
                "dose_monotonicity": monotonicity,
            }
        )
    def _worst_p95(row: dict) -> float:
        values = [
            sample["p95_absolute_error_deg"]
            for metric in row["metrics"].values()
            for sample in metric["per_sample"]
        ]
        return max(values) if values else float("inf")

    summaries.sort(
        key=lambda row: (
            not row["passed"],
            _worst_p95(row),
            row["residual_gain"],
            row["max_step_deg"],
        )
    )
    result = {"protocol": PROTOCOL_NAME, "configs": summaries}
    output = args.output or root / "summaries" / "calibration_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in summaries:
        print(
            f"gain={row['residual_gain']:g} step={row['max_step_deg']:g} "
            f"passed={row['passed']} "
            f"mae5={[round(x['mean_absolute_error_deg'], 3) for x in row['metrics']['5']['per_sample']]} "
            f"mae10={[round(x['mean_absolute_error_deg'], 3) for x in row['metrics']['10']['per_sample']]}"
        )


if __name__ == "__main__":
    main()
