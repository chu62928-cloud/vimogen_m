"""Prompt-level paired statistics for three-way recovery experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .mbench_threeway import METHODS


PAIR_ORDER = (
    ("reconciled", "absolute_position"),
    ("reconciled", "velocity_integral"),
)


def _bootstrap_ci(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _wilcoxon(values: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if len(values) == 0 or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)


def _stable_seed(*parts: str, base: int) -> int:
    value = base
    for part in parts:
        for char in part:
            value = (value * 131 + ord(char)) % 2147483647
    return value


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(index, value) for index, value in enumerate(p_values) if value is not None]
    indexed.sort(key=lambda item: item[1])
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(indexed):
        corrected = min(1.0, (total - rank) * value)
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted


def _median_by_sample(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for manifest in records:
        condition = str(manifest.get("condition", "unknown"))
        for record in manifest.get("records", []):
            sample_id = str(record["sample_id"])
            for method, metrics in record.get("methods", {}).items():
                key = (condition, sample_id, method)
                for name, value in metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        grouped[key][name].append(float(value))
    output = {}
    for key, metrics in grouped.items():
        output[key] = {name: float(np.median(values)) for name, values in metrics.items()}
    return output


def summarize_drift(
    manifest_paths: list[Path],
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260824,
) -> dict[str, Any]:
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    values = _median_by_sample(manifests)
    conditions = sorted({key[0] for key in values})
    output: dict[str, Any] = {
        "status": "VALID",
        "protocol": "vimogen_mbench_threeway_statistics_v1",
        "manifest_paths": [str(path) for path in manifest_paths],
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
        "conditions": {},
    }
    metric_names = (
        "anchor_deviation_mean_m",
        "anchor_deviation_endpoint_m",
        "root_endpoint_deviation_m",
        "anchor_drift_auc_m",
        "anchor_drift_slope_m_per_frame",
        "raw_position_velocity_residual_mean_m_per_frame",
        "raw_integrated_separation_endpoint_m",
        "raw_integrated_separation_auc_m",
    )
    for condition in conditions:
        sample_ids = sorted({key[1] for key in values if key[0] == condition})
        condition_output: dict[str, Any] = {
            "sample_count": len(sample_ids),
            "metrics": {},
        }
        for metric in metric_names:
            pairs = []
            method_summary = {}
            for method in METHODS:
                method_values = np.asarray(
                    [values[(condition, sample, method)][metric] for sample in sample_ids
                     if (condition, sample, method) in values
                     and metric in values[(condition, sample, method)]],
                    dtype=np.float64,
                )
                if len(method_values):
                    method_summary[method] = {
                        "n": int(len(method_values)),
                        "mean": float(method_values.mean()),
                        "std": float(method_values.std(ddof=0)),
                        "median": float(np.median(method_values)),
                        "q25": float(np.percentile(method_values, 25)),
                        "q75": float(np.percentile(method_values, 75)),
                    }
            metric_output = {"method_summary": method_summary, "paired": {}}
            for left, right in PAIR_ORDER:
                common = [
                    sample for sample in sample_ids
                    if (condition, sample, left) in values
                    and (condition, sample, right) in values
                    and metric in values[(condition, sample, left)]
                    and metric in values[(condition, sample, right)]
                ]
                differences = np.asarray(
                    [values[(condition, sample, left)][metric]
                     - values[(condition, sample, right)][metric]
                     for sample in common],
                    dtype=np.float64,
                )
                low, high = _bootstrap_ci(
                    differences,
                    bootstrap_repetitions,
                    _stable_seed(condition, metric, left, right, base=bootstrap_seed),
                )
                pairs.append(_wilcoxon(differences))
                metric_output["paired"][f"{left}_minus_{right}"] = {
                    "n": int(len(differences)),
                    "median_difference": float(np.median(differences)) if len(differences) else None,
                    "mean_difference": float(differences.mean()) if len(differences) else None,
                    "bootstrap_ci95": [low, high],
                    "wilcoxon_p": _wilcoxon(differences),
                }
            p_values = [pairs[0], pairs[1]]
            corrected = holm_adjust(p_values)
            for index, pair in enumerate(PAIR_ORDER):
                name = f"{pair[0]}_minus_{pair[1]}"
                metric_output["paired"][name]["holm_p"] = corrected[index]
            condition_output["metrics"][metric] = metric_output
        output["conditions"][condition] = condition_output
    return output


def load_mbench_per_motion(path: Path) -> dict[str, dict[str, float]]:
    """Load official MBench per-motion values keyed by motion id and dimension."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float]] = {}
    for motion in payload.get("motions", []):
        motion_id = str(motion.get("id"))
        dimensions = {}
        for dimension, values in motion.get("dimensions", {}).items():
            if isinstance(values, dict) and isinstance(values.get("value"), (int, float)):
                dimensions[dimension] = float(values["value"])
        result[motion_id] = dimensions
    return result


def summarize_official_mbench(
    run_record_paths: list[Path],
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260824,
) -> dict[str, Any]:
    """Summarize official MBench motion-quality results at prompt level.

    Three random seeds are first collapsed by the within-prompt median.  The
    paired bootstrap and Wilcoxon tests then use prompts, not seed-level rows,
    as the independent statistical units.
    """
    seed_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    provenance = []
    for record_path in run_record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "COMPLETED":
            raise ValueError(f"official MBench record is not completed: {record_path}")
        per_motion_path = Path(record["per_motion_path"])
        motions = load_mbench_per_motion(per_motion_path)
        condition = str(record["condition"])
        method = str(record["method"])
        seed = int(record["seed"])
        provenance.append({
            "record_path": str(record_path),
            "per_motion_path": str(per_motion_path),
            "condition": condition,
            "method": method,
            "seed": seed,
            "motion_count": len(motions),
        })
        for sample_id, dimensions in motions.items():
            for metric, value in dimensions.items():
                if np.isfinite(value):
                    seed_values[(condition, sample_id, method, metric)].append(float(value))

    collapsed: dict[tuple[str, str, str, str], float] = {
        key: float(np.median(values)) for key, values in seed_values.items()
    }
    metric_names = sorted({key[3] for key in collapsed})
    conditions = sorted({key[0] for key in collapsed})
    output: dict[str, Any] = {
        "status": "VALID",
        "protocol": "vimogen_publication_mbench_motion_quality_statistics_v1",
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
        "seed_collapse": "median_within_prompt",
        "statistical_unit": "prompt_id",
        "provenance": provenance,
        "conditions": {},
    }
    for condition in conditions:
        condition_out: dict[str, Any] = {"metrics": {}}
        for metric in metric_names:
            sample_ids = sorted({
                key[1] for key in collapsed
                if key[0] == condition and key[3] == metric
                and all((condition, key[1], method, metric) in collapsed for method in METHODS)
            })
            method_summary: dict[str, Any] = {}
            for method in METHODS:
                values = np.asarray([
                    collapsed[(condition, sample, method, metric)] for sample in sample_ids
                ], dtype=np.float64)
                if values.size:
                    method_summary[method] = {
                        "n": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "median": float(np.median(values)),
                        "q25": float(np.percentile(values, 25)),
                        "q75": float(np.percentile(values, 75)),
                    }
            metric_out: dict[str, Any] = {
                "sample_count": len(sample_ids),
                "method_summary": method_summary,
                "paired": {},
            }
            p_values: list[float | None] = []
            for left, right in PAIR_ORDER:
                differences = np.asarray([
                    collapsed[(condition, sample, left, metric)]
                    - collapsed[(condition, sample, right, metric)]
                    for sample in sample_ids
                ], dtype=np.float64)
                low, high = _bootstrap_ci(
                    differences,
                    bootstrap_repetitions,
                    _stable_seed(condition, metric, left, right, base=bootstrap_seed),
                )
                p_value = _wilcoxon(differences)
                p_values.append(p_value)
                metric_out["paired"][f"{left}_minus_{right}"] = {
                    "n": int(differences.size),
                    "median_difference": float(np.median(differences)) if differences.size else None,
                    "mean_difference": float(differences.mean()) if differences.size else None,
                    "bootstrap_ci95": [low, high],
                    "wilcoxon_p": p_value,
                }
            adjusted = holm_adjust(p_values)
            for index, pair in enumerate(PAIR_ORDER):
                metric_out["paired"][f"{pair[0]}_minus_{pair[1]}"]["holm_p"] = adjusted[index]
            condition_out["metrics"][metric] = metric_out
        output["conditions"][condition] = condition_out
    return output


__all__ = ["holm_adjust", "load_mbench_per_motion", "summarize_drift", "summarize_official_mbench"]
