#!/usr/bin/env python3
"""Run the bounded S1 screen/confirm/stress protocol.

The runner owns one GPU generation process at a time and keeps every attempt
in a method/config directory.  Screening selects two configurations, confirm
selects one, and stress only evaluates that frozen winner.  M2 is explicitly
skipped when its independent reproducibility report is not a pass.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from omegaconf import OmegaConf

from evaluation.physical_metrics import V2_HARD_METRICS
from experiments.run_sampling_guidance_smoke import (
    DEFAULT_SETTINGS,
    M2_V2_SETTINGS,
    M3_V2_SETTINGS,
    M4_V2_SETTINGS,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_sampling_guidance_smoke.py"
EVALUATOR = ROOT / "experiments/evaluate_sampling_guidance_smoke.py"
PHYSICAL_EVALUATOR = ROOT / "scripts/evaluate_s0_physical.py"
S1_CONFIG = ROOT / "configs/m1_m7/s1.yaml"

METHOD_VERSIONS = {
    "M1": "v1",
    "M2": "v2",
    "M3": "v2",
    "M4": "v2",
    "M5": "v1",
    "M6": "v1",
}
STAGE_JOBS = {
    "screen": {"seeds": (0,), "doses": (-2.0, 2.0), "physical_count": 4},
    "confirm": {
        "seeds": (0, 42),
        "doses": (-5.0, -2.0, 0.0, 2.0, 5.0),
        "physical_count": 20,
    },
    "stress": {
        "seeds": (0, 42),
        "doses": (-10.0, 10.0),
        # The config directory contains screen + confirm + stress records.
        "physical_count": 28,
    },
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _same(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _config() -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(S1_CONFIG), resolve=True)
    if not isinstance(value, dict):
        raise ValueError("S1 config must be a mapping")
    return value


def _settings(method: str, values: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "M1": DEFAULT_SETTINGS["M1"],
        "M2": M2_V2_SETTINGS,
        "M3": M3_V2_SETTINGS,
        "M4": M4_V2_SETTINGS,
        "M5": DEFAULT_SETTINGS["M5"],
        "M6": DEFAULT_SETTINGS["M6"],
    }
    result = dict(defaults[method])
    result.update(values)
    return result


def _attempt_parent(config_root: Path, method: str, seed: int, dose: float) -> Path:
    return config_root / method / f"seed_{seed:03d}" / f"dose_{dose:+g}"


def _find_attempt(
    config_root: Path,
    method: str,
    seed: int,
    dose: float,
    settings: dict[str, Any],
    code_commit: str,
) -> Path | None:
    parent = _attempt_parent(config_root, method, seed, dose)
    if not parent.is_dir():
        return None
    for attempt in sorted(parent.glob("attempt_*"), reverse=True):
        record_path = attempt / "run_record.json"
        if not record_path.is_file():
            continue
        try:
            record = _json(record_path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            str(record.get("status", "")).startswith("COMPLETED")
            and str(record.get("code_commit")) == code_commit
            and str(record.get("method")) == method
            and int(record.get("seed", -1)) == seed
            and float(record.get("target_dose_deg", 999.0)) == dose
            and _same(record.get("settings", {}), settings)
        ):
            return attempt
    return None


def _latest_attempt(config_root: Path, method: str, seed: int, dose: float) -> Path | None:
    parent = _attempt_parent(config_root, method, seed, dose)
    attempts = sorted(parent.glob("attempt_*")) if parent.is_dir() else []
    return attempts[-1] if attempts else None


def _run_generation(
    *,
    config_root: Path,
    method: str,
    seed: int,
    dose: float,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    attempt = _find_attempt(config_root, method, seed, dose, settings, args.code_commit)
    generation: dict[str, Any] = {"status": "REUSED" if attempt else "LAUNCHED"}
    if attempt is None:
        command = [
            sys.executable,
            str(RUNNER),
            "--method",
            method,
            "--method-version",
            METHOD_VERSIONS[method],
            "--dose",
            str(dose),
            "--seed",
            str(seed),
            "--code-commit",
            args.code_commit,
            "--base-config",
            str(args.base_config),
            "--protocol",
            str(args.protocol),
            "--runtime-root",
            str(args.runtime_root),
            "--manifest",
            str(args.manifest),
            "--noise-cache",
            str(args.noise_cache),
            "--output",
            str(config_root),
            "--settings-json",
            json.dumps(settings, separators=(",", ":")),
        ]
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        generation.update(
            {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        attempt = _find_attempt(config_root, method, seed, dose, settings, args.code_commit)
        if attempt is None:
            attempt = _latest_attempt(config_root, method, seed, dose)
    if attempt is None:
        return {
            "status": "GENERATION_FAILED",
            "method": method,
            "seed": seed,
            "dose": dose,
            "settings": settings,
            "generation": generation,
        }

    evaluation_dir = attempt / "evaluation"
    evaluation_summary = evaluation_dir / "summary.json"
    if evaluation_summary.is_file():
        evaluation = _json(evaluation_summary)
    elif evaluation_dir.exists():
        evaluation = {"status": "EVALUATION_OUTPUT_PARTIAL"}
    else:
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--run-root",
                str(attempt),
                "--output",
                str(evaluation_dir),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        evaluation = (
            _json(evaluation_summary)
            if completed.returncode == 0 and evaluation_summary.is_file()
            else {
                "status": "EVALUATION_FAILED",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            }
        )
    return {
        "status": "COMPLETED" if evaluation.get("records") else evaluation.get("status"),
        "method": method,
        "seed": seed,
        "dose": dose,
        "settings": settings,
        "attempt": str(attempt),
        "generation": generation,
        "evaluation": evaluation,
        "evaluation_records": evaluation.get("records", []),
    }


def _physical_evaluation(
    *, config_root: Path, stage: str, expected_count: int, args: argparse.Namespace
) -> dict[str, Any]:
    output = config_root / f"physical_evaluation_{stage}"
    summary = output / "summary.json"
    if summary.is_file():
        return _json(summary)
    if output.exists():
        return {"status": "PHYSICAL_EVALUATION_OUTPUT_PARTIAL"}
    command = [
        sys.executable,
        str(PHYSICAL_EVALUATOR),
        "--source",
        str(config_root),
        "--reference",
        str(args.reference),
        "--thresholds",
        str(args.thresholds),
        "--thresholds-v2",
        str(args.thresholds_v2),
        "--output",
        str(output),
        "--expected-count",
        str(expected_count),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0 or not summary.is_file():
        return {
            "status": "PHYSICAL_EVALUATION_FAILED",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        }
    return _json(summary)


def _physical_by_run(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["run_id"]): row for row in summary.get("records", [])}


def _job_records(jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for job in jobs:
        for path_text in job.get("evaluation_records", []):
            path = Path(path_text)
            if path.is_file():
                records.append(_json(path))
    return records


def _summary(
    *,
    method: str,
    config_index: int,
    stage: str,
    jobs: list[dict[str, Any]],
    physical: dict[str, Any],
) -> dict[str, Any]:
    physical_map = _physical_by_run(physical)
    rows = []
    for record in _job_records(jobs):
        control = record.get("all_metrics", {}).get("control", {}).get("per_sequence", [{}])[0]
        content = record.get("all_metrics", {}).get("content", {}).get("per_sequence", [{}])[0]
        diagnostics = record.get("all_diagnostics", {})
        p = physical_map.get(str(record.get("run_id")), {})
        v2 = p.get("physical_v2", {})
        metric_row = (v2.get("per_sequence") or [{}])[0]
        thresholds = v2.get("thresholds", {})
        ratios = [
            float(metric_row[name]) / float(thresholds[name])
            for name in V2_HARD_METRICS
            if metric_row.get(name) is not None
            and thresholds.get(name) not in (None, 0)
        ]
        rows.append(
            {
                "run_id": record.get("run_id"),
                "seed": record.get("seed"),
                "dose_deg": record.get("target_dose_deg"),
                "sample_id": record.get("prompt_id"),
                "angle_pass": bool(control.get("sequence_angle_pass", False)),
                "numerical_ok": bool(
                    diagnostics.get("nonfinite_count", 0) == 0
                    and not diagnostics.get("fallback_used", False)
                ),
                "physical_v2_status": v2.get("status"),
                "physical_v2_pass": v2.get("status") == "EVALUATED_PASS",
                "physical_v2_max_ratio": max(ratios, default=0.0),
                "mpjpe_mm": float(content.get("mpjpe_vs_m0_mm", float("inf"))),
                "root_translation_p95_mm": float(
                    content.get("root_translation_deviation_p95_mm", float("inf"))
                ),
                "wall_time_sec": float(diagnostics.get("wall_time_sec", 0.0)),
            }
        )
    complete = len(rows) == len(jobs) * 2
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "stage": stage,
        "method": method,
        "config_index": config_index,
        "sequence_count": len(rows),
        "expected_sequence_count": len(jobs) * 2,
        "angle_pass_count": sum(row["angle_pass"] for row in rows),
        "numerical_ok_count": sum(row["numerical_ok"] for row in rows),
        "physical_v2_pass_count": sum(row["physical_v2_pass"] for row in rows),
        "physical_v2_max_ratio_median": _median(
            row["physical_v2_max_ratio"] for row in rows
        ),
        "physical_v2_max_ratio_p95": _quantile(
            [row["physical_v2_max_ratio"] for row in rows], 0.95
        ),
        "content_mpjpe_median_mm": _median(row["mpjpe_mm"] for row in rows),
        "root_translation_median_mm": _median(
            row["root_translation_p95_mm"] for row in rows
        ),
        "wall_time_sec": sum(row["wall_time_sec"] for row in rows),
        "rows": rows,
    }


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("inf")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _sort_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    sequence_count = max(int(summary.get("sequence_count", 0)), 1)
    return (
        -int(summary.get("angle_pass_count", 0)),
        -int(
            summary.get("numerical_ok_count", 0) == sequence_count
            and summary.get("status") == "COMPLETE"
        ),
        -int(summary.get("physical_v2_pass_count", 0)),
        float(summary.get("physical_v2_max_ratio_median", float("inf"))),
        float(summary.get("physical_v2_max_ratio_p95", float("inf"))),
        float(summary.get("content_mpjpe_median_mm", float("inf"))),
        float(summary.get("root_translation_median_mm", float("inf"))),
        float(summary.get("wall_time_sec", float("inf"))),
        int(summary.get("config_index", 10**9)),
    )


def _load_progress(path: Path, code_commit: str) -> dict[str, Any]:
    if path.is_file():
        value = _json(path)
        if value.get("code_commit") != code_commit:
            raise ValueError("existing S1 progress belongs to a different code commit")
        return value
    return {
        "protocol": "vimogen_m1_m7_s1_bounded_tuning_v1",
        "code_commit": code_commit,
        "jobs": [],
        "selections": {},
        "updated_at": time.time(),
    }


def _job_key(stage: str, method: str, config_index: int, seed: int, dose: float) -> tuple[Any, ...]:
    return stage, method, config_index, seed, dose


def run_stage(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    config = _config()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "s1_progress.json"
    progress = _load_progress(progress_path, args.code_commit)
    invariant = _json(args.m2_invariant) if args.m2_invariant.is_file() else {}
    methods = args.methods or tuple(f"M{index}" for index in range(1, 7))
    stage_spec = STAGE_JOBS[stage]
    all_stage_summaries: dict[str, list[dict[str, Any]]] = {}

    for method in methods:
        if method == "M2" and invariant.get("status") != "PASS":
            progress.setdefault("selections", {}).setdefault(method, {})[
                "status"
            ] = "S1_FAILED_INVARIANT"
            continue
        values = config["method_grids"][method]["values"]
        if stage == "screen":
            selected_indices = list(range(1, min(len(values), 8) + 1))
        elif stage == "confirm":
            selected_indices = progress.get("selections", {}).get(method, {}).get(
                "screen_top2", []
            )
            if len(selected_indices) != 2:
                raise RuntimeError(f"screen selection missing for {method}")
        else:
            selected_indices = [
                progress.get("selections", {}).get(method, {}).get("confirm_winner")
            ]
            if selected_indices[0] is None:
                raise RuntimeError(f"confirm winner missing for {method}")

        stage_summaries = []
        for config_index in selected_indices:
            config_root = output / "runs" / method / f"config_{int(config_index):02d}"
            settings = _settings(method, dict(values[int(config_index) - 1]))
            jobs = []
            for seed in stage_spec["seeds"]:
                for dose in stage_spec["doses"]:
                    key = _job_key(stage, method, int(config_index), seed, dose)
                    existing = next(
                        (
                            item
                            for item in progress["jobs"]
                            if tuple(item.get(name) for name in ("stage", "method", "config_index", "seed", "dose"))
                            == key
                        ),
                        None,
                    )
                    if existing is None:
                        existing = _run_generation(
                            config_root=config_root,
                            method=method,
                            seed=seed,
                            dose=dose,
                            settings=settings,
                            args=args,
                        )
                        existing.update(
                            {
                                "stage": stage,
                                "config_index": int(config_index),
                            }
                        )
                        progress["jobs"].append(existing)
                        progress["updated_at"] = time.time()
                        _write(progress_path, progress)
                    jobs.append(existing)
            physical = _physical_evaluation(
                config_root=config_root,
                stage=stage,
                expected_count=int(stage_spec["physical_count"]),
                args=args,
            )
            result = _summary(
                method=method,
                config_index=int(config_index),
                stage=stage,
                jobs=jobs,
                physical=physical,
            )
            result["physical_status"] = physical.get("status")
            _write(config_root / f"stage_{stage}_summary.json", result)
            stage_summaries.append(result)
        all_stage_summaries[method] = stage_summaries

    for method, summaries in all_stage_summaries.items():
        ordered = sorted(summaries, key=_sort_key)
        selection = progress.setdefault("selections", {}).setdefault(method, {})
        if stage == "screen":
            selection["screen_top2"] = [item["config_index"] for item in ordered[:2]]
            selection["screen_order"] = [item["config_index"] for item in ordered]
        elif stage == "confirm":
            selection["confirm_winner"] = ordered[0]["config_index"] if ordered else None
            selection["confirm_order"] = [item["config_index"] for item in ordered]
        else:
            selection["stress_evaluated"] = ordered[0]["config_index"] if ordered else None
    progress["updated_at"] = time.time()
    _write(progress_path, progress)
    selection_path = output / f"s1_{stage}_selection.json"
    _write(
        selection_path,
        {
            "protocol": progress["protocol"],
            "stage": stage,
            "methods": all_stage_summaries,
            "selection_order": config["selection_order"],
        },
    )
    return {"status": "S1_STAGE_COMPLETE", "stage": stage, "selection": str(selection_path)}


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    progress = _json(args.output / "s1_progress.json")
    frozen = []
    for method in args.methods or tuple(f"M{index}" for index in range(1, 7)):
        selection = progress.get("selections", {}).get(method, {})
        if selection.get("status") == "S1_FAILED_INVARIANT":
            frozen.append(
                {"method": method, "status": "S1_FAILED_INVARIANT", "selection": selection}
            )
            continue
        winner = selection.get("confirm_winner")
        if winner is None:
            frozen.append({"method": method, "status": "S1_FAILED_ANGLE", "reason": "NO_CONFIRM_WINNER"})
            continue
        summary_path = args.output / "runs" / method / f"config_{int(winner):02d}" / "stage_confirm_summary.json"
        summary = _json(summary_path) if summary_path.is_file() else {}
        count = int(summary.get("sequence_count", 0))
        if summary.get("status") != "COMPLETE" or count == 0:
            status = "S1_FAILED_NUMERICAL"
        elif int(summary.get("angle_pass_count", 0)) < count:
            status = "S1_FAILED_ANGLE"
        elif int(summary.get("numerical_ok_count", 0)) < count:
            status = "S1_FAILED_NUMERICAL"
        elif int(summary.get("physical_v2_pass_count", 0)) < count:
            status = "S1_FAILED_PHYSICAL"
        elif not all(
            row.get("mpjpe_mm", float("inf")) < float("inf")
            and row.get("root_translation_p95_mm", float("inf")) < float("inf")
            for row in summary.get("rows", [])
        ):
            status = "S1_FAILED_CONTENT"
        else:
            status = "S1_PASS"
        frozen.append(
            {
                "method": method,
                "status": status,
                "config_index": winner,
                "confirm_summary": str(summary_path),
                "stress_summary": str(
                    args.output / "runs" / method / f"config_{int(winner):02d}" / "stage_stress_summary.json"
                ),
            }
        )
    result = {
        "protocol": "vimogen_m1_m7_s1_bounded_tuning_v1",
        "status": "S1_FROZEN",
        "s2_allowed": False,
        "methods": frozen,
    }
    _write(args.output / "S1_FROZEN.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "confirm", "stress", "all", "freeze"), default="screen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--m2-invariant", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--noise-cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--thresholds-v2", type=Path, required=True)
    parser.add_argument("--method", dest="methods", action="append", choices=tuple(METHOD_VERSIONS))
    args = parser.parse_args()
    if args.stage == "freeze":
        print(json.dumps(freeze(args), ensure_ascii=False, indent=2))
        return
    stages = ("screen", "confirm", "stress") if args.stage == "all" else (args.stage,)
    result = None
    for stage in stages:
        result = run_stage(args, stage)
    if args.stage == "all":
        result = freeze(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
