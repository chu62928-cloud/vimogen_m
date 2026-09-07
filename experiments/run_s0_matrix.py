"""Run and evaluate the frozen M1--M7 S0 matrix sequentially.

The matrix is deliberately resumable.  A completed attempt with the same
method, seed, dose and settings is reused; failed attempts are never removed
and a fresh attempt is created on the next invocation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_sampling_guidance_smoke.py"
EVALUATOR = ROOT / "experiments" / "evaluate_sampling_guidance_smoke.py"
OUTPUT = ROOT / "results" / "phase9" / "pelvis_m1_m7" / "s0_sampling"

METHOD_SETTINGS = {
    "M1": {"guidance_scale": 0.02, "gradient_clip_rms": 1.0, "sigma_min": 0.0662879, "sigma_max": 0.65},
    "M2": {"learning_rate": 0.005, "iterations": 2, "source_regularization": 0.001, "gradient_clip_norm": 10.0},
    "M3": {"damping": 1.0e-6, "max_step_deg": 2.0, "sigma_min": 0.0662879, "sigma_max": 0.65},
    "M4": {"shooting_sigmas": [0.55, 0.35, 0.15], "gn_iterations": 3, "damping": 1.0e-5, "trust_radius_deg": 4.0, "propagation_gain": 1.0, "terminal_tolerance_deg": 1.0e-4},
    "M5": {"primal_gain": 0.5, "penalty": 0.1, "dual_gain": 1.0, "max_dual_norm": 50.0, "gradient_clip_rms": 1.0},
    "M6": {"candidate_scale": 0.5, "delta": 0.1, "gradient_clip_rms": 1.0, "sigma_min": 0.0662879, "sigma_max": 0.65},
}

JOBS = [(method, seed, dose) for method in METHOD_SETTINGS for seed in (0, 42) for dose in (-2.0, 0.0, 2.0)]


def _same_settings(left: object, right: dict) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _find_reusable(method: str, seed: int, dose: float) -> Path | None:
    parent = OUTPUT / method / f"seed_{seed:03d}" / f"dose_{dose:+g}"
    if not parent.is_dir():
        return None
    for attempt in sorted(parent.glob("attempt_*"), reverse=True):
        record_path = attempt / "run_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("status", "").startswith("COMPLETED")
            and int(record.get("seed", -1)) == seed
            and float(record.get("target_dose_deg", 999.0)) == dose
            and _same_settings(record.get("settings", {}), METHOD_SETTINGS[method])
        ):
            return attempt
    return None


def _evaluate(attempt: Path) -> dict:
    evaluation = attempt / "evaluation"
    summary = evaluation / "summary.json"
    if summary.is_file():
        try:
            return json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"status": "EVALUATION_OUTPUT_INVALID", "path": str(summary)}
    command = [sys.executable, str(EVALUATOR), "--run-root", str(attempt), "--output", str(evaluation)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return {"status": "EVALUATION_FAILED", "returncode": completed.returncode, "stderr": completed.stderr[-2000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "EVALUATION_OUTPUT_INVALID", "stdout": completed.stdout[-2000:]}


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means all remaining jobs")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--noise-cache", type=Path)
    args = parser.parse_args()

    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrix_record = OUTPUT / "s0_matrix_progress.json"
    progress = {"protocol": "vimogen_pelvis_m1_m7_scale_v1", "jobs": [], "updated_at": time.time()}
    if matrix_record.is_file():
        try:
            progress = json.loads(matrix_record.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    completed_keys = {
        tuple(item.get(key) for key in ("method", "seed", "dose"))
        for item in progress.get("jobs", [])
        if item.get("evaluation", {}).get("status") not in {
            "EVALUATION_FAILED", "EVALUATION_OUTPUT_INVALID"
        }
    }
    selected_jobs = [job for job in JOBS if job not in completed_keys]
    if args.max_jobs > 0:
        selected_jobs = selected_jobs[: args.max_jobs]

    for method, seed, dose in selected_jobs:
        started = time.time()
        attempt = _find_reusable(method, seed, dose)
        generation = {"status": "REUSED"} if attempt is not None else {"status": "LAUNCHED"}
        if attempt is None:
            command = [sys.executable, str(RUNNER), "--method", method, "--dose", str(dose), "--seed", str(seed), "--code-commit", args.code_commit, "--output", str(OUTPUT), "--settings-json", json.dumps(METHOD_SETTINGS[method], separators=(",", ":"))]
            if args.manifest is not None:
                command.extend(["--manifest", str(args.manifest)])
            if args.noise_cache is not None:
                command.extend(["--noise-cache", str(args.noise_cache)])
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            generation["returncode"] = completed.returncode
            generation["stdout_tail"] = completed.stdout[-2000:]
            generation["stderr_tail"] = completed.stderr[-2000:]
            parent = OUTPUT / method / f"seed_{seed:03d}" / f"dose_{dose:+g}"
            attempts = sorted(parent.glob("attempt_*")) if parent.is_dir() else []
            attempt = attempts[-1] if attempts else None
        evaluation = _evaluate(attempt) if attempt is not None and (attempt / "trainer").exists() else {"status": "NOT_EVALUATED"}
        item = {"method": method, "seed": seed, "dose": dose, "attempt": str(attempt) if attempt else None, "generation": generation, "evaluation": evaluation, "elapsed_seconds": time.time() - started}
        progress.setdefault("jobs", []).append(item)
        progress["updated_at"] = time.time()
        matrix_record.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(item, ensure_ascii=False))

    # A repair pass appends a corrected evaluation record for a failed key so
    # the original failure remains auditable.  Count unique non-failed keys
    # here instead of raw history entries, otherwise the progress can become
    # negative after such a repair pass.
    completed_keys = {
        tuple(item.get(key) for key in ("method", "seed", "dose"))
        for item in progress.get("jobs", [])
        if item.get("evaluation", {}).get("status") not in {
            "EVALUATION_FAILED", "EVALUATION_OUTPUT_INVALID"
        }
    }
    progress["completed_unique_jobs"] = len(completed_keys)
    progress["remaining_jobs"] = len(JOBS) - len(completed_keys)
    progress["updated_at"] = time.time()
    matrix_record.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "MATRIX_PROGRESS", "completed_jobs": len(progress.get("jobs", [])), "total_jobs": len(JOBS), "remaining_jobs": progress["remaining_jobs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
