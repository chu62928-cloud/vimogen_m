#!/usr/bin/env python3
"""Run and evaluate the staged S1 local-pelvis development matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
OUTPUT = ROOT / "results/phase9/pelvis_m1_m7/local_pelvis_s1"

CENTER = {
    "M1": ("v1", {"guidance_scale": 0.025, "gradient_clip_rms": 1.0, "sigma_min": 0.0662879, "sigma_max": 0.65}),
    "M3": ("v2", {"sigma_min": 0.10, "sigma_max": 0.45, "damping": 1.0e-6, "max_step_deg": 1.0, "projection_stride": 2, "max_projections": 2, "max_endpoint_delta_rms": 0.025}),
    "M4": ("v2", {"shooting_sigmas": [0.15], "gn_iterations": 3, "damping": 1.0e-5, "trust_radius_deg": 4.0, "propagation_gain": 0.5, "terminal_tolerance_deg": 1.0e-4}),
    "M5": ("v1", {"primal_gain": 2.0, "penalty": 0.1, "dual_gain": 1.0, "max_dual_norm": 50.0, "gradient_clip_rms": 1.0, "sigma_min": 0.02}),
    "M6": ("v1", {"candidate_scale": 0.5, "delta": 0.1, "gradient_clip_rms": 1.0, "sigma_min": 0.15, "sigma_max": 0.65}),
}

PRIMARY = {
    "M1": "guidance_scale",
    "M3": "max_step_deg",
    "M4": "trust_radius_deg",
    "M5": "primal_gain",
    "M6": "candidate_scale",
}


def _latest_attempt(method: str, config_id: str, seed: int, dose: float) -> Path:
    parent = OUTPUT / "generation" / method / config_id / f"seed_{seed:03d}" / f"dose_{dose:+g}"
    attempts = sorted(parent.glob("attempt_*"))
    if not attempts:
        raise RuntimeError(f"generation did not create an attempt below {parent}")
    return attempts[-1]


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="append", choices=tuple(CENTER), required=True)
    parser.add_argument("--dose", action="append", type=float, required=True)
    parser.add_argument("--strength", action="append", type=float, default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--code-revision", default="scale-local-pelvis-s1-20260918")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = []
    strengths = args.strength or [1.0]
    if any(value not in {0.5, 1.0, 2.0} for value in strengths):
        parser.error("--strength must be one of 0.5, 1, or 2")
    for method in args.method:
        version, center = CENTER[method]
        for strength in strengths:
            settings = dict(center)
            settings[PRIMARY[method]] = float(center[PRIMARY[method]]) * strength
            strength_token = {0.5: "half", 1.0: "center", 2.0: "double"}[strength]
            config_id = f"s1_{method.lower()}_{strength_token}"
            for dose in args.dose:
                evaluation = OUTPUT / "evaluation" / f"{method}_{config_id}_s{args.seed}_d{dose:+g}"
                completed = sorted(evaluation.glob("attempt_*/summary.json"))
                if args.skip_existing and completed:
                    summary = json.loads(completed[-1].read_text(encoding="utf-8"))
                    rows.append({
                        "method": method,
                        "dose": dose,
                        "strength": strength,
                        "config_id": config_id,
                        "status": summary["status"],
                        "evaluation": str(completed[-1].parent),
                        "reused": True,
                    })
                    continue
                generation = [
                    str(PYTHON), str(ROOT / "experiments/run_sampling_guidance_smoke.py"),
                    "--method", method, "--method-version", version,
                    "--task", "relative_pelvis_s1", "--dose", str(dose),
                    "--seed", str(args.seed), "--code-commit", args.code_revision,
                    "--base-config", str(args.runtime_root / "configs/tm2m_infer.yaml"),
                    "--protocol", str(ROOT / "configs/m1_m7/local_pelvis_pilot_v1.json"),
                    "--runtime-root", str(args.runtime_root),
                    "--manifest", str(ROOT / "configs/m1_m7/local_pelvis_sample94.json"),
                    "--noise-cache", str(args.runtime_root / "results/phase6/absolute_mean_pelvis_v2/noise_cache"),
                    "--output", str(OUTPUT / "generation"),
                    "--settings-json", json.dumps(settings, separators=(",", ":")),
                    "--config-id", config_id, "--experiment-stage", "development",
                ]
                row = {"method": method, "dose": dose, "strength": strength, "config_id": config_id}
                try:
                    _run(generation)
                    attempt = _latest_attempt(method, config_id, args.seed, dose)
                    suffix = 1
                    target = evaluation / f"attempt_{suffix:02d}"
                    while target.exists():
                        suffix += 1
                        target = evaluation / f"attempt_{suffix:02d}"
                    _run([
                        str(PYTHON), str(ROOT / "experiments/evaluate_sampling_guidance_smoke.py"),
                        "--run-root", str(attempt), "--output", str(target),
                    ])
                    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                    row.update(status=summary["status"], generation=str(attempt), evaluation=str(target))
                except Exception as error:
                    row.update(status="IMPLEMENTATION_BLOCKED", error=repr(error))
                rows.append(row)
                progress = {"status": "RUNNING", "rows": rows}
                progress_path = OUTPUT / "development_center_progress.json"
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    status = "COMPLETE" if all(row["status"] != "IMPLEMENTATION_BLOCKED" for row in rows) else "INCOMPLETE"
    result = {"status": status, "rows": rows}
    (OUTPUT / "development_center_progress.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
