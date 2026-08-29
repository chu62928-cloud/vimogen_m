"""Run the v1.2 seed/dose matrix without averaging or overwriting runs.

This script is intended to be executed on the model server.  It invokes the
single-run runner in a separate process for every seed and dose, reusing the
sample-noise-v1 cache.  Each invocation creates a new attempt directory, so
failed attempts remain available as calibration evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_SEEDS = (0, 42, 464229750, 1057660199, 1386772747)
DEFAULT_DELTAS = (-10.0, -5.0, 5.0, 10.0)


def _format_delta(value: float) -> str:
    return f"{value:+g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--deltas", nargs="+", type=float, default=list(DEFAULT_DELTAS))
    parser.add_argument("--noise-cache", type=Path, required=True)
    parser.add_argument("--sigma-min", type=float, default=0.066)
    parser.add_argument("--sigma-max", type=float, default=0.75)
    parser.add_argument("--residual-gain", type=float, default=1.5)
    parser.add_argument("--max-step-deg", type=float, default=8.0)
    parser.add_argument("--heading-gain", type=float, default=0.75)
    parser.add_argument("--max-heading-step-deg", type=float, default=2.0)
    parser.add_argument("--trunk-gain", type=float, default=0.75)
    parser.add_argument("--max-trunk-step-deg", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runner = Path(__file__).with_name("run_relative_root_forward_v1.py")
    rows = []
    for seed in args.seeds:
        for delta in args.deltas:
            command = [
                sys.executable,
                str(runner),
                "--protocol", "v1_2",
                "--target-delta-deg", str(delta),
                "--seed", str(seed),
                "--noise-cache", str(args.noise_cache),
                "--batch-size", str(args.batch_size),
                "--sigma-min", str(args.sigma_min),
                "--sigma-max", str(args.sigma_max),
                "--residual-gain", str(args.residual_gain),
                "--max-step-deg", str(args.max_step_deg),
                "--heading-gain", str(args.heading_gain),
                "--max-heading-step-deg", str(args.max_heading_step_deg),
                "--trunk-gain", str(args.trunk_gain),
                "--max-trunk-step-deg", str(args.max_trunk_step_deg),
            ]
            row = {"seed": seed, "target_delta_deg": delta, "command": command}
            if args.skip_existing:
                # The runner's parameter key includes all calibration values;
                # only skip when a completed artifact is already present.
                parameter = (
                    f"pitch_{args.residual_gain:g}_pstep_{args.max_step_deg:g}"
                    f"_heading_{args.heading_gain:g}_hstep_{args.max_heading_step_deg:g}"
                    f"_trunk_{args.trunk_gain:g}_tstep_{args.max_trunk_step_deg:g}"
                    f"_sigma_{args.sigma_min:g}_to_{args.sigma_max:g}"
                )
                sign = "+" if delta >= 0 else ""
                root = runner.parents[1] / "results" / "phase7" / "relative_root_forward_v1_2" / "runs" / "smoke" / f"seed_{seed:03d}" / parameter / f"delta_{sign}{delta:g}deg"
                if list(root.glob("attempt_*/guided_artifacts/batch_000/g0_norm_batch.pt")):
                    row.update({"status": "SKIPPED_EXISTING", "run_root": str(root)})
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False))
                    continue
            started = time.perf_counter()
            completed = subprocess.run(command, check=False)
            row.update(
                {
                    "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)

    output = args.output or Path("results/phase7/relative_root_forward_v1_2/summaries/multiseed_runner.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol": "vimogen_relative_root_forward_v1_2_trunk_stabilized",
                "seeds": list(args.seeds),
                "deltas": list(args.deltas),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
