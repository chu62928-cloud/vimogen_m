#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vimogen_clean
PY=/root/miniconda3/envs/mdm5090/bin/python
export PYTHONPATH="$ROOT"
cd "$ROOT"
LOG_ROOT=/tmp/v1_1_calibration_logs
mkdir -p "$LOG_ROOT"

# Stage 3: extend the best step-size pair through the final non-zero sigma.
# The actual 50-step scheduler has sigma_last=0.06628788; keep the value in
# the run record and protocol config rather than applying a terminal post-edit.
for window in "0.06628788 0.65" "0.06628788 0.75"; do
  read -r sigma_min sigma_max <<< "$window"
  for delta in -10 -5 5 10; do
    "$PY" scripts/run_relative_root_forward_v1.py \
      --protocol v1_1 \
      --target-delta-deg "$delta" \
      --residual-gain 1.0 \
      --max-step-deg 6 \
      --sigma-min "$sigma_min" \
      --sigma-max "$sigma_max" \
      >"$LOG_ROOT/window_${sigma_min}_${sigma_max}_d${delta}.log" 2>&1
  done
done

echo "window_probe_completed=$(date -Is)"
