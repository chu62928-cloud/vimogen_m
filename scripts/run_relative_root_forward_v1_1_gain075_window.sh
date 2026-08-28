#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vimogen_clean
PY=/root/miniconda3/envs/mdm5090/bin/python
export PYTHONPATH="$ROOT"
cd "$ROOT"
LOG_ROOT=/tmp/v1_1_calibration_logs
mkdir -p "$LOG_ROOT"

for delta in -10 -5 5 10; do
  "$PY" scripts/run_relative_root_forward_v1.py \
    --protocol v1_1 \
    --target-delta-deg "$delta" \
    --residual-gain 0.75 \
    --max-step-deg 6 \
    --sigma-min 0.06628788 \
    --sigma-max 0.65 \
    >"$LOG_ROOT/gain_0.75_step_6_sigma_0.0662879_to_0.65_d${delta}.log" 2>&1
done

echo "gain075_window_completed=$(date -Is)"
