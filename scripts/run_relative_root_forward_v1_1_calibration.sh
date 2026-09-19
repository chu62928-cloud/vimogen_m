#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vimogen_clean
PY=/root/miniconda3/envs/mdm5090/bin/python
export PYTHONPATH="$ROOT"
cd "$ROOT"

LOG_ROOT=/tmp/v1_1_calibration_logs
mkdir -p "$LOG_ROOT"

# Stage 1: fixed sigma window, positive doses.  Each tuple is
# residual_gain max_step_deg.  Existing completed attempts are skipped by the
# runner's non-overwriting policy.
for spec in \
  "0.5 4" \
  "0.75 4" \
  "1.0 4" \
  "0.75 6" \
  "1.0 6" \
  "1.0 8"; do
  read -r gain step <<< "$spec"
  key="gain_${gain}_step_${step}"
  for delta in 5 10; do
    log="$LOG_ROOT/${key}_p${delta}.log"
    record_glob="$ROOT/results/phase7/relative_root_forward_v1_1/runs/smoke/seed_000/${key}/delta_$(printf '%+g' "$delta")deg/attempt_*/run_record.json"
    latest_record=$(find "$ROOT/results/phase7/relative_root_forward_v1_1/runs/smoke/seed_000/${key}/delta_$(printf '%+g' "$delta")deg" -name run_record.json -type f 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$latest_record" ]] && grep -q '"status": "COMPLETED_GENERATION_PENDING_EVALUATION"' "$latest_record"; then
      echo "skip completed $key delta=$delta"
      continue
    fi
    "$PY" scripts/run_relative_root_forward_v1.py \
      --protocol v1_1 \
      --target-delta-deg "$delta" \
      --residual-gain "$gain" \
      --max-step-deg "$step" \
      --sigma-min 0.25 \
      --sigma-max 0.65 \
      >"$log" 2>&1
  done
done

echo "stage1_completed=$(date -Is)"
