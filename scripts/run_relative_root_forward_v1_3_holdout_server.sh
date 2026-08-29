#!/usr/bin/env bash
# Server-side holdout runner for the frozen v1.3 parameter set.
# All generated artifacts stay under the protocol-specific phase-7 directory.

set -u

PYTHON=/root/miniconda3/envs/mdm5090/bin/python
SEEDS=(464229750 1057660199 1386772747)
DELTAS=(5 10 -5 -10)
PORT=29601

for seed in "${SEEDS[@]}"; do
  for delta in "${DELTAS[@]}"; do
    echo "[v1.3 holdout] seed=${seed} delta=${delta} port=${PORT}"
    success=0
    for retry in 0 1 2; do
      MASTER_PORT=${PORT} "${PYTHON}" scripts/run_relative_root_forward_v1.py \
        --protocol v1_3 \
        --target-delta-deg "${delta}" \
        --seed "${seed}" \
        --residual-gain 1.0 \
        --max-step-deg 6 \
        --sigma-min 0.0662879 \
        --sigma-max 0.65 \
        --heading-gain 0.75 \
        --max-heading-step-deg 2 \
        --trunk-gain 0.75 \
        --max-trunk-step-deg 6 && { success=1; break; }
      echo "[v1.3 holdout] retry=${retry} failed; changing port"
      PORT=$((PORT + 1))
    done
    if [[ "${success}" -ne 1 ]]; then
      echo "[v1.3 holdout] FAILED seed=${seed} delta=${delta}" >&2
    fi
    PORT=$((PORT + 1))
  done
done

echo "[v1.3 holdout] finished"
