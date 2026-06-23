#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
run_root=${RUN_ROOT:-/tmp/aac_cpu_baseline_bg_10k_grid64}
run_id=${RUN_ID:-cpu_baseline_bg_10k_grid64}

mkdir -p "${run_root}/models" "${run_root}/verifications"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

exec .venv/bin/python -u run_pipeline.py \
  --train-command ".venv/bin/python -u main.py --debug --fast-train --n-steps ${N_STEPS:-10000} --batch-size ${BATCH_SIZE:-2048} --n-circle-points ${N_CIRCLE_POINTS:-8} --loss-log-every ${LOSS_LOG_EVERY:-1000} --output-root ${run_root}/models --local-run-id ${run_id}" \
  --eval-gridsize "${EVAL_GRIDSIZE:-64}" \
  --verify-output-root "${run_root}/verifications" \
  --solver-time-limit "${SOLVER_TIME_LIMIT:-3600}" \
  --no-plot
