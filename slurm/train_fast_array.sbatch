#!/usr/bin/env bash
#SBATCH --job-name=aac-fast-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.err

set -euo pipefail

REPO="${REPO:-/home/htc/npelleriti/AgenticAlmostColorings}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/sweeps}"
SWEEP_ID="${SWEEP_ID:-sweep_${SLURM_ARRAY_JOB_ID:-local}}"
RUN_ROOT="${SCRATCH_ROOT}/${SWEEP_ID}"

LOCAL_PARALLEL="${LOCAL_PARALLEL:-4}"
N_STEPS="${N_STEPS:-75000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
N_CIRCLE_POINTS="${N_CIRCLE_POINTS:-8}"
LOSS_LOG_EVERY="${LOSS_LOG_EVERY:-1000}"
BASE_SEED="${BASE_SEED:-1000000}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_CACHE_DIR
export WANDB_MODE=disabled
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs"
cd "${REPO}"

if [[ -x "${REPO}/.venv/bin/python" ]]; then
  PYTHON="${REPO}/.venv/bin/python"
else
  PYTHON="python"
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
job_id="${SLURM_ARRAY_JOB_ID:-local}"

echo "job_id=${job_id} task_id=${task_id} host=$(hostname)"
echo "run_root=${RUN_ROOT}"
echo "local_parallel=${LOCAL_PARALLEL} n_steps=${N_STEPS} batch_size=${BATCH_SIZE}"

pids=()
for local_rank in $(seq 0 "$((LOCAL_PARALLEL - 1))"); do
  seed="$((BASE_SEED + task_id * LOCAL_PARALLEL + local_rank))"
  run_name="job_${job_id}_task_${task_id}_rank_${local_rank}_seed_${seed}"
  log_path="${RUN_ROOT}/logs/${run_name}.log"

  echo "starting ${run_name}"
  "${PYTHON}" main.py \
    --debug \
    --fast-train \
    --seed "${seed}" \
    --local-run-id "${run_name}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --n-circle-points "${N_CIRCLE_POINTS}" \
    --loss-log-every "${LOSS_LOG_EVERY}" \
    --output-root "${RUN_ROOT}" \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON}" scripts/summarize_training_sweep.py "${RUN_ROOT}" \
  --output-csv "${RUN_ROOT}/summary.csv" \
  | tee "${RUN_ROOT}/logs/summary_task_${task_id}.txt"

exit "${status}"
