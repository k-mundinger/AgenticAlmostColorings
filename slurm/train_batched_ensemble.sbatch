#!/usr/bin/env bash
#SBATCH --job-name=aac-ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.err

set -euo pipefail

REPO="${REPO:-/home/htc/npelleriti/AgenticAlmostColorings}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/ensembles}"
SWEEP_PREFIX="${SWEEP_PREFIX:-batched_ensemble}"

ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-32}"
N_STEPS="${N_STEPS:-75000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
N_CIRCLE_POINTS="${N_CIRCLE_POINTS:-8}"
LOSS_LOG_EVERY="${LOSS_LOG_EVERY:-1000}"
BASE_SEED="${BASE_SEED:-3000000}"

LEARNING_RATES="${LEARNING_RATES:-0.001,0.0005,0.002}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.1,0.03}"
GOOD_COLORING_WEIGHTS="${GOOD_COLORING_WEIGHTS:-0.005,0.01,0.02}"
TEMPERATURES="${TEMPERATURES:-0.0,-1.0}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_CACHE_DIR
export WANDB_MODE=disabled
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

cd "${REPO}"

if [[ -x "${REPO}/.venv/bin/python" ]]; then
  PYTHON="${REPO}/.venv/bin/python"
else
  PYTHON="python"
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
job_id="${SLURM_ARRAY_JOB_ID:-local}"
sweep_id="${SWEEP_PREFIX}_${job_id}_task_${task_id}"
task_base_seed="$((BASE_SEED + task_id * ENSEMBLE_SIZE))"

echo "job_id=${job_id} task_id=${task_id} host=$(hostname)"
echo "sweep_id=${sweep_id} ensemble_size=${ENSEMBLE_SIZE} base_seed=${task_base_seed}"
echo "n_steps=${N_STEPS} batch_size=${BATCH_SIZE} n_circle_points=${N_CIRCLE_POINTS}"
echo "learning_rates=${LEARNING_RATES}"
echo "weight_decays=${WEIGHT_DECAYS}"
echo "good_coloring_weights=${GOOD_COLORING_WEIGHTS}"
echo "temperatures=${TEMPERATURES}"

"${PYTHON}" scripts/train_batched_ensemble.py \
  --ensemble-size "${ENSEMBLE_SIZE}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --n-circle-points "${N_CIRCLE_POINTS}" \
  --loss-log-every "${LOSS_LOG_EVERY}" \
  --output-root "${OUTPUT_ROOT}" \
  --sweep-id "${sweep_id}" \
  --base-seed "${task_base_seed}" \
  --learning-rates "${LEARNING_RATES}" \
  --weight-decays "${WEIGHT_DECAYS}" \
  --good-coloring-weights "${GOOD_COLORING_WEIGHTS}" \
  --temperatures "${TEMPERATURES}"
