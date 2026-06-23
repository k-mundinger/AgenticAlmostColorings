#!/usr/bin/env bash
#SBATCH --job-name=aac-best128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.err

set -euo pipefail

REPO="${REPO:-/home/htc/npelleriti/AgenticAlmostColorings}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/ensembles}"
SWEEP_ID="${SWEEP_ID:-best128_paper_params_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"

ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-128}"
N_STEPS="${N_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
N_CIRCLE_POINTS="${N_CIRCLE_POINTS:-8}"
LOSS_LOG_EVERY="${LOSS_LOG_EVERY:-500}"
BASE_SEED="${BASE_SEED:-7000000}"

LEARNING_RATES="${LEARNING_RATES:-0.001}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.0}"
GOOD_COLORING_WEIGHTS="${GOOD_COLORING_WEIGHTS:-0.05}"
TEMPERATURES="${TEMPERATURES:-5.0}"

N_HIDDEN_LAYERS="${N_HIDDEN_LAYERS:-4}"
N_HIDDEN_UNITS="${N_HIDDEN_UNITS:-32}"
ACTIVATION="${ACTIVATION:-sin}"
INITIALIZATION="${INITIALIZATION:-siren}"
GRID_SIZE="${GRID_SIZE:-6}"
GRID_INPUT_SCALE="${GRID_INPUT_SCALE:-1}"
P_NORM="${P_NORM:-2}"

# Final trained parallelogram from the best/paper-parameter single-model run we verified.
PARALLELOGRAM="${PARALLELOGRAM:-1.8744617700576782,0.4844883978366852;0.5359286665916443,1.8974430561065674}"

export WANDB_MODE=disabled
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

cd "${REPO}"

if [[ -x "${REPO}/.venv/bin/python" ]]; then
  PYTHON="${REPO}/.venv/bin/python"
else
  PYTHON="python"
fi

echo "host=$(hostname)"
echo "sweep_id=${SWEEP_ID}"
echo "output_root=${OUTPUT_ROOT}"
echo "ensemble_size=${ENSEMBLE_SIZE}"
echo "n_steps=${N_STEPS}"
echo "batch_size=${BATCH_SIZE}"
echo "n_circle_points=${N_CIRCLE_POINTS}"
echo "base_seed=${BASE_SEED}"
echo "learning_rates=${LEARNING_RATES}"
echo "weight_decays=${WEIGHT_DECAYS}"
echo "good_coloring_weights=${GOOD_COLORING_WEIGHTS}"
echo "temperatures=${TEMPERATURES}"
echo "n_hidden_layers=${N_HIDDEN_LAYERS}"
echo "n_hidden_units=${N_HIDDEN_UNITS}"
echo "activation=${ACTIVATION}"
echo "initialization=${INITIALIZATION}"
echo "parallelogram=${PARALLELOGRAM}"

/usr/bin/time -v "${PYTHON}" scripts/train_batched_ensemble.py \
  --ensemble-size "${ENSEMBLE_SIZE}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --n-circle-points "${N_CIRCLE_POINTS}" \
  --loss-log-every "${LOSS_LOG_EVERY}" \
  --output-root "${OUTPUT_ROOT}" \
  --sweep-id "${SWEEP_ID}" \
  --base-seed "${BASE_SEED}" \
  --learning-rates "${LEARNING_RATES}" \
  --weight-decays "${WEIGHT_DECAYS}" \
  --good-coloring-weights "${GOOD_COLORING_WEIGHTS}" \
  --temperatures "${TEMPERATURES}" \
  --grid-size "${GRID_SIZE}" \
  --grid-input-scale "${GRID_INPUT_SCALE}" \
  --p-norm "${P_NORM}" \
  --parallelogram "${PARALLELOGRAM}" \
  --n-hidden-layers "${N_HIDDEN_LAYERS}" \
  --n-hidden-units "${N_HIDDEN_UNITS}" \
  --activation "${ACTIVATION}" \
  --initialization "${INITIALIZATION}"
