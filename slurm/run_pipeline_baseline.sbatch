#!/bin/bash
#SBATCH --job-name=aac-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/pipelines}
PIPELINE_ID=${PIPELINE_ID:-baseline_gw001_${SLURM_JOB_ID}}
EVAL_GRIDSIZE=${EVAL_GRIDSIZE:-64}
N_STEPS=${N_STEPS:-75000}
BATCH_SIZE=${BATCH_SIZE:-2048}
N_CIRCLE_POINTS=${N_CIRCLE_POINTS:-8}
LOSS_LOG_EVERY=${LOSS_LOG_EVERY:-1000}
SOLVER_TIME_LIMIT=${SOLVER_TIME_LIMIT:-7200}
SOLVER_THREADS=${SOLVER_THREADS:-1}
SOLVER_BACKEND=${SOLVER_BACKEND:-cbc}
SCIP_PATH=${SCIP_PATH:-/software/opt-sw/scipoptsuite-10.0.2/bin/scip}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
SKIP_VERTEX_COVER=${SKIP_VERTEX_COVER:-1}
ACTIVE_EDGE_CHUNK_SIZE=${ACTIVE_EDGE_CHUNK_SIZE:-64}

run_root="${SCRATCH_ROOT}/${PIPELINE_ID}"
model_root="${run_root}/models"
verify_root="${run_root}/verifications"
mask_cache_root="${run_root}/mask_cache"
mkdir -p "${model_root}" "${verify_root}" "${mask_cache_root}"

cd "${REPO_ROOT}"
export WANDB_MODE=disabled
export MPLCONFIGDIR="${run_root}/mplconfig"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

train_command="python -u main.py --debug --fast-train --n-steps ${N_STEPS} --batch-size ${BATCH_SIZE} --n-circle-points ${N_CIRCLE_POINTS} --loss-log-every ${LOSS_LOG_EVERY} --output-root ${model_root} ${EXTRA_TRAIN_ARGS}"

echo "Pipeline id: ${PIPELINE_ID}"
echo "Run root: ${run_root}"
echo "Train command: ${train_command}"
echo "Eval gridsize: ${EVAL_GRIDSIZE}"
echo "Skip vertex cover: ${SKIP_VERTEX_COVER}"
echo "Active edge chunk size: ${ACTIVE_EDGE_CHUNK_SIZE}"
echo "Solver threads: ${SOLVER_THREADS}"
echo "Solver backend: ${SOLVER_BACKEND}"

pipeline_args=(
  --train-command "${train_command}" \
  --eval-gridsize "${EVAL_GRIDSIZE}" \
  --verify-output-root "${verify_root}" \
  --solver-time-limit "${SOLVER_TIME_LIMIT}" \
  --solver-threads "${SOLVER_THREADS}" \
  --solver-backend "${SOLVER_BACKEND}" \
  --scip-path "${SCIP_PATH}" \
  --mask-cache-dir "${mask_cache_root}" \
  --active-edge-chunk-size "${ACTIVE_EDGE_CHUNK_SIZE}" \
  --no-plot
)

if [[ "${SKIP_VERTEX_COVER}" == "1" ]]; then
  pipeline_args+=(--skip-vertex-cover)
fi

python -u run_pipeline.py "${pipeline_args[@]}"
