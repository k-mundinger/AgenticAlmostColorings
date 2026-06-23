#!/bin/bash
#SBATCH --job-name=aac2048-rect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%8
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
CANDIDATES_CSV=${CANDIDATES_CSV:-${REPO_ROOT}/slurm/top_rect_2048_candidates.csv}
VERIFY_ROOT=${VERIFY_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/verifications/top_rect_2048_cbc30m}
MASK_CACHE_DIR=${MASK_CACHE_DIR:-/scratch/htc/npelleriti/agentic-almost-colorings/verifications/mask_cache}
BASE_EVAL_GRIDSIZE=${BASE_EVAL_GRIDSIZE:-2048}
SOLVER_THREADS=${SOLVER_THREADS:-1}
SOLVER_TIME_LIMIT=${SOLVER_TIME_LIMIT:-1700}
ACTIVE_EDGE_CHUNK_SIZE=${ACTIVE_EDGE_CHUNK_SIZE:-64}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

cd "${REPO_ROOT}"
export WANDB_MODE=disabled
export MPLCONFIGDIR="${VERIFY_ROOT}/mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

read -r RANK SCORE MODEL_INDEX RUN_ID RUN_DIR GRID_HEIGHT GRID_WIDTH CELLS < <(
  "${PYTHON}" -c '
import csv
import sys

candidate_path = sys.argv[1]
task_id = int(sys.argv[2])
with open(candidate_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not (0 <= task_id < len(rows)):
    raise SystemExit(f"task_id {task_id} out of range for {len(rows)} rows")
row = rows[task_id]
print(
    row["rank"],
    row["score"],
    row["model_index"],
    row["run_id"],
    row["run_dir"],
    row["grid_height"],
    row["grid_width"],
    row["cells"],
)
' "${CANDIDATES_CSV}" "${TASK_ID}"
)

CHECKPOINT_PATH="${RUN_DIR}/trained_model.pt"
CONFIG_JSON="${RUN_DIR}/pipeline_config.json"
OUT_DIR="${VERIFY_ROOT}/rank${RANK}_m${MODEL_INDEX}_${GRID_HEIGHT}x${GRID_WIDTH}_score${SCORE}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID:-local}"
echo "array_task_id=${TASK_ID}"
echo "rank=${RANK}"
echo "square_score=${SCORE}"
echo "model_index=${MODEL_INDEX}"
echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "config=${CONFIG_JSON}"
echo "grid_height=${GRID_HEIGHT}"
echo "grid_width=${GRID_WIDTH}"
echo "cells=${CELLS}"
echo "output=${OUT_DIR}"
echo "solver_backend=cbc"
echo "solver_threads=${SOLVER_THREADS}"
echo "solver_time_limit=${SOLVER_TIME_LIMIT}"
echo "mask_cache_dir=${MASK_CACHE_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

mkdir -p "${OUT_DIR}" "${MASK_CACHE_DIR}" "${VERIFY_ROOT}/mplconfig"
test -f "${CHECKPOINT_PATH}"
test -f "${CONFIG_JSON}"

/usr/bin/time -v "${PYTHON}" scripts/verify_paralellogram_ip_rect.py \
  --run_id "${RUN_ID}" \
  --eval_gridsize "${BASE_EVAL_GRIDSIZE}" \
  --eval_gridheight "${GRID_HEIGHT}" \
  --eval_gridwidth "${GRID_WIDTH}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --config_json "${CONFIG_JSON}" \
  --output_dir "${OUT_DIR}" \
  --solver_time_limit "${SOLVER_TIME_LIMIT}" \
  --solver_threads "${SOLVER_THREADS}" \
  --solver_backend cbc \
  --active_edge_chunk_size "${ACTIVE_EDGE_CHUNK_SIZE}" \
  --mask_cache_dir "${MASK_CACHE_DIR}" \
  --skip_vertex_cover \
  --no_plot
