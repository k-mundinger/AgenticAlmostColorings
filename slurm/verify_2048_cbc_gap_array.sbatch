#!/bin/bash
#SBATCH --job-name=aac-gap2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
RUN_ID=${RUN_ID:-local_1781965259_557750}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/scratch/htc/npelleriti/agentic-almost-colorings/pipelines/paper_params_grid128_10k_20260620/models/local_1781965259_557750/trained_model.pt}
CONFIG_JSON=${CONFIG_JSON:-/scratch/htc/npelleriti/agentic-almost-colorings/pipelines/paper_params_grid128_10k_20260620/models/local_1781965259_557750/pipeline_config.json}
VERIFY_ROOT=${VERIFY_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/verifications}
EVAL_GRIDSIZE=${EVAL_GRIDSIZE:-2048}
SOLVER_TIME_LIMIT=${SOLVER_TIME_LIMIT:-14400}
SOLVER_THREADS=${SOLVER_THREADS:-1}
ACTIVE_EDGE_CHUNK_SIZE=${ACTIVE_EDGE_CHUNK_SIZE:-64}
MASK_CACHE_DIR=${MASK_CACHE_DIR:-${VERIFY_ROOT}/mask_cache}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    LABEL="gapabs1000"
    GAP_ARGS=(--solver_gap_abs 1000)
    ;;
  1)
    LABEL="gapabs250"
    GAP_ARGS=(--solver_gap_abs 250)
    ;;
  2)
    LABEL="gaprel001"
    GAP_ARGS=(--solver_gap_rel 0.001)
    ;;
  3)
    LABEL="gapabs1000_cutsoff"
    GAP_ARGS=(--solver_gap_abs 1000 --cbc_cuts off)
    ;;
  4)
    LABEL="gapabs1000_cutson"
    GAP_ARGS=(--solver_gap_abs 1000 --cbc_cuts on)
    ;;
  5)
    LABEL="gapabs1000_strong0"
    GAP_ARGS=(--solver_gap_abs 1000 --cbc_strong 0)
    ;;
  *)
    echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT_DIR="${VERIFY_ROOT}/${RUN_ID}_grid${EVAL_GRIDSIZE}_cbc_${LABEL}"
mkdir -p "${OUT_DIR}" "${MASK_CACHE_DIR}" "${VERIFY_ROOT}/mplconfig"

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

echo "label=${LABEL}"
echo "run_id=${RUN_ID}"
echo "output=${OUT_DIR}"
echo "eval_gridsize=${EVAL_GRIDSIZE}"
echo "solver_backend=cbc"
echo "solver_threads=${SOLVER_THREADS}"
echo "solver_time_limit=${SOLVER_TIME_LIMIT}"
echo "gap_args=${GAP_ARGS[*]}"

/usr/bin/time -v "${PYTHON}" scripts/verify_paralellogram_ip.py \
  --run_id "${RUN_ID}" \
  --eval_gridsize "${EVAL_GRIDSIZE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --config_json "${CONFIG_JSON}" \
  --output_dir "${OUT_DIR}" \
  --solver_time_limit "${SOLVER_TIME_LIMIT}" \
  --solver_threads "${SOLVER_THREADS}" \
  --solver_backend cbc \
  --active_edge_chunk_size "${ACTIVE_EDGE_CHUNK_SIZE}" \
  --mask_cache_dir "${MASK_CACHE_DIR}" \
  --skip_vertex_cover \
  --no_plot \
  "${GAP_ARGS[@]}"
