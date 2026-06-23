#!/bin/bash
#SBATCH --job-name=aac-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
CANDIDATES_CSV=${CANDIDATES_CSV:?Set CANDIDATES_CSV to a CSV with run_id,checkpoint_path,config_json}
EVAL_GRIDSIZE=${EVAL_GRIDSIZE:-64}
VERIFY_ROOT=${VERIFY_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/verifications}
SOLVER_TIME_LIMIT=${SOLVER_TIME_LIMIT:-7200}
SOLVER_THREADS=${SOLVER_THREADS:-1}
SOLVER_BACKEND=${SOLVER_BACKEND:-cbc}
SCIP_PATH=${SCIP_PATH:-/software/opt-sw/scipoptsuite-10.0.2/bin/scip}
NO_PLOT=${NO_PLOT:-1}
SKIP_VERTEX_COVER=${SKIP_VERTEX_COVER:-1}
NEIGHBOR_BACKEND=${NEIGHBOR_BACKEND:-fft}
ACTIVE_EDGE_CHUNK_SIZE=${ACTIVE_EDGE_CHUNK_SIZE:-64}
MASK_CACHE_DIR=${MASK_CACHE_DIR:-${VERIFY_ROOT}/mask_cache}

line=$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR == idx + 2 { print; exit }' "${CANDIDATES_CSV}")
if [[ -z "${line}" ]]; then
  echo "No candidate for array index ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

IFS=, read -r run_id checkpoint_path config_json <<< "${line}"
out_dir="${VERIFY_ROOT}/${run_id}_grid${EVAL_GRIDSIZE}"
mkdir -p "${out_dir}"
mkdir -p "${MASK_CACHE_DIR}"

cd "${REPO_ROOT}"
export WANDB_MODE=disabled
export MPLCONFIGDIR="${VERIFY_ROOT}/mplconfig"

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python3
fi

args=(
  scripts/verify_paralellogram_ip.py
  --run_id "${run_id}"
  --eval_gridsize "${EVAL_GRIDSIZE}"
  --checkpoint_path "${checkpoint_path}"
  --config_json "${config_json}"
  --output_dir "${out_dir}"
  --solver_time_limit "${SOLVER_TIME_LIMIT}"
  --solver_threads "${SOLVER_THREADS}"
  --solver_backend "${SOLVER_BACKEND}"
  --scip_path "${SCIP_PATH}"
  --neighbor_backend "${NEIGHBOR_BACKEND}"
  --active_edge_chunk_size "${ACTIVE_EDGE_CHUNK_SIZE}"
  --mask_cache_dir "${MASK_CACHE_DIR}"
)

if [[ "${NO_PLOT}" == "1" ]]; then
  args+=(--no_plot)
fi

if [[ "${SKIP_VERTEX_COVER}" == "1" ]]; then
  args+=(--skip_vertex_cover)
fi

echo "Verifying ${run_id}"
echo "checkpoint=${checkpoint_path}"
echo "config=${config_json}"
echo "output=${out_dir}"
echo "eval_gridsize=${EVAL_GRIDSIZE}"
echo "skip_vertex_cover=${SKIP_VERTEX_COVER}"
echo "solver_threads=${SOLVER_THREADS}"
echo "solver_backend=${SOLVER_BACKEND}"
echo "neighbor_backend=${NEIGHBOR_BACKEND}"
echo "active_edge_chunk_size=${ACTIVE_EDGE_CHUNK_SIZE}"
echo "mask_cache_dir=${MASK_CACHE_DIR}"

"${PYTHON}" "${args[@]}"
