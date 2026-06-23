#!/bin/bash
#SBATCH --job-name=aac-agent2-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
GRID_CSV=${GRID_CSV:-${REPO_ROOT}/constructions/paper_almost_5_coloring_2d/grid.csv}
PARALLELOGRAM_CSV=${PARALLELOGRAM_CSV:-${REPO_ROOT}/constructions/paper_almost_5_coloring_2d/parallelogram.csv}
DIFF_CSV=${DIFF_CSV:-/scratch/htc/npelleriti/pi-sandbox/papers-concept-exploration-loop/job_1319209/worktrees/round_1/agent_2/experiments/paper_local_search/out/full_1945x1970_direct_fill_candidate_diff.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/htc/npelleriti/agentic-almost-colorings/verification/agent2_direct_fill/job_${SLURM_JOB_ID:-manual}}
PYTHON_BIN=${PYTHON_BIN:-python3}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-32}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-32}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-32}
export WANDB_MODE=disabled

mkdir -p "${OUTPUT_DIR}" /scratch/htc/npelleriti/agentic-almost-colorings/slurm
cd "${REPO_ROOT}"

echo "repo=${REPO_ROOT}"
echo "grid_csv=${GRID_CSV}"
echo "parallelogram_csv=${PARALLELOGRAM_CSV}"
echo "diff_csv=${DIFF_CSV}"
echo "output_dir=${OUTPUT_DIR}"
echo "python_bin=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import numpy
print("numpy=" + numpy.__version__)
PY

"${PYTHON_BIN}" scripts/verify_agent2_direct_fill_candidate.py \
  --grid-csv "${GRID_CSV}" \
  --parallelogram-csv "${PARALLELOGRAM_CSV}" \
  --diff-csv "${DIFF_CSV}" \
  --output-dir "${OUTPUT_DIR}"
