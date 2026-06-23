#!/bin/bash
#SBATCH --job-name=aac-heur4096
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
RUN_ID=${RUN_ID:-local_1781965259_557750}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/scratch/htc/npelleriti/agentic-almost-colorings/pipelines/paper_params_grid128_10k_20260620/models/local_1781965259_557750/trained_model.pt}
CONFIG_JSON=${CONFIG_JSON:-/scratch/htc/npelleriti/agentic-almost-colorings/pipelines/paper_params_grid128_10k_20260620/models/local_1781965259_557750/pipeline_config.json}
VERIFY_ROOT=${VERIFY_ROOT:-/scratch/htc/npelleriti/agentic-almost-colorings/verifications}
EVAL_GRIDSIZE=${EVAL_GRIDSIZE:-4096}
PASSES=${PASSES:-8}
RESTARTS=${RESTARTS:-4}
SEED=${SEED:-17}
CANDIDATE_FRACTIONS=${CANDIDATE_FRACTIONS:-1.0,0.75,0.5,0.25,0.125,0.0625}
MASK_CACHE_DIR=${MASK_CACHE_DIR:-${VERIFY_ROOT}/mask_cache}

OUT_DIR="${VERIFY_ROOT}/${RUN_ID}_grid${EVAL_GRIDSIZE}_heuristic_p${PASSES}_r${RESTARTS}_seed${SEED}"
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

echo "run_id=${RUN_ID}"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "config=${CONFIG_JSON}"
echo "output=${OUT_DIR}"
echo "eval_gridsize=${EVAL_GRIDSIZE}"
echo "passes=${PASSES}"
echo "restarts=${RESTARTS}"
echo "seed=${SEED}"
echo "candidate_fractions=${CANDIDATE_FRACTIONS}"
echo "mask_cache_dir=${MASK_CACHE_DIR}"

/usr/bin/time -v "${PYTHON}" scripts/verify_parallelogram_heuristic.py \
  --run_id "${RUN_ID}" \
  --eval_gridsize "${EVAL_GRIDSIZE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --config_json "${CONFIG_JSON}" \
  --output_dir "${OUT_DIR}" \
  --mask_cache_dir "${MASK_CACHE_DIR}" \
  --passes "${PASSES}" \
  --restarts "${RESTARTS}" \
  --candidate_fractions "${CANDIDATE_FRACTIONS}" \
  --seed "${SEED}" \
  --device cpu \
  --no_plot
