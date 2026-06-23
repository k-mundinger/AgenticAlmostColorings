#!/bin/bash
#SBATCH --job-name=aac-heur4096-kick
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
PASSES=${PASSES:-2}
RESTARTS=${RESTARTS:-4}
SEED=${SEED:-17}
CANDIDATE_FRACTIONS=${CANDIDATE_FRACTIONS:-1.0,0.75}
FILL_STRATEGY=${FILL_STRATEGY:-mis}
MIS_EDGE_CHUNK_SIZE=${MIS_EDGE_CHUNK_SIZE:-32}
MIS_MAX_ROUNDS=${MIS_MAX_ROUNDS:-64}
AUGMENT_ROUNDS=${AUGMENT_ROUNDS:-2}
KICK_ROUNDS=${KICK_ROUNDS:-2}
AUGMENT_MAX_PROPOSALS_PER_COLOR=${AUGMENT_MAX_PROPOSALS_PER_COLOR:-200000}
KICK_MAX_PROPOSALS_PER_COLOR=${KICK_MAX_PROPOSALS_PER_COLOR:-50000}
MASK_CACHE_DIR=${MASK_CACHE_DIR:-${VERIFY_ROOT}/mask_cache}

OUT_DIR="${VERIFY_ROOT}/${RUN_ID}_grid${EVAL_GRIDSIZE}_heuristic_${FILL_STRATEGY}_kick_p${PASSES}_r${RESTARTS}_a${AUGMENT_ROUNDS}_k${KICK_ROUNDS}_seed${SEED}"
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
echo "fill_strategy=${FILL_STRATEGY}"
echo "mis_edge_chunk_size=${MIS_EDGE_CHUNK_SIZE}"
echo "mis_max_rounds=${MIS_MAX_ROUNDS}"
echo "augment_rounds=${AUGMENT_ROUNDS}"
echo "kick_rounds=${KICK_ROUNDS}"
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
  --fill_strategy "${FILL_STRATEGY}" \
  --mis_edge_chunk_size "${MIS_EDGE_CHUNK_SIZE}" \
  --mis_max_rounds "${MIS_MAX_ROUNDS}" \
  --augment_rounds "${AUGMENT_ROUNDS}" \
  --augment_max_proposals_per_color "${AUGMENT_MAX_PROPOSALS_PER_COLOR}" \
  --kick_rounds "${KICK_ROUNDS}" \
  --kick_max_proposals_per_color "${KICK_MAX_PROPOSALS_PER_COLOR}" \
  --seed "${SEED}" \
  --device cpu \
  --no_plot
