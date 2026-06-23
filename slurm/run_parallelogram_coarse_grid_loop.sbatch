#!/bin/bash
#SBATCH --job-name=pi-aac-grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
PI_WRAPPER=${PI_WRAPPER:-/home/htc/npelleriti/agentic-programming-pi/scripts/pi-slurm-apptainer.sh}
WORKFLOW_NAME=${WORKFLOW_NAME:-parallelogram-coarse-grid-loop}
BASELINE_PCT=${BASELINE_PCT:-3.86731625}
ITERATIONS=${ITERATIONS:-3}
SCREEN_PARALLEL=${SCREEN_PARALLEL:-16}
MEDIUM_PARALLEL=${MEDIUM_PARALLEL:-8}
FINAL_PARALLEL=${FINAL_PARALLEL:-4}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/htc/npelleriti/pi-sandbox/parallelogram-coarse-grid-loop/job_${SLURM_JOB_ID:-manual}}

export SCRATCH=${SCRATCH:-/scratch/htc/npelleriti}
export PI_SANDBOX_BASE=${PI_SANDBOX_BASE:-/scratch/htc/npelleriti/pi-sandbox}
export PI_SANDBOX_SCRATCH=${PI_SANDBOX_SCRATCH:-/scratch/htc/npelleriti/pi-sandbox}
export PI_SANDBOX_WORKSPACE="${REPO_ROOT}"
export PI_SANDBOX_JOB_NAME="${SLURM_JOB_NAME:-pi-aac-grid}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

printf -v WORKFLOW_INPUTS '{"baselinePct":"%s","iterations":"%s","outputRoot":"%s","resetState":"true","screenParallel":"%s","mediumParallel":"%s","finalParallel":"%s"}' \
  "${BASELINE_PCT}" \
  "${ITERATIONS}" \
  "${OUTPUT_ROOT}" \
  "${SCREEN_PARALLEL}" \
  "${MEDIUM_PARALLEL}" \
  "${FINAL_PARALLEL}"

PROMPT="Use the workflow_run tool exactly once with this JSON:
{
  \"name\": \"${WORKFLOW_NAME}\",
  \"scope\": \"project\",
  \"confirmProjectWorkflows\": false,
  \"background\": false,
  \"inputs\": ${WORKFLOW_INPUTS}
}

Do not edit repository files. Run the workflow now. When it finishes, print the final state summary, the shared_context path, and the async jobs DB path."

echo "repo=${REPO_ROOT}"
echo "workflow=${WORKFLOW_NAME}"
echo "output_root=${OUTPUT_ROOT}"
echo "inputs=${WORKFLOW_INPUTS}"
echo "pi_wrapper=${PI_WRAPPER}"

exec "${PI_WRAPPER}" --workspace "${REPO_ROOT}" -p "${PROMPT}"
