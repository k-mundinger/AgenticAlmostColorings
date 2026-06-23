#!/bin/bash
#SBATCH --job-name=pi-aac-papers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
PI_WRAPPER=${PI_WRAPPER:-/home/htc/npelleriti/agentic-programming-pi/scripts/pi-slurm-apptainer.sh}
WORKFLOW_NAME=${WORKFLOW_NAME:-papers-concept-exploration-loop}
PAPER_DIR=${PAPER_DIR:-constructions/paper_almost_5_coloring_2d}
ROUNDS=${ROUNDS:-20}
AGENT_TIMEOUT_SECONDS=${AGENT_TIMEOUT_SECONDS:-3600}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/htc/npelleriti/pi-sandbox/papers-concept-exploration-loop/job_${SLURM_JOB_ID:-manual}}
OBJECTIVE=${OBJECTIVE:-Start from the paper construction and generate concrete, paper-grounded follow-up research directions for improving or extending the almost-5-coloring construction, prioritizing independently verifiable ideas and useful negative evidence.}
AAC_SHARED_VENV=${AAC_SHARED_VENV:-${OUTPUT_ROOT}/shared_venv}

export SCRATCH=${SCRATCH:-/scratch/htc/npelleriti}
export PI_SANDBOX_BASE=${PI_SANDBOX_BASE:-/scratch/htc/npelleriti/pi-sandbox}
export PI_SANDBOX_SCRATCH=${PI_SANDBOX_SCRATCH:-/scratch/htc/npelleriti/pi-sandbox}
export PI_SANDBOX_WORKSPACE="${REPO_ROOT}"
export PI_SANDBOX_JOB_NAME="${SLURM_JOB_NAME:-pi-aac-papers}"
export AAC_SHARED_VENV
export UV_PROJECT_ENVIRONMENT="${AAC_SHARED_VENV}"
export APPTAINERENV_AAC_SHARED_VENV="${AAC_SHARED_VENV}"
export APPTAINERENV_UV_PROJECT_ENVIRONMENT="${AAC_SHARED_VENV}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUTPUT_ROOT}" /scratch/htc/npelleriti/agentic-almost-colorings/slurm
cd "${REPO_ROOT}"

printf -v WORKFLOW_INPUTS '{"paperDir":"%s","objective":"%s","rounds":"%s","agentTimeoutSeconds":"%s","outputRoot":"%s"}' \
  "${PAPER_DIR}" \
  "${OBJECTIVE}" \
  "${ROUNDS}" \
  "${AGENT_TIMEOUT_SECONDS}" \
  "${OUTPUT_ROOT}"

PROMPT="Use the workflow_run tool exactly once with this JSON:
{
  \"name\": \"${WORKFLOW_NAME}\",
  \"scope\": \"project\",
  \"confirmProjectWorkflows\": false,
  \"background\": false,
  \"inputs\": ${WORKFLOW_INPUTS}
}

Do not edit repository files. Run the workflow now. When it finishes, print the final synthesis path, output root, async jobs DB path, and a concise status summary."

echo "repo=${REPO_ROOT}"
echo "workflow=${WORKFLOW_NAME}"
echo "paper_dir=${PAPER_DIR}"
echo "rounds=${ROUNDS}"
echo "agent_timeout_seconds=${AGENT_TIMEOUT_SECONDS}"
echo "output_root=${OUTPUT_ROOT}"
echo "aac_shared_venv=${AAC_SHARED_VENV}"
echo "inputs=${WORKFLOW_INPUTS}"
echo "pi_wrapper=${PI_WRAPPER}"

exec "${PI_WRAPPER}" --workspace "${REPO_ROOT}" -p "${PROMPT}"
