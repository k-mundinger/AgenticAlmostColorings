#!/bin/bash
#SBATCH --job-name=pi-aac-neural
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.out
#SBATCH --error=/scratch/htc/npelleriti/agentic-almost-colorings/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/htc/npelleriti/AgenticAlmostColorings}
PI_WRAPPER=${PI_WRAPPER:-/home/htc/npelleriti/agentic-programming-pi/scripts/pi-slurm-apptainer.sh}
WORKFLOW_NAME=${WORKFLOW_NAME:-batched-neural-repair-verification-loop}
BASE_REF=${BASE_REF:-origin/konrad-manual-sweep}
ROUNDS=${ROUNDS:-20}
TARGET_PCT=${TARGET_PCT:-3.60}
BASELINE_PCT=${BASELINE_PCT:-3.733221980}
CALIBRATION_SECONDS=${CALIBRATION_SECONDS:-300}
AGENT_TIMEOUT_SECONDS=${AGENT_TIMEOUT_SECONDS:-3600}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/htc/npelleriti/pi-sandbox/batched-neural-repair-verification-loop/job_${SLURM_JOB_ID:-manual}}
AAC_SHARED_VENV=${AAC_SHARED_VENV:-${OUTPUT_ROOT}/shared_venv}

export SCRATCH=${SCRATCH:-/scratch/htc/npelleriti}
export PI_SANDBOX_BASE=${PI_SANDBOX_BASE:-/scratch/htc/npelleriti/pi-sandbox}
export PI_SANDBOX_SCRATCH=${PI_SANDBOX_SCRATCH:-/scratch/htc/npelleriti/pi-sandbox}
export PI_SANDBOX_WORKSPACE="${REPO_ROOT}"
export PI_SANDBOX_JOB_NAME="${SLURM_JOB_NAME:-pi-aac-neural}"
export PI_SANDBOX_GPU=${PI_SANDBOX_GPU:-1}
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

if ! git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null 2>&1; then
  if [[ "${BASE_REF}" == "origin/konrad-manual-sweep" ]]; then
    git fetch origin konrad-manual-sweep:refs/remotes/origin/konrad-manual-sweep
  fi
fi

printf -v WORKFLOW_INPUTS '{"baseRef":"%s","rounds":"%s","targetPct":"%s","baselinePct":"%s","calibrationSeconds":"%s","agentTimeoutSeconds":"%s","outputRoot":"%s"}' \
  "${BASE_REF}" \
  "${ROUNDS}" \
  "${TARGET_PCT}" \
  "${BASELINE_PCT}" \
  "${CALIBRATION_SECONDS}" \
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

Do not edit repository files. Run the workflow now. When it finishes, print the final synthesis path, final fanout plan path, output root, async jobs DB path, and a concise status summary."

echo "repo=${REPO_ROOT}"
echo "workflow=${WORKFLOW_NAME}"
echo "base_ref=${BASE_REF}"
echo "rounds=${ROUNDS}"
echo "target_pct=${TARGET_PCT}"
echo "baseline_pct=${BASELINE_PCT}"
echo "calibration_seconds=${CALIBRATION_SECONDS}"
echo "agent_timeout_seconds=${AGENT_TIMEOUT_SECONDS}"
echo "output_root=${OUTPUT_ROOT}"
echo "aac_shared_venv=${AAC_SHARED_VENV}"
echo "pi_sandbox_gpu=${PI_SANDBOX_GPU}"
echo "inputs=${WORKFLOW_INPUTS}"
echo "pi_wrapper=${PI_WRAPPER}"

exec "${PI_WRAPPER}" --workspace "${REPO_ROOT}" -p "${PROMPT}"
