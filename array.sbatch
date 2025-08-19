#!/bin/bash
#
# Example array submit: sbatch --array=0-3%2 slurm/array.sbatch
#
#SBATCH --job-name=tfe-array
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Uncomment / adjust if you plan to use GPUs
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --qos=gpu-short

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${REPO_ROOT}/scripts/jobs.tsv"

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/results/runs"

# Optional: activate your conda env here
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate /path/to/your/env

# Pick the command from the TSV (skip header)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$((IDX + 2))
CMD="$(sed -n "${LINE}p" "${JOBS}" | cut -f2-)"
if [[ -z "${CMD}" ]]; then
  echo "No command for index ${IDX} (line ${LINE})"
  exit 1
fi

echo "[SLURM ${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}] Running: ${CMD}"
cd "${REPO_ROOT}"
# You can prefix with srun if desired:
# srun ${CMD}
eval ${CMD}
