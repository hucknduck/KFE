#!/usr/bin/env bash
#SBATCH --job-name=eval_leaf
#SBATCH --output=logs/leaf_%j.out
#SBATCH --error=logs/leaf_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# Thin wrapper: forward all flags to your Python entry.
set -euo pipefail

mkdir -p logs results/records

# Prefer run.sh if it exists; otherwise fall back to tfe_training.py
REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

if [[ -x "$REPO_ROOT/run.sh" ]]; then
  exec "$REPO_ROOT/run.sh" "$@"
else
  exec python3 "$REPO_ROOT/tfe_training.py" "$@"
fi
