#!/usr/bin/env bash
#SBATCH --job-name=eval_leaf
#SBATCH --output=logs/leaf_%j.out
#SBATCH --error=logs/leaf_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
###SBATCH --time=01:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=geforce_rtx_3090

# Thin wrapper: forward all flags to your Python entry.
cd /vol/joberant_nobck/data/NLP_368307701_2425a/yuvall2/KFE

# Redirect HOME so that DGL (and others) don’t touch the quota-limited homedir
export HOME="/vol/joberant_nobck/data/NLP_368307701_2425a/yuvall2/tmp_home"
mkdir -p "$HOME"

# Ensure DGL honors this
export DGL_DOWNLOAD_DIR="$HOME/.dgl"
export DGL_HOME="$HOME/.dgl"
mkdir -p "$DGL_HOME"
set -euo pipefail

mkdir -p logs results/records

# Prefer run.sh if it exists; otherwise fall back to tfe_training.py
#REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
#echo $REPO_ROOT
#if [[ -x "$REPO_ROOT/run.sh" ]]; then
#  exec "$REPO_ROOT/run.sh" "$@"
#else
exec python3 tfe_training.py "$@"
#fi
