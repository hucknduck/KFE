#!/bin/bash
#SBATCH --job-name=mbtfe
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --constraint=geforce_rtx_3090
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

#cd /vol/joberant_nobck/data/NLP_368307701_2425a/yuvall2/KFE
#source /vol/joberant_nobck/data/NLP_368307701_2425a/yuvall2/anaconda3/etc/profile.d/conda.sh
#conda activate GRAPHS
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ---- DGL & cache fixes (no root/home write needed) ----
export DGLBACKEND=pytorch
export DGL_BACKEND=pytorch
export XDG_CACHE_HOME="$PWD/.cache"
export HOME="$PWD/.home"
mkdir -p "$XDG_CACHE_HOME" "$HOME/.dgl"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

# run
srun bash slurm/run.sh "$@"