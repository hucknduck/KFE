#!/usr/bin/env bash
set -euo pipefail

# help PyTorch allocator on large graphs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# if fallback script exists AND KFE_NO_FALLBACK is NOT set ? use fallback
if [ -x scripts/train_fallback.sh ] && [ -z "${KFE_NO_FALLBACK:-}" ]; then
  echo "[run.sh] using fallback wrapper"
  bash scripts/train_fallback.sh "$@"
else
  echo "[run.sh] using direct training call"
  python3 tfe_training.py "$@"
fi