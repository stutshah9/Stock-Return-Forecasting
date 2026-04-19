#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Repo root: $REPO_ROOT"
echo "Python: $(command -v python3)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset}"

mkdir -p data/transcript_cache

python3 experiments/sweep.py "$@"
