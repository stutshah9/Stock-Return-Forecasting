#!/usr/bin/env bash
#SBATCH --job-name=earnings-train-eval
#SBATCH --account=cp-spring2026-iac
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

bash scripts/falcon_train_eval.sh
