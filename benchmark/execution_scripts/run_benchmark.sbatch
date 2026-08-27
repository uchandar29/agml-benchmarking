#!/bin/bash -l
#SBATCH --partition=low
#SBATCH --account=publicgrp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=agbenchmark_all
#SBATCH --exclude=gpu-12-92
# --output and --error are set by submit_job.py using dated absolute paths

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12

export UV_PROJECT_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.agml-benchmarking/
export HF_HOME=/group/jmearlesgrp/uchandar/.cache/huggingface
export AGML_DATA_DIR=/group/jmearlesgrp/uchandar/.cache

# REPO_ROOT is injected by submit_job.py.
# For a manual one-off submission from the repo root:
#   REPO_ROOT=$(pwd) sbatch benchmark/execution_scripts/run_benchmark.sbatch
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

export AGML_CONFIG="$REPO_ROOT/benchmark/configs/config.json"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "Working directory : $PWD"
echo "AGML_CONFIG       : $AGML_CONFIG"
echo "Started           : $(date)"
echo ""

# ── Run all datasets defined in datasets.yaml ─────────────────────────────────
uv run python benchmark/execution_scripts/run_all.py

echo ""
echo "Finished : $(date)"
