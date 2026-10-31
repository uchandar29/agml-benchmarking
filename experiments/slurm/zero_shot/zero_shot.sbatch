#!/bin/bash -l
#SBATCH --job-name=agml_benchmarking_zero_shot
#SBATCH --account=jmearlesgrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/group/jmearlesgrp/uchandar/agml-benchmarking/experiments/logs/%x_%j.out
#SBATCH --error=/group/jmearlesgrp/uchandar/agml-benchmarking/experiments/logs/%x_%j.err

# Submit from the repo root:
#   sbatch experiments/slurm/zero_shot/zero_shot.sbatch
#   sbatch experiments/slurm/zero_shot/zero_shot.sbatch qwen_3_5_27b           # only some models (space separated)
#
# Cache weights and datasets on the login node first (offline mode unset there):
#   uv run python -m experiments.pipelines.zero_shot.run --prefetch-only
#
# Uses its own environment because vLLM pins its own torch version, which
# should not touch the benchmark / fine-tuning environment.

module load cuda/12

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
if [ ! -d "$REPO_ROOT/experiments" ] || [ ! -d "$REPO_ROOT/core" ]; then
    echo "Submit this job from the repo root (got $REPO_ROOT)" >&2
    exit 1
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Caches on group storage, and the environment this job runs in
source farm_environment.sh
export UV_PROJECT_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.zero-shot-env

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(siglip_so400m gemma_4_31b qwen_3_5_27b)
fi

echo "Working directory : $PWD"
echo "Node              : $(hostname)"
echo "Models            : ${MODELS[*]}"
echo "Started           : $(date)"
echo "---------------------------------------------"

# One process per model so each one starts with a clean GPU
EXIT_STATUS=0
for MODEL_NAME in "${MODELS[@]}"; do
    uv run --no-sync python -m experiments.pipelines.zero_shot.run --config experiments/slurm/zero_shot/config.yaml --models "$MODEL_NAME" || EXIT_STATUS=1
done

echo "---------------------------------------------"
echo "Finished : $(date)"
exit $EXIT_STATUS
