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
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err

# Submit from the repo root:
#   sbatch experiments/zero_shot.sbatch
#   sbatch experiments/zero_shot.sbatch qwen_3_5_27b           # only some models (space separated)
#
# Cache weights and datasets on the login node first (offline mode unset there):
#   uv run python -m experiments.zero_shot.run --prefetch-only
#
# Uses its own environment because vLLM pins its own torch version, which
# should not touch the benchmark / fine-tuning environment.

module load cuda/12

export UV_PROJECT_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.zero-shot-env/
export HF_HOME=/group/jmearlesgrp/uchandar/.cache/huggingface
export AGML_DATA_DIR=/group/jmearlesgrp/uchandar/.cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

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
    uv run python -m experiments.zero_shot.run --config "$REPO_ROOT/experiments/config.yaml" --models "$MODEL_NAME" || EXIT_STATUS=1
done

echo "---------------------------------------------"
echo "Finished : $(date)"
exit $EXIT_STATUS
