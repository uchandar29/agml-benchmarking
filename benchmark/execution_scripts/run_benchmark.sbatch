#!/bin/bash -l
#SBATCH --partition=low
#SBATCH --account=publicgrp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── How this script is driven ─────────────────────────────────────────────────
# Normally submitted via benchmark/execution_scripts/submit_all.py, which sets
# --job-name, --mem, --time, and the dataset environment variables automatically.
#
# For a manual one-off run (from repo root):
#   DATASET="Project-AgML/..." \
#   PHASES="1 2 3" \
#   sbatch --job-name=agbenchmark_mydataset benchmark/run_benchmark.sbatch
# ─────────────────────────────────────────────────────────────────────────────

module load cuda/12
export UV_PROJECT_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.agml-benchmarking/
export HF_HOME=/group/jmearlesgrp/$USER/hf

# SLURM_SUBMIT_DIR is the directory from which sbatch was called.
# submit_all.py runs from repo root, so this always resolves correctly.
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $PWD"
echo "Dataset          : $DATASET"
echo "Phases           : $PHASES"

export AGML_CONFIG=$PWD/benchmark/configs/config.json
export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

EXTRA_ARGS=""
[ -n "$HF_CONFIG_NAME"      ] && EXTRA_ARGS="$EXTRA_ARGS --hf-config-name $HF_CONFIG_NAME"
[ -n "$LABEL_COL"           ] && EXTRA_ARGS="$EXTRA_ARGS --label-col $LABEL_COL"
[ -n "$IMAGE_COL"           ] && EXTRA_ARGS="$EXTRA_ARGS --image-col $IMAGE_COL"
[ -n "$COMPOUND_LABEL_COLS" ] && EXTRA_ARGS="$EXTRA_ARGS --compound-label-cols $COMPOUND_LABEL_COLS"

uv run python -m benchmark.app \
	--dataset "$DATASET" \
	--phases $PHASES \
	$EXTRA_ARGS
