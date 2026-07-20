#!/bin/bash
#SBATCH --job-name=agml-benchmark
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --account=jmearlesgrp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# ── Edit these before submitting ──────────────────────────────────────────────
DATASET="Project-AgML/watermelon_disease_classification"
PHASES="1 2"

# Dataset-specific overrides — leave empty to use auto-detection
HF_CONFIG_NAME="raw"   # e.g. "raw" for watermelon_disease_classification
LABEL_COL="label"        # e.g. "label"
IMAGE_COL="image"        # e.g. "image"
METADATA_COLS=""    # e.g. "variety scale"
# ─────────────────────────────────────────────────────────────────────────────

module load cuda/12
module load conda
conda activate agvlm

export HF_HOME=/group/jmearlesgrp/$USER/hf
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export AGML_CONFIG=$PWD/benchmark/configs/config.json

mkdir -p logs

# Build optional flags dynamically
EXTRA_ARGS=""
[ -n "$HF_CONFIG_NAME" ] && EXTRA_ARGS="$EXTRA_ARGS --hf-config-name $HF_CONFIG_NAME"
[ -n "$LABEL_COL"      ] && EXTRA_ARGS="$EXTRA_ARGS --label-col $LABEL_COL"
[ -n "$IMAGE_COL"      ] && EXTRA_ARGS="$EXTRA_ARGS --image-col $IMAGE_COL"
[ -n "$METADATA_COLS"  ] && EXTRA_ARGS="$EXTRA_ARGS --metadata-cols $METADATA_COLS"

python -m benchmark.app \
    --dataset "$DATASET" \
    --phases $PHASES \
    $EXTRA_ARGS
