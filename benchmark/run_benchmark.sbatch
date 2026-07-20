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
LABEL_COL="label"
IMAGE_COL="image"
METADATA_COLS=""
# ─────────────────────────────────────────────────────────────────────────────

module load cuda/12
module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agvlm

export HF_HOME=/group/jmearlesgrp/$USER/hf
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Move to repo root (sbatch is submitted from benchmark/)
cd ..
echo "Working directory: $PWD"

export AGML_CONFIG=$PWD/benchmark/configs/config.json
export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

# Build optional flags dynamically
EXTRA_ARGS=""
[ -n "$HF_CONFIG_NAME" ] && EXTRA_ARGS="$EXTRA_ARGS --hf-config-name $HF_CONFIG_NAME"
[ -n "$LABEL_COL"      ] && EXTRA_ARGS="$EXTRA_ARGS --label-col $LABEL_COL"
[ -n "$IMAGE_COL"      ] && EXTRA_ARGS="$EXTRA_ARGS --image-col $IMAGE_COL"
[ -n "$METADATA_COLS"  ] && EXTRA_ARGS="$EXTRA_ARGS --metadata-cols $METADATA_COLS"

echo "Starting benchmarking for dataset: $DATASET"
python3 -m benchmark.app \
    --dataset "$DATASET" \
    --phases $PHASES \
    $EXTRA_ARGS
