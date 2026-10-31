#!/bin/bash -l
#SBATCH --job-name=agml_benchmarking_fine_tuning
#SBATCH --account=jmearlesgrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/group/jmearlesgrp/uchandar/agml-benchmarking/experiments/logs/%x_%j.out
#SBATCH --error=/group/jmearlesgrp/uchandar/agml-benchmarking/experiments/logs/%x_%j.err

# Submit from the repo root. Extra arguments are passed straight to the runner:
#   sbatch experiments/slurm/fine_tuning/fine_tuning.sbatch
#   sbatch experiments/slurm/fine_tuning/fine_tuning.sbatch --datasets bean_disease_uganda --models vit_small
#
# Compute nodes do not have reliable internet, so cache datasets and weights on the login node first:
#   uv run python -m experiments.pipelines.fine_tuning.run --prefetch-only

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
export UV_PROJECT_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.agml-benchmarking

echo "Working directory : $PWD"
echo "Node              : $(hostname)"
echo "Started           : $(date)"
echo "---------------------------------------------------------"

uv run --no-sync python -m experiments.pipelines.fine_tuning.run --config experiments/slurm/fine_tuning/config.yaml "$@"
EXIT_STATUS=$?

echo "---------------------------------------------------------"
echo "Finished : $(date)"
exit $EXIT_STATUS
