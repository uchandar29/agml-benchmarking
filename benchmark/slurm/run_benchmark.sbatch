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
#SBATCH --output=benchmark/logs/%x_%j.out
#SBATCH --error=benchmark/logs/%x_%j.err

# Submit from the repo root:
#   sbatch benchmark/slurm/run_benchmark.sbatch
#   sbatch benchmark/slurm/run_benchmark.sbatch --yaml benchmark/slurm/datasets.yaml
# Or use python benchmark/slurm/submit_job.py, which writes logs to a dated folder under benchmark/logs.

module load cuda/12

# The repo is not a uv project, so activate the environment directly
PYTHON_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.agml-benchmarking
source "$PYTHON_ENVIRONMENT/bin/activate"
export HF_HOME=/group/jmearlesgrp/uchandar/.cache/huggingface
export AGML_DATA_DIR=/group/jmearlesgrp/uchandar/.cache

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
if [ ! -d "$REPO_ROOT/benchmark" ] || [ ! -d "$REPO_ROOT/core" ]; then
    echo "Submit this job from the repo root (got $REPO_ROOT)" >&2
    exit 1
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export AGML_CONFIG="$REPO_ROOT/benchmark/slurm/config.json"

echo "Working directory : $PWD"
echo "AGML_CONFIG       : $AGML_CONFIG"
echo "Started           : $(date)"
echo ""

python -m benchmark.run_all "$@"
EXIT_STATUS=$?

echo ""
echo "Finished : $(date)"
exit $EXIT_STATUS
