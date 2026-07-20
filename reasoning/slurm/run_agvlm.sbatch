#!/bin/bash
#SBATCH --job-name=agvlm
#SBATCH --partition=gpu-6000_ada-h  
#SBATCH --account=jmearlesgrp         
#SBATCH --gres=gpu:1               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G                  
#SBATCH --time=12:00:00
#SBATCH --output=logs/agvlm_%j.out
#SBATCH --error=logs/agvlm_%j.err

module load cuda/12
module load conda
conda activate agvlm

export HF_HOME=/group/jmearlesgrp/$USER/hf
export AGML_DATA_DIR=/group/jmearlesgrp/$USER/agml
export HF_HUB_OFFLINE=1            # only if compute nodes are offline
export TRANSFORMERS_OFFLINE=1

python -m ag_vlm.run \
  --datasets bean_disease_uganda \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --output-dir /group/jmearlesgrp/$USER/job/agvlm_out/$SLURM_JOB_ID
