#!/bin/bash
# Keeps every cache on group storage instead of the home folder.
# Sourced by every sbatch file. On the login node run it once per shell before uv sync, uv run or prefetching:
#   source farm_environment.sh

CACHE_ROOT=/group/jmearlesgrp/$USER/.cache

# uv: package cache, downloaded Python interpreters and tools
export UV_CACHE_DIR=$CACHE_ROOT/uv
export UV_PYTHON_INSTALL_DIR=$CACHE_ROOT/uv/python
export UV_TOOL_DIR=$CACHE_ROOT/uv/tools

# Models and datasets
export HF_HOME=$CACHE_ROOT/huggingface
export TORCH_HOME=$CACHE_ROOT/torch
export AGML_DATA_DIR=$CACHE_ROOT

# Anything else that defaults to ~/.cache (pip, matplotlib, triton, vLLM)
export XDG_CACHE_HOME=$CACHE_ROOT
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export MPLCONFIGDIR=$CACHE_ROOT/matplotlib
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm

# Environments, one per kind of job. Each sbatch picks one with UV_PROJECT_ENVIRONMENT.
ENVIRONMENT_ROOT=/group/jmearlesgrp/$USER/agml-benchmarking
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$ENVIRONMENT_ROOT/.agml-benchmarking}
