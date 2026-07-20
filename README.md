# AgML Benchmarking Pipeline

A generic, phased benchmarking pipeline for HuggingFace image-classification datasets. Computes structural quality, embedding-based, and (upcoming) model-training metrics for any dataset regardless of size, class count, or schema.

## Layout

```
benchmark/
  app.py                    # entry point — programmatic and CLI
  config.py                 # PipelineConfig (infrastructure settings)
  configs/
    config.json             # universal config file
  core/
    dataset_adapter.py      # HF dataset loading + schema auto-detection
    split_manager.py        # stratified 70 / 15 / 15 splits
    embedding_engine.py     # DINOv2 inference + embedding cache
  metrics/
    structural/
      class_imbalance/      # class distribution + entropy
      exact_duplicate/      # MD5-based pixel-exact duplicate detection
      resolution_consistency/  # image size and aspect ratio stats
      near_duplicate/       # FAISS cosine similarity near-dup detection
    diversity/
      metadata_coverage/    # label × metadata contingency analysis
      intra_class_diversity/  # mean L2 distance to per-class centroid
    difficulty/
      feature_separability/ # silhouette score + Davies-Bouldin index
  output/
    writer.py               # incremental JSON report writer
  run_benchmark.sbatch      # SLURM job script for FARM @ UC Davis
```

## Phases

**Phase 1 — Structural Quality** (no GPU required)
- Class Imbalance
- Exact Duplicate Detection
- Resolution Consistency
- Metadata Coverage

**Phase 2 — Embedding-Based Metrics** (GPU required, DINOv2)
- Near-Duplicate Detection (FAISS)
- Feature Separability (Silhouette + Davies-Bouldin)
- Intra-Class Diversity

**Phase 3 — Model Training Metrics** *(coming soon)*

## Config

Infrastructure settings live in `benchmark/configs/config.json`. The config is auto-discovered in this order:

1. `$AGML_CONFIG` environment variable
2. `benchmark/config.json` in the current working directory
3. Built-in defaults

Dataset-specific arguments (dataset name, column names) are passed at runtime, not stored in the config.

## CLI

```bash
# Minimal — config auto-loaded
python -m benchmark.app \
    --dataset Project-AgML/rice_leaf_disease_classification \
    --phases 1 2

# Multi-config HF dataset
python -m benchmark.app \
    --dataset Project-AgML/watermelon_disease_classification \
    --hf-config-name raw \
    --phases 1 2

# Dataset with explicit column names
python -m benchmark.app \
    --dataset Project-AgML/banana_grade_variety_classification \
    --label-col label --image-col image \
    --metadata-cols variety scale \
    --phases 1 2
```

## Programmatic

```python
from benchmark.app import AgMLBenchmarkPipeline

pipeline = AgMLBenchmarkPipeline("Project-AgML/rice_leaf_disease_classification")
pipeline.run(phases=[1, 2])
```

## Setup on FARM (conda)

```bash
module load cuda/12
module load conda
conda activate agvlm
pip install -r benchmark/requirements.txt
```

Pre-download dataset and model from the **login node** before submitting (compute nodes may be offline):

```bash
export HF_HOME=/group/jmearlesgrp/$USER/hf

# Dataset
python -c "
from datasets import load_dataset
load_dataset('Project-AgML/rice_leaf_disease_classification', cache_dir='$HF_HOME')
"

# DINOv2 model (Phase 2)
python -c "
from transformers import AutoModel, AutoImageProcessor
AutoImageProcessor.from_pretrained('facebook/dinov2-base', cache_dir='$HF_HOME')
AutoModel.from_pretrained('facebook/dinov2-base', cache_dir='$HF_HOME')
"
```

## Running on FARM

Edit the variables at the top of `benchmark/run_benchmark.sbatch` and submit:

```bash
sbatch benchmark/run_benchmark.sbatch
```

Logs are written to `logs/benchmark_<job_id>.out` and `.err`.

---

> **Reasoning pipeline** — a separate VLM-based visual reasoning dataset pipeline (`ag_vlm/`) also lives in this repo. Documentation for that module is maintained separately.
