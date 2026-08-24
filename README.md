# AgML Benchmarking Pipeline

A phased benchmarking pipeline for agricultural image-classification datasets. It computes structural, embedding-based, and training-dynamics metrics across any dataset regardless of size, class count, or schema — then writes a self-contained JSON report for each run.

Datasets can come from HuggingFace (`Project-AgML/...`) or directly from the agml library (`iNatAg/...`, `iNatAg-mini/...`).

## Layout

```
agml-benchmarking/
├── benchmark/
│   ├── app.py                        # entry point — CLI and programmatic API
│   ├── config.py                     # PipelineConfig dataclass
│   ├── requirements.txt
│   ├── configs/
│   │   └── config.json               # infrastructure settings (seeds, thresholds, paths)
│   ├── core/
│   │   ├── dataset_adapter.py        # dataset loading (HF + agml) + schema detection
│   │   ├── split_manager.py          # stratified 70 / 15 / 15 splits with _orig_idx tracking
│   │   ├── embedding_engine.py       # DINOv2 inference + on-disk embedding cache
│   │   ├── reference_trainer.py      # ResNet-18 training for Phase 3 metrics
│   │   └── umap_projector.py         # 2D UMAP projection → JSON for visualisation
│   ├── metrics/
│   │   ├── structural/
│   │   │   ├── class_imbalance/      # label distribution + normalised entropy
│   │   │   ├── exact_duplicate/      # MD5 pixel-hash duplicate detection
│   │   │   ├── resolution_consistency/  # image size / aspect ratio stats
│   │   │   └── near_duplicate/       # FAISS cosine-similarity near-dup detection
│   │   ├── diversity/
│   │   │   ├── metadata_coverage/    # label × metadata contingency tables
│   │   │   └── intra_class_diversity/   # mean L2 distance to per-class centroid
│   │   ├── difficulty/
│   │   │   ├── feature_separability/ # silhouette score + Davies-Bouldin index
│   │   │   ├── dataset_cartography/  # training dynamics (easy / ambiguous / hard)
│   │   │   └── class_confusability/  # per-class confusion from reference model
│   │   └── annotation/
│   │       └── label_noise/          # cleanlab-based noisy label detection
│   ├── output/
│   │   └── writer.py                 # incremental JSON report writer
│   └── execution_scripts/
│       ├── datasets.yaml             # dataset registry — what to run and how
│       ├── run_all.py                # loops through datasets.yaml and runs the pipeline
│       ├── submit_job.py             # submits a single SLURM job for all datasets
│       └── run_benchmark.sbatch      # SLURM job template (FARM @ UC Davis)
├── SCORING_FORMULAS.md               # metric weights, axis formulas, overall score
└── README.md
```

## Phases

**Phase 1 — Structural Quality** (CPU only)
- Class Imbalance — label distribution skew, normalised entropy
- Exact Duplicate Detection — pixel-level MD5 hashing, cross-split leakage
- Resolution Consistency — width/height/aspect stats, coefficient of variation
- Metadata Coverage — class × metadata contingency analysis

**Phase 2 — Embedding-Based Metrics** (GPU, DINOv2-base)
- Near-Duplicate Detection — FAISS IndexFlatIP / IndexIVFFlat, cosine similarity
- Feature Separability — silhouette score + Davies-Bouldin index
- Intra-Class Diversity — mean L2 distance to class centroid
- UMAP Projection — 2D coordinates saved to `umap_projection.json`

**Phase 3 — Training Dynamics + Annotation Reliability** (GPU, ResNet-18)
- Dataset Cartography — confidence mean/std over epochs → easy/ambiguous/hard split
- Class Confusability — confusion matrix from reference model on test split
- Label Noise — cleanlab cross-validated noise detection

Scoring and axis weights are documented in `SCORING_FORMULAS.md`.

## Dataset Sources

| Prefix | Source | Example |
|---|---|---|
| `Project-AgML/...` | HuggingFace | `Project-AgML/rice_leaf_disease_classification` |
| `iNatAg/<name>` | agml library | `iNatAg/acacia_auriculiformis` |
| `iNatAg-mini/<name>` | agml library | `iNatAg-mini/acacia_auriculiformis` |

For compound labels (e.g. crop type + disease), pass `--compound-label-cols` to join multiple columns into a single stratified label.

## Config

Infrastructure settings live in `benchmark/configs/config.json` and are auto-discovered at startup:

1. `$AGML_CONFIG` environment variable
2. `benchmark/config.json` in the working directory
3. Built-in defaults

Key settings:

```json
{
  "split_seed": 42,
  "train_ratio": 0.70,
  "val_ratio": 0.15,
  "embed_model": "facebook/dinov2-base",
  "embed_batch_size": 64,
  "near_dup_threshold": 0.98,
  "backbone": "resnet18",
  "cartography_epochs": 30,
  "cv_folds": 5
}
```

Dataset-specific arguments (name, column names, compound labels) are passed at runtime, not stored in config.

## CLI

```bash
# Simple
python -m benchmark.app \
    --dataset Project-AgML/rice_leaf_disease_classification \
    --phases 1 2 3

# If HF Dataset has multiple configurations (e.g. default, train, raw)
python -m benchmark.app \
    --dataset Project-AgML/watermelon_disease_classification \
    --hf-config-name raw \
    --phases 1 2 3

# Compound label (joins crop_type + label into a single class)
python -m benchmark.app \
    --dataset Project-AgML/crop_pest_disease_classification \
    --compound-label-cols label crop \
    --phases 1 2 3

# iNatAg dataset via agml
python -m benchmark.app \
    --dataset iNatAg-mini/acacia_auriculiformis \
    --phases 1 2 3
```
Will remove the phases as we go forward.

## Setup on FARM

```bash
module load cuda/12
export UV_PROJECT_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.agml-benchmarking/
export HF_HOME=/group/jmearlesgrp/$USER/hf
```

Pre-download datasets and the DINOv2 model from the **login node** before submitting — compute nodes may not have internet access:

```bash
# Dataset
python -c "
from datasets import load_dataset
load_dataset('Project-AgML/rice_leaf_disease_classification')
"

# DINOv2 (Phase 2)
python -c "
from transformers import AutoModel, AutoImageProcessor
AutoImageProcessor.from_pretrained('facebook/dinov2-base')
AutoModel.from_pretrained('facebook/dinov2-base')
"
```

## Batch Submission

Edit `benchmark/execution_scripts/datasets.yaml` to list the datasets you want to run, then submit a single SLURM job for all of them:

```bash
python benchmark/execution_scripts/submit_job.py
```

All datasets run sequentially in one job (48hr wall clock, 64GB GPU). If a dataset fails, the error is written to the `.err` log and the pipeline moves on to the next one. Logs land in `benchmark/logs/YYYY-MM-DD/`.

Each entry in `datasets.yaml` inherits from `defaults` and can override schema fields:

```yaml
defaults:
  phases: "1 2 3"
  hf_config_name: ""
  label_col: "label"
  image_col: "image"
  compound_label_cols: []

datasets:
  - name: "Project-AgML/corn_leaf_pest_classification"
  - name: "Project-AgML/date_grade_variety_classification"
    compound_label_cols: ["label", "variety", "size"]
  - name: "iNatAg-mini/acacia_auriculiformis"
```

You can also run locally without SLURM:

```bash
python benchmark/execution_scripts/run_all.py
```

## Output

Each run writes to `benchmark_results/<dataset_name>/`:

```
benchmark_results/rice_leaf_disease_classification/
├── result.json          # all metric outputs + reproducibility metadata
├── config_used.json     # exact config snapshot for the run
└── embeddings/
    ├── embeddings.npy   # DINOv2 CLS embeddings, shape (N, 768), L2-normalised
    ├── labels.npy       # integer labels, shape (N,)
    └── umap_projection.json
```

Results are written incrementally — if a job is killed mid-run, completed phases are preserved.
