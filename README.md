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
    └── umap_projection.bin
```

Results are written incrementally — if a job is killed mid-run, completed phases are preserved.

# UMAP Embedding Binary Format (UEB2)

## Purpose

Per-dataset UMAP projection data (2D and 3D scatter coordinates, one row per
sample) used to be stored as two JSON files per dataset:
<dataset>_umap_2D.json   -> [{ x, y, label, split, index }, ...]
<dataset>_umap_3D.json   -> [{ x, y, z, label, split, index }, ...]

This is now stored as one custom binary file per dataset:
<dataset>_umap.bin

`UEB2` is a made-up format name ("UMAP Embedding Binary, version 2"), not an
external/standard format. The name only matters as a magic-byte signature so a
reader can confirm it's looking at the right kind of file.

Why binary at all: JSON re-encodes every float as decimal text and repeats the
same label/split strings on every row. For ~200 datasets this JSON totaled
~88MB. Deduplicating labels/splits into small tables, referencing them by
index, and storing coordinates as 4-byte float32 instead of text cuts total
size to ~15MB (83% smaller) with no loss of precision that matters for a
scatter plot.

## Core invariant (must hold before writing)
**A pipeline change must assert this before writing**, e.g.:

```python
assert len(points_2d) == len(points_3d)
for a, b in zip(points_2d, points_3d):
    assert a["label"] == b["label"]
    assert a.get("split", "train") == b.get("split", "train")
    assert a.get("index") == b.get("index")
```

If this ever stops holding (e.g. the 2D and 3D fits get produced from
differently-ordered or differently-filtered sample sets), the format doesn't
error — it silently attaches row i's label/split/index to the wrong
coordinates in whichever block was reordered.

## Byte layout
All multi-byte integers and floats are little-endian.
```
offset 0    4 bytes    magic = ASCII "UEB2"
offset 4    1 byte     uint8  flags         bit0 = has 2D coords section
                                            bit1 = has 3D coords section
offset 5    2 bytes    uint16 labelCount
offset 7    1 byte     uint8  splitCount
offset 8    4 bytes    uint32 pointCount

--- label table: labelCount entries ---
each entry:
    1 byte      uint8 len
    len bytes   UTF-8 string (the class/label name)

--- split table: splitCount entries ---
each entry:
    1 byte      uint8 len
    len bytes   UTF-8 string (e.g. "train" / "val" / "test")

--- metadata block: pointCount records, shared by both coordinate sets ---
each record (7 bytes):
    2 bytes     uint16 labelIdx     (index into label table)
    1 byte      uint8  splitIdx     (index into split table)
    4 bytes     uint32 index        (original sample index)

--- 2D coordinate block: pointCount records — ONLY present if flags bit0 set ---
each record (8 bytes):
    4 bytes     float32 x
    4 bytes     float32 y

--- 3D coordinate block: pointCount records — ONLY present if flags bit1 set ---
each record (12 bytes):
    4 bytes     float32 x
    4 bytes     float32 y
    4 bytes     float32 z
Point record sizes: 7 bytes metadata + 8 bytes per 2D point + 12
bytes per 3D point (when both sections are present).
```

### Worked example
3 points, 2 labels (Cat, Dog), 2 splits (test, train), both 2D and 3D
present. Labels/splits are sorted alphabetically and assigned indices in that
order — Cat=0, Dog=1, test=0, train=1 (sort order isn't required by the
reader, it's just how the reference writer builds the table deterministically).
```
     x     y     z   label   split  index 
0   1.5   2.5   3.5   Cat   train   0 
1   -0.25 4.0   1.0   Dog   test    1 
2   0.0   0.0   0.0   Cat   train   2 
```

Resulting 112-byte file:
```
0  55 45 42 32                      magic "UEB2"
4  03                               flags = 0b11 (2D + 3D both present)
5  02 00                            labelCount = 2
7  02                               splitCount = 2
8  03 00 00 00                      pointCount = 3

12  03 43 61 74                      label[0]: len=3, "Cat"
16  03 44 6f 67                      label[1]: len=3, "Dog"
20  04 74 65 73 74                   split[0]: len=4, "test"
25  05 74 72 61 69 6e                split[1]: len=5, "train"

31  00 00 01 00 00 00 00             meta[0]: labelIdx=0(Cat) splitIdx=1(train) index=0
38  01 00 00 01 00 00 00             meta[1]: labelIdx=1(Dog) splitIdx=0(test)  index=1
45  00 00 01 02 00 00 00             meta[2]: labelIdx=0(Cat) splitIdx=1(train) index=2

52  00 00 c0 3f 00 00 20 40          2D[0]: x=1.5, y=2.5
60  00 00 80 be 00 00 80 40          2D[1]: x=-0.25, y=4.0
68  00 00 00 00 00 00 00 00          2D[2]: x=0.0, y=0.0

76  00 00 c0 3f 00 00 20 40 00 00 60 40   3D[0]: x=1.5, y=2.5, z=3.5
88  00 00 80 be 00 00 80 40 00 00 80 3f   3D[1]: x=-0.25, y=4.0, z=1.0
100  00 00 00 00 00 00 00 00 00 00 00 00   3D[2]: x=0.0, y=0.0, z=0.0
```


#### Reading (reference implementation)
A function named parseEmbeddingBinary(buffer) is the canonical
reader. Steps, in order, tracking a running offset:

1. **Header Check:** Read 4 bytes, confirm they equal `"UEB2"`. If not, return `null` / empty result (do not throw).
2. **Flags:** Read `flags` (uint8):
   * `has2d = (flags & 1) !== 0`
   * `has3d = (flags & 2) !== 0`
3. **Counts:** Read `labelCount` (uint16), `splitCount` (uint8), and `pointCount` (uint32).
4. **Labels:** Loop `labelCount` times: read 1-byte length $L$, then decode $L$ bytes as UTF-8 string $\rightarrow$ push onto `labels`.
5. **Splits:** Loop `splitCount` times: read 1-byte length $L$, then decode $L$ bytes as UTF-8 string $\rightarrow$ push onto `splits`.
6. **Metadata Arrays:**
   * Read `pointCount` uint16s into `labelIdx`.
   * Read `pointCount` uint8s into `splitIdx`.
   * Read `pointCount` uint32s into `index`.
7. **2D Coordinates Block (if `has2d`):**
   * Loop `i` from `0` to `pointCount - 1`:
     * Read `x` (float32), `y` (float32).
     * Resolve `label = labels[labelIdx[i]]` and `split = splits[splitIdx[i]]`.
     * Construct point object `{ label, split, index: index[i], x, y }` and push to `points2D`.
8. **3D Coordinates Block (if `has3d`):**
   * Loop `i` from `0` to `pointCount - 1`:
     * Read `x` (float32), `y` (float32), `z` (float32).
     * Resolve `label = labels[labelIdx[i]]` and `split = splits[splitIdx[i]]`.
     * Construct point object `{ label, split, index: index[i], x, y, z }` and push to `points3D`.
9. **Return:** Return `points2D` (or `null` if `!has2d`) and `points3D` (or `null` if `!has3d`).


### File paths
Same naming/nesting convention as before, minus the _2D/_3D suffix:
```
static/data/dataset-benchmarking/embeddings/<dataset>_umap.bin
static/data/dataset-benchmarking/embeddings/iNatAg/<species>_umap.bin
static/data/dataset-benchmarking/embeddings/iNatAg-mini/<species>_umap.bin
```