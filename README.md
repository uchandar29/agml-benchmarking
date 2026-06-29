# Ag-VLM Reasoning Dataset Pipeline

Takes AgML image-classification datasets, runs an open-source VLM via vLLM
(in-process, single GPU node), and emits a structured visual-reasoning dataset
plus per-dataset accuracy analytics.

For each image the pipeline runs two passes:

1. **Predict-pass (Stage A)** — image + candidate class list; the model predicts
   the class independently. Produces the `correct` flag and all analytics.
2. **Rationalize-pass (Stage B)** — image + ground-truth label; the model
   justifies the label with structured visual evidence. Produces the reasoning.

`--no-predict` skips Stage A (pure rationalization, no analytics).
`--contrastive` adds "why X and not Y" to Stage B, where Y is mined from the
predict-pass confusion matrix.

## Layout

```
ag_vlm/
  run.py        # CLI entry, multi-dataset orchestration
  engine.py     # in-process vLLM init + batched guided-JSON generate
  prompts.py    # predict + rationalize prompts and JSON schemas
  data.py       # AgML loading, image prep, stable image_id
  analytics.py  # accuracy, confusion matrix, per-class metrics
  io.py         # JSONL append, resume, run_meta
  config.yaml   # tp, max_len, dtype, evidence fields, sampling
slurm/run_agvlm.sbatch
```

## CLI

```
python -m ag_vlm.run \
  --datasets bean_disease_uganda,plant_village \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --output-dir /path/to/out \
  [--limit 200] [--no-predict] [--contrastive]
```

Everything else (tensor-parallel size, max-model-len, dtype, evidence-field
schema, sampling, image resolution cap) lives in `ag_vlm/config.yaml`.

## Outputs (per dataset, under `<output-dir>/<dataset>/`)

- `records.jsonl` — one record per image (true_label, predicted_class, correct,
  predict_reasoning, rationale). Written incrementally; reruns skip finished ids.
- `analytics.json` / `analytics.md` — accuracy, error rate, per-class
  precision/recall/support, confusion matrix, confidence calibration, malformed count.
- `run_meta.json` (at output-dir root) — model, flags, datasets, git commit.

## Setup on FARM (conda)

```bash
module load conda
conda create -n agvlm python=3.11 -y
conda activate agvlm
pip install -r requirements.txt
```

Pre-stage model + data from the **login node** (compute nodes may be offline):

```bash
export HF_HOME=/group_or_scratch/$USER/hf
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
export AGML_DATA_DIR=/group_or_scratch/$USER/agml   # pre-trigger AgML download here
```

In the job set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` if nodes are offline.

### Apptainer alternative

If FARM's CUDA/torch conflicts with the pip wheels, pull a vLLM container and run
the same `python -m ag_vlm.run` command inside `apptainer exec --nv <image> ...`.

## Running

Smoke-test interactively first:

```bash
srun --partition=low --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --pty /bin/bash
python -m ag_vlm.run --datasets bean_disease_uganda \
  --model Qwen/Qwen2.5-VL-7B-Instruct --output-dir ./out --limit 16
```

Then submit the batch job: `sbatch slurm/run_agvlm.sbatch`.

The `low` partition is preemptible with a 12h limit — runs are resumable, so just
resubmit and finished `image_id`s are skipped.

## Verify before scaling

Inspect 10–20 rationales for faithfulness, confirm the confusion-matrix math by
hand on a small slice, and confirm a resubmit skips already-finished records.
```
