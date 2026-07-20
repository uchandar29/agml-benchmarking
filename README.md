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

**vLLM version:** this pipeline targets **vLLM >= 0.11.0** (it uses the
`StructuredOutputsParams` structured-output API). Older vLLM will fail at import
with a clear message; upgrade rather than downgrade the code.

### GPU kernel toolchain (FARM-specific, learned the hard way)

On recent GPUs (Hopper / Ada, e.g. the `gpu-6000_ada-h` nodes) vLLM JIT-compiles
some kernels (the FlashInfer sampler, and DeepGEMM for FP8 models) at startup.
Those builds need a CUDA toolkit **and** a matching host C++ compiler on the node.
The clean setup that works:

```bash
module load cuda/12          # provides nvcc 12.x (matches the torch CUDA build)
module load conda            # NOTE: gcc/13 and conda conflict as modules — don't load gcc as a module
conda activate agvlm
conda install -c conda-forge gxx=13 -y   # one-time: g++ 13 inside the env, no module conflict
```

Confirm `nvcc --version` reports **12.x** (not an older system CUDA) before running.

**Fallback if the JIT toolchain still fights you:** disable the JIT kernels and
let vLLM use its precompiled/native paths (same outputs, slightly slower):

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0   # avoids the FlashInfer sampler JIT
export VLLM_USE_DEEP_GEMM=0            # avoids the DeepGEMM FP8 JIT (FP8 models only)
```

Both env vars are harmless when not needed, so they are a safe default for the
sbatch. FP8 models are the most JIT-heavy — for those, the Apptainer route
(below) is the least fragile.

Pre-stage model + data from the **login node** (compute nodes may be offline):

```bash
export HF_HOME=/group_or_scratch/$USER/hf
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
export AGML_DATA_DIR=/group_or_scratch/$USER/agml   # pre-trigger AgML download here
```

In the job set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` if nodes are offline.

### Apptainer alternative

If FARM's CUDA/torch conflicts with the pip wheels — or the FP8 kernel JIT builds
keep failing — pull a vLLM container (it ships a matched CUDA + GCC toolchain, so
every JIT build succeeds and none of the env vars above are needed) and run the
same `python -m ag_vlm.run` command inside `apptainer exec --nv <image> ...`.

## Running

Smoke-test interactively first:

```bash
srun --partition=gpu-6000_ada-h --account=jmearlesgrp --gres=gpu:1 \
     --cpus-per-task=12 --mem=64G --time=01:00:00 --pty /bin/bash
# then: module load cuda/12 conda && conda activate agvlm && export HF_HOME=... AGML_DATA_DIR=...
python -m ag_vlm.run --datasets bean_disease_uganda \
  --model Qwen/Qwen2.5-VL-7B-Instruct --output-dir ./out --limit 16
```

Give the interactive session enough wall-time: the first run pays a one-time
model-load + `torch.compile` cost (minutes) before inference, and Slurm kills the
job the instant `--time` expires regardless of progress. Subsequent runs are fast
(weights and compiled graphs are cached).

Then submit the batch job: `sbatch slurm/run_agvlm.sbatch`.

Runs are resumable — on a preemptible partition, just resubmit and finished
`image_id`s are skipped (records are flushed every `batch_size` images).

## Verify before scaling

Inspect 10–20 rationales for faithfulness, confirm the confusion-matrix math by
hand on a small slice, and confirm a resubmit skips already-finished records.
```
