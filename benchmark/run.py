#!/usr/bin/env python3
"""
benchmark/run.py
================
The single entry point of the benchmark. Reads datasets.yaml and runs the pipeline of
each entry's task, one dataset after another. A dataset that fails (download error,
corrupt data, pipeline bug) is logged to stderr and skipped, the rest keeps running.

stdout  →  progress banners, metric summaries, per-dataset results  (SLURM .out)
stderr  →  full tracebacks, prefixed with the dataset name           (SLURM .err)

Usage (from repo root, or via SLURM):
    uv run python -m benchmark.run                                  # every entry
    uv run python -m benchmark.run --task object_detection          # one task
    uv run python -m benchmark.run --dataset Project-AgML/GEMINI_cowpea_pod_detection
    uv run python -m benchmark.run --task object_detection --prefetch-only   # login node: downloads only

Adding a task (for example segmentation) means one pipeline class in
benchmark/pipelines/ and one entry in TASK_PIPELINES below.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from benchmark.config import REPO_ROOT, PipelineConfig
from benchmark.pipelines.image_classification import AgMLBenchmarkPipeline
from benchmark.pipelines.object_detection import DetectionBenchmarkPipeline

DEFAULT_YAML = REPO_ROOT / "benchmark" / "slurm" / "datasets.yaml"

TASK_ALIASES = {"classification": "image_classification"}


@dataclass(frozen=True)
class TaskPipeline:
    pipeline_class: type
    # constructor argument -> yaml key
    arguments: Dict[str, str]


TASK_PIPELINES: Dict[str, TaskPipeline] = {
    "image_classification": TaskPipeline(
        AgMLBenchmarkPipeline,
        {
            "config_name": "hf_config_name",
            "label_col": "label_col",
            "image_col": "image_col",
            "compound_label_cols": "compound_label_cols",
        },
    ),
    "object_detection": TaskPipeline(
        DetectionBenchmarkPipeline,
        {
            "config_name": "hf_config_name",
            "image_col": "image_col",
            "objects_col": "objects_col",
            "bbox_key": "bbox_key",
            "category_key": "category_key",
            "bbox_format": "bbox_format",
        },
    ),
}

# Used when datasets.yaml has no defaults block (same layout as the yaml)
BUILTIN_DEFAULTS: Dict[str, Any] = {
    "task":           "image_classification",
    "phases":         "1 2 3",
    "hf_config_name": "",
    "image_col":      "image",
    "image_classification": {
        "label_col":           "label",
        "compound_label_cols": [],
    },
    "object_detection": {
        "objects_col":  "objects",
        "bbox_key":     "bbox",
        "category_key": "categories",
        "bbox_format":  "auto",
    },
}

WIDE  = 70
THICK = "═"


# ── Logging helpers ───────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _flush(*args, **kwargs):
    """print() that always flushes — important inside long SLURM jobs."""
    print(*args, **kwargs, flush=True)


def _dataset_header(index: int, total: int, name: str, task: str, phases: list, report_path: str) -> None:
    _flush()
    _flush(THICK * WIDE)
    _flush(f"{'AgML Benchmark Pipeline':^{WIDE}}")
    _flush(f"  Dataset [{index}/{total}]: {name}")
    _flush(f"  Task    : {task}")
    _flush(f"  Started : {_ts()}")
    _flush(f"  Phases  : {phases}")
    _flush(f"  Report  : {report_path}")
    _flush()
    _flush(f"{'─' * 19} BENCHMARKING LOGS {'─' * 19}")
    _flush()


def _dataset_footer() -> None:
    _flush()
    _flush(THICK * WIDE)
    _flush()


def _dataset_failure(name: str, elapsed: float) -> None:
    _flush(f"  ✗  {name}  (failed after {elapsed:.0f}s — see stderr for traceback)")
    _flush()


# ── YAML parsing ──────────────────────────────────────────────────────────────

def normalise_task(task: str) -> str:
    task = TASK_ALIASES.get(task, task)
    if task not in TASK_PIPELINES:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(TASK_PIPELINES)}")
    return task


def load_yaml(yaml_path: Path) -> tuple[dict, list[dict]]:
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)

    merged_defaults = {**BUILTIN_DEFAULTS, **(raw.get("defaults") or {})}
    # Task blocks merge key by key, so the yaml only has to list what it changes
    for section in list(TASK_PIPELINES) + list(TASK_ALIASES):
        target = normalise_task(section)
        if section in (raw.get("defaults") or {}):
            merged_defaults[target] = {**BUILTIN_DEFAULTS.get(target, {}), **(raw["defaults"][section] or {})}
    for alias in TASK_ALIASES:
        merged_defaults.pop(alias, None)

    return merged_defaults, raw.get("datasets", []) or []


def resolve_entry(defaults: dict, entry: dict) -> dict:
    """Entry keys win over the task block of the defaults, which wins over the shared defaults."""
    task = normalise_task(entry.get("task", defaults.get("task", "image_classification")))
    shared = {key: value for key, value in defaults.items() if key not in TASK_PIPELINES}
    cfg = {**shared, **defaults.get(task, {}), **entry, "task": task}

    # Phases: "1 2 3" → [1, 2, 3]
    phases_raw = cfg.get("phases", "1 2 3")
    if isinstance(phases_raw, str):
        cfg["phases"] = [int(p) for p in phases_raw.split()]

    # Compound label cols: list[str] or None
    compound = cfg.get("compound_label_cols") or []
    if isinstance(compound, str):
        compound = compound.split() if compound.strip() else []
    cfg["compound_label_cols"] = compound if compound else None

    # Empty strings → None
    cfg["hf_config_name"] = cfg.get("hf_config_name") or None

    return cfg


def build_pipeline(dataset_name: str, cfg: dict, pipeline_cfg: PipelineConfig):
    task_pipeline = TASK_PIPELINES[cfg["task"]]
    arguments = {argument: cfg.get(key) for argument, key in task_pipeline.arguments.items() if key in cfg}
    return task_pipeline.pipeline_class(dataset_name, cfg=pipeline_cfg, show_banner=False, **arguments)


# ── Selection ─────────────────────────────────────────────────────────────────

def select_entries(entries: List[dict], defaults: dict, dataset: Optional[str], task: Optional[str]) -> List[dict]:
    selected = entries
    if dataset is not None:
        selected = [entry for entry in selected if entry.get("name", "").strip() == dataset]
        if not selected:
            raise ValueError(f"Dataset '{dataset}' is not in the yaml.")
    if task is not None:
        wanted = normalise_task(task)
        selected = [entry for entry in selected if resolve_entry(defaults, entry)["task"] == wanted]
        if not selected:
            raise ValueError(f"No yaml entry has task '{wanted}'.")
    return selected


# ── Main runner ───────────────────────────────────────────────────────────────

def run_entries(
    yaml_path: Path,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
    phases: Optional[List[int]] = None,
    prefetch_only: bool = False,
) -> int:
    defaults, all_entries = load_yaml(yaml_path)
    entries = select_entries(all_entries, defaults, dataset, task)

    total      = len(entries)
    succeeded  = []
    failed     = []

    _flush()
    _flush(THICK * WIDE)
    _flush(f"  AgML Benchmark  —  {total} dataset(s) queued" + ("  (prefetch only)" if prefetch_only else ""))
    _flush(f"  YAML   : {yaml_path}")
    _flush(f"  Started: {_ts()}")
    _flush(THICK * WIDE)
    _flush()

    # Load PipelineConfig once — all datasets share the same infrastructure config
    pipeline_cfg = PipelineConfig.load()

    for index, entry in enumerate(entries, start=1):
        dataset_name = entry.get("name", "").strip()
        if not dataset_name:
            _flush(f"  ⚠  Entry {index} is missing a 'name' field — skipping.")
            continue

        cfg = resolve_entry(defaults, entry)
        if phases is not None:
            cfg["phases"] = phases

        start = datetime.now()
        path = None
        try:
            pipeline = build_pipeline(dataset_name, cfg, pipeline_cfg)

            if prefetch_only:
                _flush(f"  [{index}/{total}] prefetch {dataset_name}  ({cfg['task']})")
                pipeline.prefetch()
                succeeded.append(dataset_name)
                continue

            # Resolve report path before run() so we can show it in the header
            path = pipeline.report_path()
            _dataset_header(index, total, dataset_name, cfg["task"], cfg["phases"], path)

            pipeline.run(phases=cfg["phases"])

            _dataset_footer()
            succeeded.append(dataset_name)

        except Exception:
            elapsed = (datetime.now() - start).total_seconds()

            # Full traceback to stderr so it ends up in the .err file
            print(f"\n[ERROR] {dataset_name}  ({_ts()})", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            print(file=sys.stderr, flush=True)

            if path is not None:
                # Remove the incomplete run directory so no partial data lingers
                run_dir = Path(path).parent
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    print(f"  Removed incomplete run dir: {run_dir}", file=sys.stderr, flush=True)

                # Also remove the dataset parent dir if it's now empty (the task dir stays)
                dataset_dir = run_dir.parent
                if dataset_dir.exists() and not any(dataset_dir.iterdir()):
                    dataset_dir.rmdir()
                    print(f"  Removed empty dataset dir: {dataset_dir}", file=sys.stderr, flush=True)

            _dataset_footer()
            _dataset_failure(dataset_name, elapsed)
            failed.append(dataset_name)

    # ── Final summary ─────────────────────────────────────────────────────────
    _flush()
    _flush(THICK * WIDE)
    _flush(f"  Finished: {_ts()}")
    _flush(f"  Results : {len(succeeded)}/{total} succeeded,  {len(failed)}/{total} failed")
    if failed:
        _flush()
        _flush("  Failed datasets:")
        for name in failed:
            _flush(f"    ✗  {name}")
    _flush(THICK * WIDE)
    _flush()

    # Non-zero exit so SLURM marks the job as failed if anything went wrong
    return 1 if failed else 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgML benchmark for the datasets in datasets.yaml")
    parser.add_argument("--yaml", default=str(DEFAULT_YAML), metavar="PATH", help=f"Path to datasets.yaml  (default: {DEFAULT_YAML})")
    parser.add_argument("--dataset", default=None, metavar="NAME", help="Run only this yaml entry (exact name)")
    parser.add_argument("--task", default=None, choices=sorted(TASK_PIPELINES) + sorted(TASK_ALIASES), help="Run only the entries of this task")
    parser.add_argument("--phases", nargs="+", type=int, default=None, help="Phases to run, overrides the yaml. Example: --phases 1 2")
    parser.add_argument(
        "--prefetch-only", action="store_true",
        help="Download the datasets and model weights of the selected entries into the caches, then exit",
    )
    args = parser.parse_args(argv)

    try:
        return run_entries(Path(args.yaml), args.dataset, args.task, args.phases, args.prefetch_only)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
