#!/usr/bin/env python3
"""
run_all.py
==========
Reads datasets.yaml and runs the benchmark pipeline for every listed dataset
sequentially. A dataset that fails (download error, corrupt data, pipeline bug)
is logged to stderr and skipped — the rest of the queue keeps running.

stdout  →  progress banners, metric summaries, per-dataset results  (SLURM .out)
stderr  →  full tracebacks, prefixed with the dataset name           (SLURM .err)

Usage (from repo root, or via SLURM):
    uv run python benchmark/execution_scripts/run_all.py
    uv run python benchmark/execution_scripts/run_all.py \
        --yaml benchmark/execution_scripts/datasets.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Repo root is two levels up from this file:
#   benchmark/execution_scripts/run_all.py  →  benchmark/  →  repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.app import AgMLBenchmarkPipeline  # noqa: E402
from benchmark.config import PipelineConfig       # noqa: E402

DEFAULT_YAML = REPO_ROOT / "benchmark" / "execution_scripts" / "datasets.yaml"

# Fields that every dataset entry inherits if not explicitly set
FIELD_DEFAULTS: dict[str, Any] = {
    "phases":              "1 2 3",
    "label_col":           "label",
    "image_col":           "image",
    "hf_config_name":      "",
    "compound_label_cols": [],
}

WIDE  = 70
THICK = "═"
THIN  = "─"


# ── Logging helpers ───────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _flush(*args, **kwargs):
    """print() that always flushes — important inside long SLURM jobs."""
    print(*args, **kwargs, flush=True)


def _dataset_header(index: int, total: int, name: str) -> None:
    _flush()
    _flush(THICK * WIDE)
    _flush(f"  Dataset [{index}/{total}]  {name}")
    _flush(f"  Started: {_ts()}")
    _flush(THICK * WIDE)


def _dataset_success(name: str, report_path: str, elapsed: float) -> None:
    _flush()
    _flush(THIN * WIDE)
    _flush(f"  ✓  {name}")
    _flush(f"     report  : {report_path}")
    _flush(f"     elapsed : {elapsed:.0f}s")
    _flush(THIN * WIDE)
    _flush()


def _dataset_failure(name: str, elapsed: float) -> None:
    _flush()
    _flush(THIN * WIDE)
    _flush(f"  ✗  {name}  (failed after {elapsed:.0f}s — see stderr for traceback)")
    _flush(THIN * WIDE)
    _flush()


# ── YAML parsing ──────────────────────────────────────────────────────────────

def load_yaml(yaml_path: Path) -> tuple[dict, list[dict]]:
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)

    yaml_defaults = raw.get("defaults", {})
    # YAML defaults override hard-coded field defaults
    merged_defaults = {**FIELD_DEFAULTS, **yaml_defaults}

    dataset_entries = raw.get("datasets", [])
    return merged_defaults, dataset_entries


def resolve_entry(defaults: dict, entry: dict) -> dict:
    """Merge per-dataset overrides on top of defaults and normalise types."""
    cfg = {**defaults, **entry}

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


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(yaml_path: Path) -> None:
    defaults, entries = load_yaml(yaml_path)

    total      = len(entries)
    succeeded  = []
    failed     = []

    _flush()
    _flush(THICK * WIDE)
    _flush(f"  AgML Benchmark  —  {total} dataset(s) queued")
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
        _dataset_header(index, total, dataset_name)

        start = datetime.now()
        try:
            pipeline = AgMLBenchmarkPipeline(
                dataset_name        = dataset_name,
                config_name         = cfg["hf_config_name"],
                label_col           = cfg.get("label_col"),
                image_col           = cfg.get("image_col"),
                compound_label_cols = cfg["compound_label_cols"],
                cfg                 = pipeline_cfg,
            )
            report_path = pipeline.run(phases=cfg["phases"])
            elapsed = (datetime.now() - start).total_seconds()

            _dataset_success(dataset_name, report_path, elapsed)
            succeeded.append(dataset_name)

        except Exception:
            elapsed = (datetime.now() - start).total_seconds()

            # Full traceback to stderr so it ends up in the .err file
            print(f"\n[ERROR] {dataset_name}  ({_ts()})", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            print(file=sys.stderr, flush=True)

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
    if failed:
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AgML benchmark pipeline for all datasets in datasets.yaml"
    )
    parser.add_argument(
        "--yaml",
        default=str(DEFAULT_YAML),
        metavar="PATH",
        help=f"Path to datasets.yaml  (default: {DEFAULT_YAML})",
    )
    args = parser.parse_args()
    run_all(Path(args.yaml))
