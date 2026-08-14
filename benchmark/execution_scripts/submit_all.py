#!/usr/bin/env python3
"""
Submit one SLURM job per dataset listed in datasets.yaml.
SLURM handles scheduling and prioritization — jobs are submitted as-is.

Usage (run from repo root):
	python benchmark/execution_scripts/submit_all.py
	python benchmark/execution_scripts/submit_all.py --dry-run
	python benchmark/execution_scripts/submit_all.py --config path/to/datasets.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
	import yaml
except ImportError:
	sys.exit("PyYAML is required.  Install with:  pip install pyyaml")

REPO_ROOT     = Path(__file__).resolve().parents[2]
SBATCH_SCRIPT = REPO_ROOT / "benchmark" / "execution_scripts" / "run_benchmark.sbatch"
DEFAULT_CONFIG = Path(__file__).parent / "datasets.yaml"


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: Path) -> tuple[dict, list[dict]]:
	with open(path) as fh:
		raw = yaml.safe_load(fh)
	return raw.get("defaults", {}), raw.get("datasets", [])


def merge(defaults: dict, entry: dict) -> dict:
	"""Merge a dataset entry with defaults. Entry values take priority."""
	return {**defaults, **entry}


def short_name(full_name: str) -> str:
	"""Strip org prefix — 'Project-AgML/foo' → 'foo'."""
	return full_name.split("/")[-1]


# ── Job submission ────────────────────────────────────────────────────────────

def submit(cfg: dict, dry_run: bool) -> str:
	"""
	Build and run the sbatch command for one dataset.
	Returns the submitted SLURM job ID (or 'DRY_RUN').
	"""

	name     = cfg["name"]
	job_name = f"agbenchmark_{short_name(name)}"

	# Create the logs dir here so the sbatch --output path always exists
	logs_dir = REPO_ROOT / "benchmark/logs"
	logs_dir.mkdir(exist_ok=True)

	cmd = [
		"sbatch",
		f"--job-name={job_name}",
		f"--mem={cfg.get('mem', '32G')}",
		f"--time={cfg.get('time', '5:00:00')}",
		f"--output={logs_dir}/%x_%j.out",
		f"--error={logs_dir}/%x_%j.err",
		str(SBATCH_SCRIPT),
	]

	# Dataset variables are passed via the subprocess environment.
	# SLURM's default --export=ALL forwards them to the batch job.
	compound = cfg.get("compound_label_cols", [])
	env = os.environ.copy()
	env.update({
		"REPO_ROOT":           str(REPO_ROOT),
		"DATASET":             name,
		"PHASES":              cfg.get("phases", "1 2 3"),
		"HF_CONFIG_NAME":      cfg.get("hf_config_name", ""),
		"LABEL_COL":           cfg.get("label_col", "label"),
		"IMAGE_COL":           cfg.get("image_col", "image"),
		"COMPOUND_LABEL_COLS": " ".join(compound) if compound else "",
	})

	print(f"  Dataset : {name}")
	print(f"  Job     : {job_name}")
	print(f"  Mem     : {cfg.get('mem', '32G')}   Time: {cfg.get('time', '5:00:00')}")

	if dry_run:
		print("  [dry-run — not submitted]\n")
		return "DRY_RUN"

	result = subprocess.run(cmd, capture_output=True, text=True, env=env)
	if result.returncode != 0:
		sys.exit(f"  sbatch failed:\n  {result.stderr.strip()}")

	# SLURM prints "Submitted batch job 12345"
	job_id = result.stdout.strip().split()[-1]
	print(f"  Job ID  : {job_id}\n")
	return job_id


# Entry point
def main() -> None:
	parser = argparse.ArgumentParser(
		description="Submit AgML benchmark jobs for all datasets in datasets.yaml"
	)
	
	parser.add_argument(
		"--config",
		type=Path,
		default=DEFAULT_CONFIG,
		help="Path to datasets.yaml (default: execution_scripts/datasets.yaml)",
	)

	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print what would be submitted without actually submitting",
	)

	args = parser.parse_args()

	defaults, datasets = load_config(args.config)

	if not datasets:
		sys.exit("No datasets found in the config file.")

	print(f"Submitting {len(datasets)} dataset(s)\n")

	for entry in datasets:
		cfg = merge(defaults, entry)
		submit(cfg, args.dry_run)

	print(f"Done — {len(datasets)} job(s) {'queued' if not args.dry_run else 'would be queued'}.")
	if not args.dry_run:
		print("Monitor with:  squeue -u $USER")


if __name__ == "__main__":
	main()
