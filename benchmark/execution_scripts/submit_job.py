#!/usr/bin/env python3
"""
submit_job.py
=============
Submits a single SLURM job that runs the full benchmark pipeline for every
dataset listed in datasets.yaml.

Usage (from repo root):
    python benchmark/execution_scripts/submit_job.py
    python benchmark/execution_scripts/submit_job.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[2]
SBATCH_SCRIPT = REPO_ROOT / "benchmark" / "execution_scripts" / "run_benchmark.sbatch"
LOGS_BASE    = REPO_ROOT / "benchmark" / "logs"


def submit(dry_run: bool = False) -> None:
    # Date-stamped log directory so runs don't overwrite each other
    log_dir = LOGS_BASE / date.today().isoformat()
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        f"--output={log_dir}/agbenchmark_all_%j.out",
        f"--error={log_dir}/agbenchmark_all_%j.err",
        str(SBATCH_SCRIPT),
    ]

    env = os.environ.copy()
    env["REPO_ROOT"] = str(REPO_ROOT)

    print(f"SLURM script : {SBATCH_SCRIPT}")
    print(f"Logs         : {log_dir}/")
    print(f"REPO_ROOT    : {REPO_ROOT}")
    print()

    if dry_run:
        print("[dry-run] Would run:", " ".join(cmd))
        return

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print("sbatch failed:", result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)

    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted job {job_id}")
    print(f"  stdout → {log_dir}/agbenchmark_all_{job_id}.out")
    print(f"  stderr → {log_dir}/agbenchmark_all_{job_id}.err")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit one SLURM job to benchmark all datasets in datasets.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command without actually submitting",
    )
    args = parser.parse_args()
    submit(dry_run=args.dry_run)
