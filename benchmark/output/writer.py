"""
ReportWriter
============
Accumulates metric results across phases and writes a single JSON report
per pipeline run.

File layout
-----------
    results/
    └── <dataset_name>/          (slashes in dataset name replaced with '__')
        └── <run_id>/            (YYYYMMDD_HHMMSS)
            ├── report.json      ← the report written here
            └── splits.json      ← written by SplitManager

The report is flushed to disk after every metric is added so a run that
is interrupted mid-way still leaves a partial-but-valid JSON file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


class ReportWriter:
    """
    Usage::

        writer = ReportWriter("Project-AgML/rice_leaf_disease_classification")
        writer.add("class_imbalance", {...})
        writer.add("exact_duplicate", {...})
        writer.complete_phase(1)
        print(writer.path())   # path to the JSON file
    """

    def __init__(self, dataset_name: str, output_dir: str = "results") -> None:
        # Sanitise dataset name for use as a directory component
        safe_name = dataset_name.replace("/", "__").replace("\\", "__")

        self.run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, safe_name, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self._report: Dict[str, Any] = {
            "dataset": dataset_name,
            "run_id": self.run_id,
            "phases_completed": [],
            "metrics": {},
        }
        # Write an empty skeleton immediately so the directory is non-empty
        # even before any metric finishes.
        self._flush()

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, metric_name: str, result: Dict[str, Any]) -> None:
        """
        Record the result dict for *metric_name* and flush to disk.

        Can be called multiple times; later calls for the same metric_name
        overwrite the earlier entry.
        """
        self._report["metrics"][metric_name] = result
        self._flush()

    def complete_phase(self, phase: int) -> None:
        """Mark a phase as finished and flush."""
        if phase not in self._report["phases_completed"]:
            self._report["phases_completed"].append(phase)
        self._flush()

    def path(self) -> str:
        """Absolute path to the report JSON file."""
        return os.path.join(self.run_dir, "report.json")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        with open(self.path(), "w") as fh:
            json.dump(self._report, fh, indent=2, default=str)
