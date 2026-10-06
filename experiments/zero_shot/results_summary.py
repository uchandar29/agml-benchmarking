from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from experiments.common.output_files import write_json

SUMMARY_COLUMNS = [
	"dataset", "model", "accuracy", "macro_f1", "weighted_f1",
	"invalid_predictions", "test_size", "number_of_classes",
]


def rebuild_summary(experiment_output_dir: Path) -> Path:
	"""Collects every run_result.json under the experiment folder into one summary."""
	summary_rows: List[Dict[str, Any]] = []
	for result_path in sorted(experiment_output_dir.glob("*/*/run_result.json")):
		with open(result_path) as result_file:
			run_result = json.load(result_file)
		summary_rows.append({
			"dataset": run_result["dataset"],
			"model": run_result["model"]["name"],
			"accuracy": run_result["test_metrics"]["accuracy"],
			"macro_f1": run_result["test_metrics"]["macro_f1"],
			"weighted_f1": run_result["test_metrics"]["weighted_f1"],
			"invalid_predictions": run_result["invalid_predictions"],
			"test_size": run_result["test_metrics"]["number_of_test_samples"],
			"number_of_classes": run_result["number_of_classes"],
		})

	summary_csv_path = experiment_output_dir / "summary.csv"
	with open(summary_csv_path, "w", newline="") as summary_file:
		writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
		writer.writeheader()
		writer.writerows(summary_rows)
	write_json(experiment_output_dir / "summary.json", summary_rows)
	return summary_csv_path
