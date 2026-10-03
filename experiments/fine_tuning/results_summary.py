from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

SUMMARY_COLUMNS = [
	"dataset", "model", "macro_f1", "weighted_f1", "accuracy",
	"best_epoch", "epochs_completed", "train_size", "val_size", "test_size", "number_of_classes",
]


def write_json(output_path: Path, payload: Any) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as output_file:
		json.dump(payload, output_file, indent=4)


def rebuild_summary(experiment_output_dir: Path) -> Path:
	"""Collects every run_result.json under the experiment folder, so resumed jobs still give one summary."""
	summary_rows: List[Dict[str, Any]] = []
	for result_path in sorted(experiment_output_dir.glob("*/*/run_result.json")):
		with open(result_path) as result_file:
			run_result = json.load(result_file)
		summary_rows.append({
			"dataset": run_result["dataset"],
			"model": run_result["model"]["name"],
			"macro_f1": run_result["test_metrics"]["macro_f1"],
			"weighted_f1": run_result["test_metrics"]["weighted_f1"],
			"accuracy": run_result["test_metrics"]["accuracy"],
			"best_epoch": run_result["training"]["best_epoch"],
			"epochs_completed": run_result["training"]["epochs_completed"],
			"train_size": run_result["split_sizes"]["train"],
			"val_size": run_result["split_sizes"]["val"],
			"test_size": run_result["split_sizes"]["test"],
			"number_of_classes": run_result["number_of_classes"],
		})

	summary_csv_path = experiment_output_dir / "summary.csv"
	with open(summary_csv_path, "w", newline="") as summary_file:
		writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
		writer.writeheader()
		writer.writerows(summary_rows)
	write_json(experiment_output_dir / "summary.json", summary_rows)
	return summary_csv_path
