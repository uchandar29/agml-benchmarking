"""
Experiment 2: zero-shot top-1 accuracy on the held-out test split, no training.

Each model runs in its own process (zero_shot.sbatch loops over them), because
vLLM does not reliably give GPU memory back when a second model is loaded.

Usage (from repo root):
	uv run python -m experiments.zero_shot.run --models siglip_so400m
	uv run python -m experiments.zero_shot.run --models qwen_3_5_27b --datasets bean_disease_uganda
	uv run python -m experiments.zero_shot.run --prefetch-only    # login node, downloads weights + datasets
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common import (
	PreparedDataset,
	load_dataset_with_schema,
	load_experiment_config,
	parse_dataset_entries,
	prepare_dataset,
	set_global_seed,
)
from experiments.common.class_names import build_display_class_names
from experiments.common.classification_metrics import compute_classification_metrics
from experiments.common.output_files import write_json
from experiments.zero_shot.results_summary import rebuild_summary

SUPPORTED_MODEL_TYPES = ("siglip", "vision_language_model")
IMAGE_READ_CHUNK = 256


def log(message: str = "") -> None:
	print(message, flush=True)


def timestamp() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Experiment 2: zero-shot performance")
	parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to experiments/config.yaml)")
	parser.add_argument("--datasets", default=None, help="Comma separated dataset names to run (full or short names)")
	parser.add_argument("--models", default=None, help="Comma separated model names from zero_shot.models")
	parser.add_argument("--prefetch-only", action="store_true", help="Download datasets and weights, then exit")
	return parser.parse_args()


def split_comma_list(raw_value: Optional[str]) -> Optional[List[str]]:
	if not raw_value:
		return None
	return [item.strip() for item in raw_value.split(",") if item.strip()]


def select_models(zero_shot_settings: Dict[str, Any], selected_model_names: Optional[List[str]]) -> List[Dict[str, Any]]:
	configured_models = zero_shot_settings["models"]
	for model_settings in configured_models:
		if model_settings["model_type"] not in SUPPORTED_MODEL_TYPES:
			raise ValueError(f"{model_settings['name']}: model_type must be one of {SUPPORTED_MODEL_TYPES}")
	if not selected_model_names:
		return configured_models
	known_model_names = {model_settings["name"] for model_settings in configured_models}
	unknown_model_names = set(selected_model_names) - known_model_names
	if unknown_model_names:
		raise ValueError(f"Models not found in config.yaml: {sorted(unknown_model_names)}")
	return [model_settings for model_settings in configured_models if model_settings["name"] in selected_model_names]


def build_classifier(model_settings: Dict[str, Any], zero_shot_settings: Dict[str, Any], random_seed: int):
	# Imported here so a SigLIP run does not need vLLM installed, and the other way round
	if model_settings["model_type"] == "siglip":
		from experiments.zero_shot.siglip_classifier import SiglipZeroShotClassifier

		return SiglipZeroShotClassifier(model_settings)
	from experiments.zero_shot.vlm_classifier import VisionLanguageZeroShotClassifier

	return VisionLanguageZeroShotClassifier(model_settings, zero_shot_settings["vision_language_prompt"], random_seed)


def predict_split(classifier, hf_split, image_column: str, label_column: str):
	# Reads the split in chunks so large-image datasets never sit in memory all at once
	true_labels, predicted_labels = [], []
	for chunk_start in range(0, len(hf_split), IMAGE_READ_CHUNK):
		chunk = hf_split[chunk_start:chunk_start + IMAGE_READ_CHUNK]
		chunk_images = [image.convert("RGB") if image.mode != "RGB" else image for image in chunk[image_column]]
		chunk_predictions = classifier.predict(chunk_images)
		if len(chunk_predictions) != len(chunk_images):
			raise RuntimeError(f"Got {len(chunk_predictions)} predictions for {len(chunk_images)} images")
		true_labels.extend(int(label) for label in chunk[label_column])
		predicted_labels.extend(chunk_predictions)
	return true_labels, predicted_labels


def run_single_dataset(classifier, prepared_dataset: PreparedDataset, model_settings: Dict[str, Any], save_predictions: bool, run_output_dir: Path) -> Dict[str, Any]:
	from experiments.zero_shot.vlm_classifier import INVALID_PREDICTION

	schema = prepared_dataset.schema
	display_class_names = build_display_class_names(schema.label_names, prepared_dataset.entry.class_name_overrides)
	classifier.set_class_names(display_class_names)

	start_time = time.time()
	true_labels, predicted_labels = predict_split(classifier, prepared_dataset.splits.test, schema.image_col, schema.label_col)
	inference_seconds = round(time.time() - start_time, 1)

	invalid_predictions = sum(1 for label in predicted_labels if label == INVALID_PREDICTION)
	# The confusion matrix only has columns for real classes, so invalid answers are counted here per true class
	invalid_predictions_per_class = {
		class_name: sum(
			1 for true_label, predicted_label in zip(true_labels, predicted_labels)
			if true_label == class_index and predicted_label == INVALID_PREDICTION
		)
		for class_index, class_name in enumerate(schema.label_names)
	}
	test_metrics = compute_classification_metrics(true_labels, predicted_labels, schema.label_names)
	run_result = {
		"dataset": prepared_dataset.entry.name,
		"model": {"name": model_settings["name"], "model_id": model_settings["model_id"], "model_type": model_settings["model_type"]},
		"number_of_classes": schema.num_classes,
		"class_names": schema.label_names,
		"display_class_names": display_class_names,
		"split_sizes": prepared_dataset.split_sizes,
		"test_metrics": test_metrics,
		"invalid_predictions": invalid_predictions,
		"invalid_predictions_per_class": invalid_predictions_per_class,
		"inference_seconds": inference_seconds,
		"settings": classifier.describe(),
		"finished_at": timestamp(),
	}
	if save_predictions:
		test_row_indices = prepared_dataset.split_indices["test"].tolist()
		write_json(run_output_dir / "predictions.json", [
			{"dataset_row": int(row_index), "true_label": true_label, "predicted_label": predicted_label}
			for row_index, true_label, predicted_label in zip(test_row_indices, true_labels, predicted_labels)
		])
	# Written last: its presence marks the run as complete for skip_completed_runs
	write_json(run_output_dir / "run_result.json", run_result)
	return run_result


def prefetch_everything(dataset_entries, model_settings_list) -> None:
	# Only fills the HF cache. Splitting happens later, inside the evaluation run.
	from huggingface_hub import snapshot_download

	for model_settings in model_settings_list:
		log(f"Downloading weights for {model_settings['name']} ({model_settings['model_id']})")
		snapshot_download(model_settings["model_id"])
	for dataset_entry in dataset_entries:
		log(f"Caching dataset {dataset_entry.name}")
		load_dataset_with_schema(dataset_entry)
	log("Prefetch complete.")


def main() -> None:
	arguments = parse_arguments()
	experiment_config = load_experiment_config(arguments.config)
	shared_settings = experiment_config["shared"]
	zero_shot_settings = experiment_config["zero_shot"]

	dataset_entries = parse_dataset_entries(experiment_config, split_comma_list(arguments.datasets))
	model_settings_list = select_models(zero_shot_settings, split_comma_list(arguments.models))
	experiment_output_dir = Path(shared_settings["output_root"]) / zero_shot_settings["experiment_name"]
	experiment_output_dir.mkdir(parents=True, exist_ok=True)

	log(f"Experiment 2 (zero-shot) started {timestamp()}")
	log(f"  datasets : {len(dataset_entries)}")
	log(f"  models   : {[model_settings['name'] for model_settings in model_settings_list]}")
	log(f"  output   : {experiment_output_dir}")
	log()

	if arguments.prefetch_only:
		prefetch_everything(dataset_entries, model_settings_list)
		return

	random_seed = shared_settings["training_seed"]
	completed_runs, skipped_runs, failed_runs = [], [], []
	for model_settings in model_settings_list:
		model_name = model_settings["name"]
		pending_entries = [
			entry for entry in dataset_entries
			if not (
				zero_shot_settings.get("skip_completed_runs", True)
				and (experiment_output_dir / entry.short_name / model_name / "run_result.json").exists()
			)
		]
		skipped_runs.extend(f"{entry.short_name}/{model_name}" for entry in dataset_entries if entry not in pending_entries)
		if not pending_entries:
			log(f"{model_name}: all datasets already done, skipping")
			continue

		log("=" * 70)
		log(f"Loading {model_name} ({model_settings['model_id']})  {timestamp()}")
		set_global_seed(random_seed)
		try:
			classifier = build_classifier(model_settings, zero_shot_settings, random_seed)
		except Exception:
			print(f"\n[ERROR] loading model {model_name}", file=sys.stderr, flush=True)
			traceback.print_exc(file=sys.stderr)
			failed_runs.extend(f"{entry.short_name}/{model_name}" for entry in pending_entries)
			continue

		for dataset_position, dataset_entry in enumerate(pending_entries, start=1):
			run_label = f"{dataset_entry.short_name}/{model_name}"
			log(f"[{dataset_position}/{len(pending_entries)}] {dataset_entry.name}  ({timestamp()})")
			try:
				prepared_dataset = prepare_dataset(dataset_entry, shared_settings["split"])
				run_result = run_single_dataset(
					classifier,
					prepared_dataset,
					model_settings,
					zero_shot_settings.get("save_predictions", True),
					experiment_output_dir / dataset_entry.short_name / model_name,
				)
				test_metrics = run_result["test_metrics"]
				log(
					f"     accuracy={test_metrics['accuracy']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}  "
					f"invalid={run_result['invalid_predictions']}  ({run_result['inference_seconds']}s)"
				)
				completed_runs.append(run_label)
			except Exception:
				print(f"\n[ERROR] {run_label}  ({timestamp()})", file=sys.stderr, flush=True)
				traceback.print_exc(file=sys.stderr)
				failed_runs.append(run_label)

	summary_path = rebuild_summary(experiment_output_dir)
	log()
	log("=" * 70)
	log(f"Finished {timestamp()}")
	log(f"  completed: {len(completed_runs)}  skipped: {len(skipped_runs)}  failed: {len(failed_runs)}")
	for run_label in failed_runs:
		log(f"    failed: {run_label}")
	log(f"  summary  : {summary_path}")

	if failed_runs:
		sys.exit(1)


if __name__ == "__main__":
	main()
