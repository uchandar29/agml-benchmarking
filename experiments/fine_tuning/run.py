"""
Experiment 1: fine-tune every configured model on every configured dataset and
score macro F1 on the held-out test split.

Usage (from repo root):
	uv run python -m experiments.fine_tuning.run
	uv run python -m experiments.fine_tuning.run --datasets bean_disease_uganda --models vit_small
	uv run python -m experiments.fine_tuning.run --prefetch-only    # run on the login node first
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common import (
	ImageClassificationDataset,
	PreparedDataset,
	load_experiment_config,
	parse_dataset_entries,
	prepare_dataset,
	set_global_seed,
)
from experiments.common.reproducibility import seed_dataloader_worker
from experiments.fine_tuning.evaluation import compute_classification_metrics
from experiments.fine_tuning.image_transforms import build_evaluation_transform, build_training_transform
from experiments.fine_tuning.model_factory import (
	build_classifier,
	download_pretrained_weights,
	get_normalization_statistics,
)
from experiments.fine_tuning.results_summary import rebuild_summary, write_json
from experiments.fine_tuning.trainer import FineTuningTrainer


def log(message: str = "") -> None:
	print(message, flush=True)


def timestamp() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Experiment 1: fine-tuned performance")
	parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to experiments/config.yaml)")
	parser.add_argument("--datasets", default=None, help="Comma separated dataset names to run (full or short names)")
	parser.add_argument("--models", default=None, help="Comma separated model names from fine_tuning.models")
	parser.add_argument("--prefetch-only", action="store_true", help="Download datasets and weights, then exit")
	return parser.parse_args()


def split_comma_list(raw_value: Optional[str]) -> Optional[List[str]]:
	if not raw_value:
		return None
	return [item.strip() for item in raw_value.split(",") if item.strip()]


def select_models(fine_tuning_settings: Dict[str, Any], selected_model_names: Optional[List[str]]) -> List[Dict[str, Any]]:
	configured_models = fine_tuning_settings["models"]
	if not selected_model_names:
		return configured_models
	known_model_names = {model_settings["name"] for model_settings in configured_models}
	unknown_model_names = set(selected_model_names) - known_model_names
	if unknown_model_names:
		raise ValueError(f"Models not found in config.yaml: {sorted(unknown_model_names)}")
	return [model_settings for model_settings in configured_models if model_settings["name"] in selected_model_names]


def build_data_loader(dataset, batch_size: int, shuffle: bool, worker_count: int, loader_seed: int):
	import torch

	random_generator = torch.Generator()
	random_generator.manual_seed(loader_seed)
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=worker_count,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=worker_count > 0,
		worker_init_fn=seed_dataloader_worker,
		generator=random_generator,
	)


def run_single_model(
	prepared_dataset: PreparedDataset,
	model_settings: Dict[str, Any],
	fine_tuning_settings: Dict[str, Any],
	shared_settings: Dict[str, Any],
	run_output_dir: Path,
) -> Dict[str, Any]:
	import torch

	# Per model overrides (learning_rate, batch_size, ...) sit on top of the shared fine_tuning values
	training_settings = {**fine_tuning_settings, **model_settings}
	training_seed = shared_settings["training_seed"]
	image_size = training_settings["image_size"]
	batch_size = training_settings["batch_size"]
	worker_count = shared_settings["dataloader_workers"]
	schema = prepared_dataset.schema

	set_global_seed(training_seed)
	model = build_classifier(model_settings["timm_model"], schema.num_classes, image_size)
	normalization_mean, normalization_std = get_normalization_statistics(model)

	training_transform = build_training_transform(
		image_size, training_settings["augmentation"], normalization_mean, normalization_std
	)
	evaluation_transform = build_evaluation_transform(image_size, normalization_mean, normalization_std)

	def make_split_dataset(hf_split, image_transform):
		return ImageClassificationDataset(hf_split, schema.image_col, schema.label_col, image_transform)

	train_loader = build_data_loader(
		make_split_dataset(prepared_dataset.splits.train, training_transform),
		batch_size, shuffle=True, worker_count=worker_count, loader_seed=training_seed,
	)
	validation_loader = build_data_loader(
		make_split_dataset(prepared_dataset.splits.val, evaluation_transform),
		batch_size, shuffle=False, worker_count=worker_count, loader_seed=training_seed,
	)
	test_loader = build_data_loader(
		make_split_dataset(prepared_dataset.splits.test, evaluation_transform),
		batch_size, shuffle=False, worker_count=worker_count, loader_seed=training_seed,
	)

	trainer = FineTuningTrainer(model, training_settings)
	training_outcome = trainer.fit(train_loader, validation_loader)
	true_labels, predicted_labels = trainer.predict(test_loader)
	test_metrics = compute_classification_metrics(true_labels, predicted_labels, schema.label_names)

	run_output_dir.mkdir(parents=True, exist_ok=True)
	if fine_tuning_settings.get("save_best_checkpoint", False):
		torch.save(trainer.model.state_dict(), run_output_dir / "best_model.pt")

	run_result = {
		"dataset": prepared_dataset.entry.name,
		"model": {"name": model_settings["name"], "timm_model": model_settings["timm_model"]},
		"number_of_classes": schema.num_classes,
		"class_names": schema.label_names,
		"split_sizes": prepared_dataset.split_sizes,
		"test_metrics": test_metrics,
		"training": {
			"best_epoch": training_outcome.best_epoch,
			"best_validation_loss": training_outcome.best_validation_loss,
			"epochs_completed": training_outcome.epochs_completed,
			"stopped_early": training_outcome.stopped_early,
			"training_seconds": training_outcome.training_seconds,
		},
		"settings": {
			"image_size": image_size,
			"batch_size": batch_size,
			"learning_rate": training_settings["learning_rate"],
			"weight_decay": training_settings["weight_decay"],
			"max_epochs": training_settings["max_epochs"],
			"early_stopping_patience": training_settings["early_stopping_patience"],
			"warmup_epochs": training_settings.get("warmup_epochs", 0),
			"augmentation": training_settings["augmentation"],
			"split": shared_settings["split"],
			"training_seed": training_seed,
			"normalization_mean": list(normalization_mean),
			"normalization_std": list(normalization_std),
		},
		"finished_at": timestamp(),
	}
	write_json(run_output_dir / "training_history.json", training_outcome.history)
	# Written last: its presence marks the run as complete for skip_completed_runs
	write_json(run_output_dir / "run_result.json", run_result)

	del trainer, model
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return run_result


def prefetch_everything(dataset_entries, model_settings_list, shared_settings, experiment_output_dir: Path) -> None:
	for model_settings in model_settings_list:
		log(f"Downloading weights for {model_settings['name']} ({model_settings['timm_model']})")
		download_pretrained_weights(model_settings["timm_model"])
	for dataset_entry in dataset_entries:
		log(f"Caching dataset {dataset_entry.name}")
		prepare_dataset(dataset_entry, shared_settings["split"])
	log("Prefetch complete.")


def main() -> None:
	arguments = parse_arguments()
	experiment_config = load_experiment_config(arguments.config)
	shared_settings = experiment_config["shared"]
	fine_tuning_settings = experiment_config["fine_tuning"]

	dataset_entries = parse_dataset_entries(experiment_config, split_comma_list(arguments.datasets))
	model_settings_list = select_models(fine_tuning_settings, split_comma_list(arguments.models))
	experiment_output_dir = Path(shared_settings["output_root"]) / fine_tuning_settings["experiment_name"]
	experiment_output_dir.mkdir(parents=True, exist_ok=True)

	log(f"Experiment 1 (fine-tuning) started {timestamp()}")
	log(f"  datasets : {len(dataset_entries)}")
	log(f"  models   : {[model_settings['name'] for model_settings in model_settings_list]}")
	log(f"  output   : {experiment_output_dir}")
	log()

	if arguments.prefetch_only:
		prefetch_everything(dataset_entries, model_settings_list, shared_settings, experiment_output_dir)
		return

	completed_runs, skipped_runs, failed_runs = [], [], []
	for dataset_position, dataset_entry in enumerate(dataset_entries, start=1):
		dataset_output_dir = experiment_output_dir / dataset_entry.short_name
		pending_models = [
			model_settings for model_settings in model_settings_list
			if not (
				fine_tuning_settings.get("skip_completed_runs", True)
				and (dataset_output_dir / model_settings["name"] / "run_result.json").exists()
			)
		]
		for model_settings in model_settings_list:
			if model_settings not in pending_models:
				skipped_runs.append(f"{dataset_entry.short_name}/{model_settings['name']}")
		if not pending_models:
			log(f"[{dataset_position}/{len(dataset_entries)}] {dataset_entry.name}: all models already done, skipping")
			continue

		log("=" * 70)
		log(f"[{dataset_position}/{len(dataset_entries)}] {dataset_entry.name}  ({timestamp()})")
		try:
			prepared_dataset = prepare_dataset(dataset_entry, shared_settings["split"])
		except Exception:
			print(f"\n[ERROR] loading {dataset_entry.name}", file=sys.stderr, flush=True)
			traceback.print_exc(file=sys.stderr)
			failed_runs.extend(
				f"{dataset_entry.short_name}/{model_settings['name']}" for model_settings in pending_models
			)
			continue

		for model_settings in pending_models:
			run_label = f"{dataset_entry.short_name}/{model_settings['name']}"
			log(f"  -> {model_settings['name']}")
			try:
				run_result = run_single_model(
					prepared_dataset,
					model_settings,
					fine_tuning_settings,
					shared_settings,
					dataset_output_dir / model_settings["name"],
				)
				test_metrics = run_result["test_metrics"]
				log(
					f"     test macro_f1={test_metrics['macro_f1']:.4f}  "
					f"accuracy={test_metrics['accuracy']:.4f}  "
					f"best_epoch={run_result['training']['best_epoch']}"
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
