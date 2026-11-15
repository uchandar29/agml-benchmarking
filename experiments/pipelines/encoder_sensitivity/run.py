"""
Encoder sensitivity: recompute every embedding-based benchmark metric with several
image encoders to see how much the metrics depend on the encoder.

Labels, class names and splits come from the benchmark run of each dataset
(benchmark/results/<dataset>/<run>/embeddings/labels.npy), so every encoder is scored
on exactly the same images, labels and splits as the benchmark.

Usage (from repo root):
	uv run python -m experiments.pipelines.encoder_sensitivity.run
	uv run python -m experiments.pipelines.encoder_sensitivity.run --encoders dinov2_base,clip_vit_large --datasets bean_disease_uganda
	uv run python -m experiments.pipelines.encoder_sensitivity.run --prefetch-only    # run on the login node first
	uv run python -m experiments.pipelines.encoder_sensitivity.run --smoke-test       # a few images per encoder, nothing saved
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.data import SPLIT_NAMES, DatasetSchema, ImageDataset
from core.embeddings import (
	EncoderSpec,
	ImageEncoder,
	build_encoder,
	download_encoder_weights,
	parse_encoder_specs,
)
from core.metrics.annotation import LabelNoiseMetric
from core.metrics.difficulty import FeatureSeparabilityMetric
from core.metrics.diversity import IntraClassDiversityMetric
from core.metrics.structural import NearDuplicateMetric
from experiments.common import (
	DatasetEntry,
	build_split_manager,
	download_dataset,
	load_dataset_with_schema,
	load_experiment_config,
	parse_dataset_entries,
	resolve_repo_path,
)
from experiments.common.output_files import write_json



def log(message: str = "") -> None:
	print(message, flush=True)


def timestamp() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Encoder sensitivity: benchmark metrics under different encoders")
	parser.add_argument("--config", default=None, help="Path to this experiment's config.yaml (defaults to experiments/slurm/encoder_sensitivity/config.yaml)")
	parser.add_argument("--datasets", default=None, help="Comma separated dataset names to run (full or short names)")
	parser.add_argument("--encoders", default=None, help="Comma separated encoder names from encoder_sensitivity.encoders")
	parser.add_argument("--prefetch-only", action="store_true", help="Download datasets and encoder weights, then exit")
	parser.add_argument("--smoke-test", action="store_true", help="Encode smoke_test_images images with every encoder and exit")
	return parser.parse_args()


def split_comma_list(raw_value: Optional[str]) -> Optional[List[str]]:
	if not raw_value:
		return None
	return [value.strip() for value in raw_value.split(",") if value.strip()]


def select_encoders(experiment_config: Dict[str, Any], selected_names: Optional[List[str]]) -> List[EncoderSpec]:
	specs = parse_encoder_specs(experiment_config["encoder_sensitivity"]["encoders"])
	if not selected_names:
		return specs
	unknown_names = set(selected_names) - {spec.name for spec in specs}
	if unknown_names:
		raise ValueError(f"Encoders not found in config.yaml: {sorted(unknown_names)}")
	return [spec for spec in specs if spec.name in selected_names]


def find_benchmark_run(benchmark_results_root: Path, dataset_short_name: str) -> Optional[Path]:
	"""Latest benchmark run of a dataset that saved its labels."""
	dataset_folder = benchmark_results_root / dataset_short_name
	if not dataset_folder.is_dir():
		return None
	runs = sorted(
		run_folder for run_folder in dataset_folder.iterdir()
		if (run_folder / "embeddings" / "labels.npy").exists()
	)
	return runs[-1] if runs else None


def read_class_names(benchmark_run: Path, number_of_classes: int) -> List[str]:
	"""per_class_silhouette is written in label index order, so it gives the benchmark's class names."""
	report_path = benchmark_run / "report.json"
	if report_path.exists():
		with open(report_path) as report_file:
			metrics = json.load(report_file).get("metrics", {})
		per_class = metrics.get("feature_separability", {}).get("per_class_silhouette") or {}
		if len(per_class) == number_of_classes:
			return list(per_class.keys())
	return [f"class_{label_index}" for label_index in range(number_of_classes)]


class PreparedImages:
	"""Images of one dataset in benchmark row order, with the benchmark's labels and splits."""

	def __init__(self, entry: DatasetEntry, benchmark_results_root: Path, split_settings: Dict[str, Any]) -> None:
		self.entry = entry
		self.full_dataset, adapter_schema = load_dataset_with_schema(entry)
		self.image_column = adapter_schema.image_col
		self.benchmark_run = find_benchmark_run(benchmark_results_root, entry.short_name)

		if self.benchmark_run is not None:
			self.labels = np.load(self.benchmark_run / "embeddings" / "labels.npy").astype(np.int64)
			self.label_source = str(self.benchmark_run)
			if len(self.labels) != len(self.full_dataset):
				raise ValueError(
					f"{entry.name}: benchmark run has {len(self.labels):,} labels but the loaded dataset has "
					f"{len(self.full_dataset):,} rows. Check hf_config_name in the dataset list."
				)
			number_of_classes = int(self.labels.max()) + 1
			class_names = read_class_names(self.benchmark_run, number_of_classes)
		else:
			# No benchmark run with labels (it never finished Phase 2), so use the adapter's labels
			self.labels = np.array(self.full_dataset[adapter_schema.label_col], dtype=np.int64)
			self.label_source = "dataset_adapter"
			class_names = list(adapter_schema.label_names)

		self.schema = DatasetSchema(
			image_col=self.image_column,
			label_col=adapter_schema.label_col,
			label_names=class_names,
			num_classes=len(class_names),
			metadata_cols=[],
		)
		# Labels can come from the benchmark run rather than the dataset, so only the indices are needed here
		split_indices = build_split_manager(split_settings).compute_indices(self.labels)
		self.split_indices = {name: set(indices.tolist()) for name, indices in zip(SPLIT_NAMES, split_indices)}

	def benchmark_embeddings(self) -> Optional[np.ndarray]:
		if self.benchmark_run is None:
			return None
		embedding_path = self.benchmark_run / "embeddings" / "embeddings.npy"
		return np.load(embedding_path) if embedding_path.exists() else None


def embed_images(encoder: ImageEncoder, prepared: PreparedImages, workers: int, row_limit: Optional[int] = None) -> np.ndarray:
	from torch.utils.data import DataLoader

	image_dataset = ImageDataset(prepared.full_dataset, prepared.image_column, encoder.preprocess, row_limit=row_limit)
	loader = DataLoader(image_dataset, batch_size=encoder.spec.batch_size, shuffle=False, num_workers=workers)
	batches = [encoder.encode(pixel_batch) for pixel_batch in loader]
	return np.concatenate(batches, axis=0)


def compute_embedding_metrics(
	embeddings: np.ndarray,
	prepared: PreparedImages,
	settings: Dict[str, Any],
	encoder_name: str,
) -> Dict[str, Any]:
	metrics: Dict[str, Any] = {"near_duplicate": {}}
	for threshold in settings["near_duplicate_thresholds"]:
		metrics["near_duplicate"][str(threshold)] = NearDuplicateMetric(
			threshold=threshold,
			ivf_threshold=settings["faiss_ivf_threshold"],
		).run(
			embeddings=embeddings,
			labels=prepared.labels,
			train_indices=prepared.split_indices["train"],
			val_indices=prepared.split_indices["val"],
			test_indices=prepared.split_indices["test"],
			schema=prepared.schema,
			embed_model=encoder_name,
		)
	metrics["feature_separability"] = FeatureSeparabilityMetric().run(
		embeddings=embeddings,
		labels=prepared.labels,
		schema=prepared.schema,
		embed_model=encoder_name,
		max_silhouette_samples=settings["max_silhouette_samples"],
	)
	metrics["intra_class_diversity"] = IntraClassDiversityMetric().run(
		embeddings=embeddings,
		labels=prepared.labels,
		schema=prepared.schema,
		embed_model=encoder_name,
	)
	if prepared.schema.num_classes >= 2:
		metrics["label_noise"] = LabelNoiseMetric(cv_folds=settings["label_noise_cv_folds"]).run(
			embeddings=embeddings,
			labels=prepared.labels,
			orig_idx_order=np.arange(len(embeddings), dtype=np.int64),
			schema=prepared.schema,
		)
	return metrics


def row_alignment_check(new_embeddings: np.ndarray, benchmark_embeddings: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
	"""For the benchmark encoder, row-wise cosine with the saved benchmark embeddings should be close to 1."""
	if benchmark_embeddings is None or benchmark_embeddings.shape != new_embeddings.shape:
		return None
	cosine = np.sum(new_embeddings * benchmark_embeddings, axis=1)
	return {"mean_cosine": round(float(cosine.mean()), 6), "min_cosine": round(float(cosine.min()), 6)}


def run_one(
	encoder: ImageEncoder,
	prepared: PreparedImages,
	settings: Dict[str, Any],
	output_folder: Path,
	workers: int,
) -> Dict[str, Any]:
	started = time.time()
	embeddings = embed_images(encoder, prepared, workers)
	embedding_seconds = time.time() - started
	metrics = compute_embedding_metrics(embeddings, prepared, settings, encoder.spec.name)

	result = {
		"dataset": prepared.entry.name,
		"encoder": encoder.spec.name,
		"library": encoder.spec.library,
		"model_id": encoder.spec.model_id,
		"pretrained": encoder.spec.pretrained,
		"embedding_dimension": int(embeddings.shape[1]),
		"number_of_images": int(embeddings.shape[0]),
		"number_of_classes": prepared.schema.num_classes,
		"label_source": prepared.label_source,
		"embedding_seconds": round(embedding_seconds, 1),
		"total_seconds": round(time.time() - started, 1),
		"finished_at": timestamp(),
		"metrics": metrics,
	}
	if encoder.spec.name == settings["baseline_encoder"]:
		result["benchmark_row_check"] = row_alignment_check(embeddings, prepared.benchmark_embeddings())

	output_folder.mkdir(parents=True, exist_ok=True)
	if settings.get("save_embeddings", True):
		np.save(output_folder / "embeddings.npy", embeddings)
	write_json(output_folder / "metrics.json", result)
	return result


def prefetch(dataset_entries: List[DatasetEntry], encoder_specs: List[EncoderSpec]) -> None:
	for entry in dataset_entries:
		log(f"Downloading dataset {entry.name}")
		download_dataset(entry)
	for spec in encoder_specs:
		log(f"Downloading weights for {spec.name} ({spec.model_id})")
		download_encoder_weights(spec)


def smoke_test(dataset_entries: List[DatasetEntry], encoder_specs: List[EncoderSpec], benchmark_results_root: Path,
		split_settings: Dict[str, Any], device: str, image_count: int) -> bool:
	prepared = PreparedImages(dataset_entries[0], benchmark_results_root, split_settings)
	log(f"Smoke test on {prepared.entry.name} ({image_count} images, labels from {prepared.label_source})")
	all_passed = True
	for spec in encoder_specs:
		encoder = build_encoder(spec, device)
		try:
			encoder.load()
			embeddings = embed_images(encoder, prepared, workers=0, row_limit=image_count)
			norms = np.linalg.norm(embeddings, axis=1)
			log(f"  [ok]   {spec.name:16s} shape={embeddings.shape}  norm={norms.min():.3f}-{norms.max():.3f}")
		except Exception as error:
			all_passed = False
			log(f"  [fail] {spec.name:16s} {type(error).__name__}: {error}")
		finally:
			encoder.unload()
	return all_passed


def main() -> None:
	arguments = parse_arguments()
	experiment_config = load_experiment_config("encoder_sensitivity", arguments.config)
	settings = experiment_config["encoder_sensitivity"]
	split_settings = experiment_config["shared"]["split"]
	workers = int(experiment_config["shared"].get("dataloader_workers", 4))
	benchmark_results_root = resolve_repo_path(settings["benchmark_results_root"])
	output_root = Path(experiment_config["shared"]["output_root"]) / settings["experiment_name"]

	dataset_entries = parse_dataset_entries(experiment_config, split_comma_list(arguments.datasets))
	encoder_specs = select_encoders(experiment_config, split_comma_list(arguments.encoders))

	if arguments.prefetch_only:
		prefetch(dataset_entries, encoder_specs)
		return

	import torch

	device = "cuda" if torch.cuda.is_available() else "cpu"
	if arguments.smoke_test:
		sys.exit(0 if smoke_test(dataset_entries, encoder_specs, benchmark_results_root, split_settings, device, int(settings["smoke_test_images"])) else 1)

	log(f"[{timestamp()}] {len(encoder_specs)} encoders x {len(dataset_entries)} datasets on {device}")
	log(f"Results: {output_root}")
	failures = []

	# Encoders on the outside so each model is loaded once
	for spec in encoder_specs:
		pending_entries = [
			entry for entry in dataset_entries
			if not (settings.get("skip_completed_runs", True) and (output_root / entry.short_name / spec.name / "metrics.json").exists())
		]
		log(f"\n[{timestamp()}] Encoder {spec.name} ({spec.model_id}): {len(pending_entries)} datasets to run")
		if not pending_entries:
			continue
		encoder = build_encoder(spec, device)
		try:
			encoder.load()
		except Exception as error:
			failures.append((spec.name, "load", f"{type(error).__name__}: {error}"))
			log(f"  Could not load {spec.name}: {error}")
			continue

		for entry in pending_entries:
			try:
				prepared = PreparedImages(entry, benchmark_results_root, split_settings)
				result = run_one(encoder, prepared, settings, output_root / entry.short_name / spec.name, workers)
				separability = result["metrics"]["feature_separability"]
				noise = result["metrics"].get("label_noise", {})
				log(
					f"  [ok] {entry.short_name:50s} silhouette={separability.get('silhouette_score')}  "
					f"noise={noise.get('estimated_noise_rate')}  ({result['total_seconds']}s)"
				)
			except Exception as error:
				failures.append((spec.name, entry.name, f"{type(error).__name__}: {error}"))
				log(f"  [fail] {entry.short_name}: {type(error).__name__}: {error}")
				traceback.print_exc()
		encoder.unload()

	log(f"\n[{timestamp()}] Finished with {len(failures)} failures")
	for encoder_name, dataset_name, message in failures:
		log(f"  {encoder_name:16s} {dataset_name}: {message}")


if __name__ == "__main__":
	main()
