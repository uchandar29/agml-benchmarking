"""
Object detection benchmark pipeline (library class, started by benchmark/run.py).

    from benchmark.pipelines.object_detection import DetectionBenchmarkPipeline

    pipeline = DetectionBenchmarkPipeline("Project-AgML/GEMINI_cowpea_pod_detection")
    pipeline.run(phases=[1, 2, 3])

Phases
------
  1  box statistics, instance imbalance, crowding, co-occurrence, exact duplicates   (CPU)
  2  image embeddings (near duplicates), crop embeddings (crop diversity, label noise)  (GPU)
  3  reference detector (cartography on boxes, missing annotation estimate, AP50)    (GPU)

Results go to <output_dir>/object_detection/<dataset>/<timestamp>/.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional

import numpy as np

from benchmark.components.reference_detector import ReferenceDetector
from benchmark.config import PipelineConfig
from benchmark.output.writer import ReportWriter
from benchmark.pipelines.image_classification import _banner
from core.data import DetectionDatasetAdapter, DetectionSchema, split_detection_dataset
from core.data.detection_splits import proxy_labels
from core.embeddings import (
	EncoderSpec,
	ImageEncoder,
	build_encoder,
	download_encoder_weights,
	embed_crops,
	embed_images,
	sample_boxes_per_class,
)
from core.metrics.annotation import LabelNoiseMetric, MissingAnnotationMetric
from core.metrics.detection_scoring import score_detection_report
from core.metrics.difficulty import BoxOcclusionMetric, CrowdingMetric, DatasetCartographyMetric, ScaleVarianceMetric
from core.metrics.diversity import CoOccurrenceMetric, IntraClassDiversityMetric
from core.metrics.structural import (
	AspectRatioDistributionMetric,
	BoxCountDistributionMetric,
	BoxSizeDistributionMetric,
	ExactDuplicateMetric,
	InstanceClassImbalanceMetric,
	NearDuplicateMetric,
)


TASK = "object_detection"


class DetectionReportWriter(ReportWriter):
	"""ReportWriter with the detection report fields."""

	def __init__(self, dataset_name: str, output_dir: str) -> None:
		super().__init__(dataset_name, output_dir, TASK)

	@property
	def report(self) -> dict:
		return self._report

	def set_field(self, key: str, value) -> None:
		self._report[key] = value
		self._flush()

	def set_reproducibility(self, cfg) -> None:
		self._report["reproducibility"] = {
			"split_seed": cfg.split_seed,
			"train_ratio": cfg.train_ratio,
			"val_ratio": cfg.val_ratio,
			"test_ratio": cfg.test_ratio,
			"embed_model": cfg.embed_model,
			"near_dup_threshold": cfg.near_dup_threshold,
			"cv_folds": cfg.cv_folds,
			"object_detection": dataclasses.asdict(cfg.object_detection),
		}
		self._flush()


class DetectionBenchmarkPipeline:
	"""
	Parameters
	----------
	cfg      : PipelineConfig, auto-loaded when None. Its `object_detection` section holds the detector and crop settings.
	encoder  : optional ready ImageEncoder used instead of loading DINOv2 (for tests and offline runs).
	"""

	def __init__(
		self,
		dataset_name: str,
		*,
		config_name: Optional[str] = None,
		image_col: str = "image",
		objects_col: str = "objects",
		bbox_key: str = "bbox",
		category_key: str = "categories",
		bbox_format: str = "auto",
		cfg: Optional[PipelineConfig] = None,
		show_banner: bool = True,
		encoder: Optional[ImageEncoder] = None,
	) -> None:
		self.dataset_name = dataset_name
		self.config_name = config_name
		self.image_col = image_col
		self.objects_col = objects_col
		self.bbox_key = bbox_key
		self.category_key = category_key
		self.bbox_format = bbox_format
		self.cfg = cfg if cfg is not None else PipelineConfig.load()
		self.show_banner = show_banner
		self._encoder = encoder
		self._writer: Optional[DetectionReportWriter] = None

	def _get_writer(self) -> DetectionReportWriter:
		if self._writer is None:
			self._writer = DetectionReportWriter(self.dataset_name, self.cfg.output_dir)
		return self._writer

	def report_path(self) -> str:
		return self._get_writer().path()

	def _adapter(self) -> DetectionDatasetAdapter:
		return DetectionDatasetAdapter(
			dataset_name=self.dataset_name,
			config_name=self.config_name,
			image_col=self.image_col,
			objects_col=self.objects_col,
			bbox_key=self.bbox_key,
			category_key=self.category_key,
			bbox_format=self.bbox_format,
		)

	def _encoder_spec(self) -> EncoderSpec:
		return EncoderSpec(
			name="benchmark", library="transformers_cls",
			model_id=self.cfg.embed_model, batch_size=self.cfg.embed_batch_size,
		)

	def prefetch(self) -> None:
		"""Download the dataset, the detector weights and the encoder into the caches. Nothing else."""
		self._adapter().download()
		if self.cfg.object_detection.pretrained:
			ReferenceDetector.download_weights(self.cfg.object_detection.arch)
		if self._encoder is None:
			download_encoder_weights(self._encoder_spec())

	def run(self, phases: Optional[List[int]] = None) -> str:
		phases = phases or [1]
		writer = self._get_writer()
		writer.set_reproducibility(self.cfg)
		self.cfg.to_json(os.path.join(writer.run_dir, "config_used.json"))
		if self.show_banner:
			_banner(self.dataset_name, phases, writer.path())

		adapter = self._adapter()
		dataset = adapter.load()
		box_table = adapter.build_box_table(dataset)
		schema = adapter.schema()

		splits = split_detection_dataset(
			dataset, box_table,
			train_ratio=self.cfg.train_ratio, val_ratio=self.cfg.val_ratio, seed=self.cfg.split_seed,
		)
		index_sets = splits.index_sets()
		writer.set_field("dataset_info", {
			"num_images": box_table.num_images,
			"num_boxes": box_table.num_boxes,
			"num_classes": box_table.num_classes,
			"label_names": box_table.label_names,
			"bbox_format": schema.bbox_format,
			"dropped_boxes": box_table.dropped_boxes,
			"split_sizes": splits.sizes,
		})
		print(
			f"\nImages={box_table.num_images:,}  boxes={box_table.num_boxes:,}  "
			f"classes={box_table.num_classes}  dropped boxes={box_table.dropped_boxes}"
		)

		if 1 in phases:
			self._phase1(writer, splits, box_table, schema, index_sets)
		if 2 in phases:
			self._phase2(writer, splits, box_table, schema, index_sets)
		if 3 in phases:
			self._phase3(writer, splits, box_table, schema)

		writer.set_field("scores", score_detection_report(writer.report))
		print(f"\n  Scores: {writer.report['scores']}")
		print(f"\n  ✓ Successfully generated report at: {writer.path()}")
		return writer.path()

	# Phases
	def _phase1(self, writer, splits, box_table, schema, index_sets) -> None:
		print("\n── Phase 1: Box Statistics + Structural Quality ──\n")
		steps = [
			("box_count_distribution", lambda: BoxCountDistributionMetric().run(box_table=box_table)),
			("box_size_distribution", lambda: BoxSizeDistributionMetric().run(box_table=box_table)),
			("aspect_ratio_distribution", lambda: AspectRatioDistributionMetric().run(box_table=box_table)),
			("instance_class_imbalance", lambda: InstanceClassImbalanceMetric().run(
				box_table=box_table, train_image_indices=splits.indices["train"])),
			("scale_variance", lambda: ScaleVarianceMetric().run(box_table=box_table)),
			("crowding", lambda: CrowdingMetric().run(box_table=box_table)),
			("box_occlusion", lambda: BoxOcclusionMetric().run(box_table=box_table)),
			("co_occurrence", lambda: CoOccurrenceMetric().run(box_table=box_table)),
			("exact_duplicate", lambda: ExactDuplicateMetric().run(
				full_dataset=splits.full, split_indices=index_sets, schema=schema)),
		]
		for name, step in steps:
			result = step()
			writer.add(name, result)
			print(f"  [{'–' if result.get('skipped') else '✓'}] {name}")
		writer.complete_phase(1)

	def _phase2(self, writer, splits, box_table, schema, index_sets) -> None:
		print("\n── Phase 2: Embedding-Based Metrics ──\n")
		import torch

		detection = self.cfg.object_detection
		encoder = self._encoder
		owns_encoder = encoder is None
		if owns_encoder:
			device = "cuda" if torch.cuda.is_available() else "cpu"
			encoder = build_encoder(self._encoder_spec(), device)
			encoder.load()

		embedding_dir = os.path.join(writer.run_dir, "embeddings")
		os.makedirs(embedding_dir, exist_ok=True)

		image_embeddings = embed_images(encoder, splits.full, schema.image_col, self.cfg.embed_batch_size)
		np.save(os.path.join(embedding_dir, "image_embeddings.npy"), image_embeddings)
		result = NearDuplicateMetric(
			threshold=self.cfg.near_dup_threshold, ivf_threshold=self.cfg.faiss_ivf_threshold,
		).run(
			embeddings=image_embeddings,
			labels=proxy_labels(box_table),
			train_indices=index_sets["train"],
			val_indices=index_sets["val"],
			test_indices=index_sets["test"],
			schema=schema,
			embed_model=self.cfg.embed_model,
		)
		writer.add("near_duplicate", result)
		print(f"  [✓] near_duplicate  rate={result['near_duplicate_rate']}")

		sampled = sample_boxes_per_class(box_table, detection.crop_max_per_class, detection.crop_max_total)
		crop_embeddings, kept_boxes = embed_crops(
			encoder, splits.full, schema.image_col, box_table, sampled,
			batch_size=self.cfg.embed_batch_size, context_pad=detection.crop_context_pad,
		)
		np.save(os.path.join(embedding_dir, "crop_embeddings.npy"), crop_embeddings)
		np.save(os.path.join(embedding_dir, "crop_box_indices.npy"), kept_boxes)
		if owns_encoder:
			encoder.unload()

		crop_labels = box_table.class_id[kept_boxes]
		result = self._crop_diversity(crop_embeddings, crop_labels, box_table, schema)
		writer.add("crop_diversity", result)
		if result.get("skipped"):
			print(f"  [–] crop_diversity  skipped ({result['reason']})")
		else:
			print(f"  [✓] crop_diversity  mean={result['mean_diversity']}")

		result = self._label_noise(crop_embeddings, crop_labels, kept_boxes, box_table, schema)
		writer.add("label_noise", result)
		if result.get("skipped"):
			print(f"  [–] label_noise     skipped ({result['reason']})")
		else:
			print(f"  [✓] label_noise     rate={result['estimated_noise_rate']:.1%}")
		writer.complete_phase(2)

	def _phase3(self, writer, splits, box_table, schema) -> None:
		print("\n── Phase 3: Reference Detector ──\n")
		detection = self.cfg.object_detection
		detector = ReferenceDetector(
			arch=detection.arch, n_epochs=detection.n_epochs, lr=detection.lr,
			batch_size=detection.batch_size, image_min_size=detection.image_min_size,
			max_eval_images=detection.max_eval_images, score_threshold=detection.score_threshold,
			pretrained=detection.pretrained,
		)
		output = detector.run(
			dataset=splits.full, box_table=box_table, image_col=schema.image_col,
			split_indices=splits.indices, run_dir=writer.run_dir,
		)

		result = DatasetCartographyMetric().run(
			conf_history=output.conf_history, orig_idx_order=output.box_order, schema=schema,
		)
		writer.add("dataset_cartography", result)
		print(f"  [✓] dataset_cartography  easy={result['pct_easy']}%  hard={result['pct_hard']}%  (boxes)")

		result = MissingAnnotationMetric(
			score_threshold=detection.missing_score_threshold, iou_threshold=detection.missing_iou,
		).run(predictions=output.predictions, box_table=box_table, image_indices=sorted(output.predictions))
		writer.add("missing_annotation", result)
		print(f"  [✓] missing_annotation   estimate={result['missing_rate_estimate']}  candidates={result['candidate_count']}")

		writer.add("detector_eval", {"ap50": round(output.ap50, 4), "arch": detection.arch, "n_epochs": detection.n_epochs})
		print(f"  [✓] detector_eval        ap50={output.ap50:.3f}")
		writer.complete_phase(3)

	# Crop metrics
	def _classes_with_at_least(self, labels: np.ndarray, minimum: int) -> List[int]:
		counts = np.bincount(labels, minlength=1)
		return [class_id for class_id, count in enumerate(counts) if count >= minimum]

	def _compact_schema(self, schema: DetectionSchema, label_names: List[str]) -> DetectionSchema:
		return dataclasses.replace(schema, label_names=label_names, num_classes=len(label_names))

	def _crop_diversity(self, embeddings, labels, box_table, schema) -> Dict:
		keep = self._classes_with_at_least(labels, 2)
		if not keep:
			return {"skipped": True, "reason": "No class has two or more usable crops."}
		remap = {class_id: position for position, class_id in enumerate(keep)}
		mask = np.isin(labels, keep)
		compact_labels = np.array([remap[int(class_id)] for class_id in labels[mask]], dtype=np.int64)
		compact = self._compact_schema(schema, [box_table.label_names[class_id] for class_id in keep])
		result = IntraClassDiversityMetric().run(
			embeddings=embeddings[mask], labels=compact_labels, schema=compact, embed_model=self.cfg.embed_model,
		)
		result["n_crops"] = int(mask.sum())
		return result

	def _label_noise(self, embeddings, labels, kept_boxes, box_table, schema) -> Dict:
		folds = self.cfg.cv_folds
		keep = self._classes_with_at_least(labels, folds)
		if len(keep) < 2:
			return {
				"skipped": True,
				"reason": f"Fewer than 2 classes have at least {folds} crops (needed for {folds}-fold CV).",
			}
		remap = {class_id: position for position, class_id in enumerate(keep)}
		mask = np.isin(labels, keep)
		compact_labels = np.array([remap[int(class_id)] for class_id in labels[mask]], dtype=np.int64)
		compact = self._compact_schema(schema, [box_table.label_names[class_id] for class_id in keep])
		result = LabelNoiseMetric(cv_folds=folds).run(
			embeddings=embeddings[mask], labels=compact_labels, orig_idx_order=kept_boxes[mask], schema=compact,
		)
		result["flagged_box_indices"] = result.pop("flagged_orig_indices")
		result["classes_dropped"] = [name for class_id, name in enumerate(box_table.label_names) if class_id not in remap and (labels == class_id).any()]
		return result
