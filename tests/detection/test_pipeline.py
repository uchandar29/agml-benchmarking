import json

import pytest

from benchmark.config import DetectionConfig, PipelineConfig
from benchmark.pipelines.object_detection import DetectionBenchmarkPipeline
from core.data import DetectionDatasetAdapter
from tests.detection.synthetic import TinyEncoder, make_detection_dataset


def test_phases_1_to_3_write_report_with_scores(tmp_path, monkeypatch):
	pytest.importorskip("torchvision")
	dataset = make_detection_dataset(num_images=30)
	monkeypatch.setattr(DetectionDatasetAdapter, "_load_hf", lambda self: dataset)

	config = PipelineConfig(
		output_dir=str(tmp_path),
		embed_batch_size=8,
		cv_folds=3,
		object_detection=DetectionConfig(n_epochs=1, batch_size=4, image_min_size=96, pretrained=False, crop_max_per_class=100),
	)
	pipeline = DetectionBenchmarkPipeline(
		"fake/synthetic_detection", cfg=config, show_banner=False, encoder=TinyEncoder(),
	)
	report_path = pipeline.run(phases=[1, 2, 3])
	assert report_path.startswith(str(tmp_path / "object_detection" / "synthetic_detection"))

	with open(report_path) as handle:
		report = json.load(handle)
	assert report["task"] == "object_detection"
	assert report["phases_completed"] == [1, 2, 3]
	info = report["dataset_info"]
	assert info["num_images"] == 30 and info["num_classes"] == 2 and info["bbox_format"] == "coco"
	assert sum(info["split_sizes"].values()) == 30
	for name in (
		"box_count_distribution", "box_size_distribution", "aspect_ratio_distribution",
		"instance_class_imbalance", "scale_variance", "crowding", "box_occlusion", "co_occurrence", "exact_duplicate",
		"near_duplicate", "crop_diversity", "label_noise", "dataset_cartography", "missing_annotation", "detector_eval",
	):
		assert name in report["metrics"], name
	scores = report["scores"]
	assert scores["overall"] is not None and 0 <= scores["overall"] <= 10
