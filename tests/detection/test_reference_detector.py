import numpy as np
import pytest
import torch

from benchmark.components import reference_detector as detector_module
from benchmark.components.reference_detector import ReferenceDetector, learning_rate_factor
from core.data import DetectionDatasetAdapter
from tests.detection.synthetic import make_box_table, make_detection_dataset


def test_learning_rate_warms_up_then_decays():
	factors = [learning_rate_factor(step, 10, 100) for step in range(101)]
	assert factors[0] == pytest.approx(detector_module.WARMUP_START_FACTOR)
	assert all(earlier < later for earlier, later in zip(factors[:10], factors[1:11]))
	assert factors[10] == pytest.approx(1.0)
	assert all(earlier >= later for earlier, later in zip(factors[10:], factors[11:]))
	assert factors[100] == pytest.approx(0.0, abs=1e-9)


def test_sub_pixel_boxes_are_left_out_of_training_targets():
	table = make_box_table([(100, 100)], [[[10, 10, 40, 40], [50, 50, 50.4, 70]]], [[0, 1]], ["a", "b"])
	target = ReferenceDetector(pretrained=False)._target(table, 0)
	assert target["boxes"].tolist() == [[10.0, 10.0, 40.0, 40.0]]
	assert target["labels"].tolist() == [1]


def _tiny_setup(monkeypatch):
	dataset = make_detection_dataset(num_images=12)
	monkeypatch.setattr(DetectionDatasetAdapter, "_load_hf", lambda self: dataset)
	adapter = DetectionDatasetAdapter("fake/coco")
	table = adapter.build_box_table(adapter.load())
	split_indices = {"train": np.arange(8), "val": np.arange(8, 10), "test": np.arange(10, 12)}
	return dataset, table, split_indices


def _inject_nan_losses(monkeypatch, nan_calls):
	"""Training forward calls listed in nan_calls return a nan loss."""
	original_build = ReferenceDetector._build_model
	calls = {"train": 0}

	def build_with_nan(self, num_classes):
		model = original_build(self, num_classes)
		original_forward = model.forward

		def forward(images, targets=None):
			if targets is None:
				return original_forward(images)
			calls["train"] += 1
			losses = original_forward(images, targets)
			if calls["train"] in nan_calls:
				losses["loss_classifier"] = losses["loss_classifier"] * float("nan")
			return losses

		model.forward = forward
		return model

	monkeypatch.setattr(ReferenceDetector, "_build_model", build_with_nan)


def test_one_nan_batch_is_skipped(monkeypatch, tmp_path):
	dataset, table, split_indices = _tiny_setup(monkeypatch)
	_inject_nan_losses(monkeypatch, {1})
	detector = ReferenceDetector(n_epochs=1, batch_size=1, image_min_size=96, pretrained=False, device="cpu")
	output = detector.run(dataset, table, "image", split_indices, str(tmp_path), reuse=False)
	assert np.isfinite(output.conf_history).all()
	model_weights = torch.load(tmp_path / "phase3_detection" / "detector.pt")
	assert all(torch.isfinite(weights).all() for weights in model_weights.values() if weights.is_floating_point())


def test_many_nan_batches_raise(monkeypatch, tmp_path):
	dataset, table, split_indices = _tiny_setup(monkeypatch)
	_inject_nan_losses(monkeypatch, {1, 2, 3})
	detector = ReferenceDetector(n_epochs=1, batch_size=1, image_min_size=96, pretrained=False, device="cpu")
	with pytest.raises(RuntimeError, match="non-finite"):
		detector.run(dataset, table, "image", split_indices, str(tmp_path), reuse=False)
