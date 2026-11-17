"""
ReferenceDetector
=================
Fine-tunes a torchvision Faster R-CNN (COCO weights) on the training split and records,
after every epoch, how confidently it finds each ground-truth box of the training images.
That confidence history feeds DatasetCartographyMetric. After training it predicts on
the validation and test images (class-agnostic boxes and scores) for MissingAnnotationMetric,
and computes AP50 on the test split.

Confidence of a ground-truth box = highest score among predictions of the same class
with IoU >= 0.5, or 0 when there is none.

Output (written to run_dir/phase3_detection/)
---------------------------------------------
  detector.pt            -- final state dict
  conf_history.npy       -- float32 (M_eval, n_epochs)
  box_order.npy          -- int64 (M_eval,), BoxTable rows of the evaluated boxes
  predictions.npz        -- class-agnostic predictions on val + test images
  summary.json           -- ap50 and the evaluated image numbers
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from core.data import BoxTable, to_rgb

IOU_MATCH = 0.5
NMS_IOU = 0.5


@dataclass
class DetectorOutput:
	conf_history: np.ndarray                      # float32 (M_eval, n_epochs)
	box_order: np.ndarray                         # int64 (M_eval,)
	predictions: Dict[int, Dict[str, np.ndarray]]  # image index -> {"boxes", "scores"}
	ap50: float


def compute_ap50(
	predictions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
	ground_truth: List[Tuple[np.ndarray, np.ndarray]],
	num_classes: int,
) -> float:
	"""
	Mean over classes of AP at IoU 0.5 (all-point interpolation, VOC style).
	predictions[i] = (boxes, labels, scores) and ground_truth[i] = (boxes, labels) for image i.
	Classes without ground truth are left out of the mean.
	"""
	import torch
	from torchvision.ops import box_iou

	average_precisions = []
	for class_id in range(num_classes):
		gt_boxes = {}
		for image_number, (boxes, labels) in enumerate(ground_truth):
			class_boxes = boxes[labels == class_id]
			if len(class_boxes):
				gt_boxes[image_number] = class_boxes
		total_gt = sum(len(boxes) for boxes in gt_boxes.values())
		if total_gt == 0:
			continue

		detections = []
		for image_number, (boxes, labels, scores) in enumerate(predictions):
			for box, score in zip(boxes[labels == class_id], scores[labels == class_id]):
				detections.append((float(score), image_number, box))
		detections.sort(key=lambda item: -item[0])

		used = {image_number: np.zeros(len(boxes), dtype=bool) for image_number, boxes in gt_boxes.items()}
		true_positive = np.zeros(len(detections))
		for rank, (_, image_number, box) in enumerate(detections):
			if image_number not in gt_boxes:
				continue
			ious = box_iou(torch.as_tensor(box[None]), torch.as_tensor(gt_boxes[image_number]))[0].numpy()
			best = int(ious.argmax())
			if ious[best] >= IOU_MATCH and not used[image_number][best]:
				used[image_number][best] = True
				true_positive[rank] = 1

		cumulative_tp = np.cumsum(true_positive)
		recall = cumulative_tp / total_gt
		precision = cumulative_tp / np.maximum(np.arange(1, len(detections) + 1), 1)
		recall = np.concatenate([[0.0], recall, [1.0]])
		precision = np.concatenate([[0.0], precision, [0.0]])
		precision = np.maximum.accumulate(precision[::-1])[::-1]
		changed = np.flatnonzero(recall[1:] != recall[:-1])
		average_precisions.append(float(((recall[changed + 1] - recall[changed]) * precision[changed + 1]).sum()))

	return float(np.mean(average_precisions)) if average_precisions else 0.0


class ReferenceDetector:
	"""
	Usage::

		detector = ReferenceDetector(n_epochs=12)
		output = detector.run(
			dataset=splits.full, box_table=box_table, image_col="image",
			split_indices=splits.indices, run_dir=writer.run_dir,
		)
	"""

	def __init__(
		self,
		arch: str = "fasterrcnn_mobilenet_v3_large_fpn",
		n_epochs: int = 12,
		lr: float = 0.01,
		batch_size: int = 8,
		image_min_size: int = 640,
		max_eval_images: int = 2000,
		score_threshold: float = 0.05,
		device: Optional[str] = None,
		pretrained: bool = True,
	) -> None:
		self.arch = arch
		self.n_epochs = n_epochs
		self.lr = lr
		self.batch_size = batch_size
		self.image_min_size = image_min_size
		self.max_eval_images = max_eval_images
		self.score_threshold = score_threshold
		self.pretrained = pretrained
		self._device_override = device

	@staticmethod
	def download_weights(arch: str) -> None:
		"""Fetch the COCO weights into the torch cache (for the login node)."""
		import torchvision.models.detection as detection

		getattr(detection, arch)(weights="DEFAULT")

	def run(
		self,
		dataset: Dataset,
		box_table: BoxTable,
		image_col: str,
		split_indices: Dict[str, np.ndarray],
		run_dir: str,
		reuse: bool = True,
	) -> DetectorOutput:
		"""Train (or restore from cache) and evaluate. split_indices are row numbers in the full dataset."""
		folder = os.path.join(run_dir, "phase3_detection")
		paths = {
			"model": os.path.join(folder, "detector.pt"),
			"conf": os.path.join(folder, "conf_history.npy"),
			"order": os.path.join(folder, "box_order.npy"),
			"predictions": os.path.join(folder, "predictions.npz"),
			"summary": os.path.join(folder, "summary.json"),
		}
		settings = {
			"arch": self.arch,
			"n_epochs": self.n_epochs,
			"num_classes": box_table.num_classes,
			"max_eval_images": self.max_eval_images,
			"lr": self.lr,
			"batch_size": self.batch_size,
			"image_min_size": self.image_min_size,
			"score_threshold": self.score_threshold,
		}
		if reuse and all(os.path.exists(path) for path in paths.values()):
			with open(paths["summary"]) as handle:
				cached = json.load(handle)
			if all(cached.get(key) == value for key, value in settings.items()):
				print(f"  Reusing cached detector outputs from {folder}")
				return self._load_output(paths)
			print("  Cached detector settings differ, retraining.")

		os.makedirs(folder, exist_ok=True)
		output, model = self._train_and_evaluate(dataset, box_table, image_col, split_indices)

		import torch

		torch.save(model.state_dict(), paths["model"])
		np.save(paths["conf"], output.conf_history)
		np.save(paths["order"], output.box_order)
		self._save_predictions(paths["predictions"], output.predictions)
		with open(paths["summary"], "w") as handle:
			json.dump({"ap50": output.ap50, **settings}, handle)
		print(f"  Detector outputs saved → {folder}")
		return output

	# Model and data
	def _build_model(self, num_classes: int):
		import torchvision.models.detection as detection
		from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

		builder = getattr(detection, self.arch, None)
		if builder is None:
			raise ValueError(f"Unknown detection architecture '{self.arch}'")
		options = dict(min_size=self.image_min_size, box_score_thresh=self.score_threshold, box_detections_per_img=300)
		if self.pretrained:
			model = builder(weights="DEFAULT", **options)
			in_features = model.roi_heads.box_predictor.cls_score.in_features
			model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
		else:
			model = builder(weights=None, weights_backbone=None, num_classes=num_classes + 1, **options)
		return model

	def _load_image(self, dataset: Dataset, image_col: str, image_index: int):
		from torchvision.transforms.functional import to_tensor

		return to_tensor(to_rgb(dataset[int(image_index)][image_col]))

	def _target(self, box_table: BoxTable, image_index: int) -> Dict[str, "object"]:
		import torch

		rows = box_table.subset([image_index])
		return {
			"boxes": torch.as_tensor(box_table.boxes_xyxy[rows], dtype=torch.float32).reshape(-1, 4),
			"labels": torch.as_tensor(box_table.class_id[rows] + 1, dtype=torch.int64),  # 0 is background
		}

	def _predict(self, model, dataset: Dataset, image_col: str, image_indices: np.ndarray, device: str, desc: str):
		"""Yields (boxes, labels, scores) as numpy arrays for each image. Labels are class ids without the background shift."""
		import torch

		model.eval()
		for start in tqdm(range(0, len(image_indices), self.batch_size), desc=desc, leave=False):
			chunk = image_indices[start : start + self.batch_size]
			images = [self._load_image(dataset, image_col, image_index).to(device) for image_index in chunk]
			with torch.no_grad():
				outputs = model(images)
			for output in outputs:
				yield (
					output["boxes"].cpu().numpy().astype(np.float32),
					output["labels"].cpu().numpy().astype(np.int64) - 1,
					output["scores"].cpu().numpy().astype(np.float32),
				)

	def _train_and_evaluate(self, dataset, box_table: BoxTable, image_col: str, split_indices):
		import torch
		from torchvision.ops import box_iou, nms

		device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
		print(f"  Training {self.arch} on {device} for {self.n_epochs} epochs …")

		has_boxes = box_table.boxes_per_image() > 0
		train_images = np.asarray(split_indices["train"], dtype=np.int64)
		train_images = train_images[has_boxes[train_images]]
		if len(train_images) == 0:
			raise ValueError("No training image has a box.")

		# Fixed sample of training images whose boxes are tracked every epoch
		rng = np.random.default_rng(42)
		eval_images = train_images
		if len(eval_images) > self.max_eval_images:
			eval_images = np.sort(rng.choice(train_images, size=self.max_eval_images, replace=False))
		box_order = box_table.subset(eval_images)
		row_of_box = {int(box_row): position for position, box_row in enumerate(box_order)}
		conf_history = np.zeros((len(box_order), self.n_epochs), dtype=np.float32)

		model = self._build_model(box_table.num_classes).to(device)
		parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
		optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=0.9, weight_decay=5e-4)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
		iterations_per_epoch = -(-len(train_images) // self.batch_size)
		warmup_iterations = min(200, iterations_per_epoch)
		iteration = 0

		for epoch in range(self.n_epochs):
			model.train()
			order = rng.permutation(train_images)
			for start in tqdm(range(0, len(order), self.batch_size), desc=f"Epoch {epoch + 1}/{self.n_epochs}", leave=False):
				images, targets = [], []
				for image_index in order[start : start + self.batch_size]:
					image = self._load_image(dataset, image_col, image_index)
					target = self._target(box_table, int(image_index))
					if rng.random() < 0.5:  # horizontal flip
						image = image.flip(-1)
						boxes = target["boxes"].clone()
						boxes[:, [0, 2]] = image.shape[-1] - target["boxes"][:, [2, 0]]
						target["boxes"] = boxes
					images.append(image.to(device))
					targets.append({key: value.to(device) for key, value in target.items()})
				losses = model(images, targets)
				loss = sum(losses.values())
				if not torch.isfinite(loss):
					raise RuntimeError(
						f"Detector loss became {loss.item()} in epoch {epoch + 1}. Try a lower lr in the detection config."
					)
				if iteration < warmup_iterations:  # linear warmup
					for group in optimizer.param_groups:
						group["lr"] = self.lr * (iteration + 1) / warmup_iterations
				iteration += 1
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()

			for image_index, (boxes, labels, scores) in zip(
				eval_images, self._predict(model, dataset, image_col, eval_images, device, "Tracking train boxes")
			):
				rows = box_table.subset([image_index])
				ious = box_iou(torch.as_tensor(box_table.boxes_xyxy[rows]), torch.as_tensor(boxes)).numpy() if len(boxes) else None
				for local, box_row in enumerate(rows):
					if ious is None:
						continue
					match = (ious[local] >= IOU_MATCH) & (labels == box_table.class_id[box_row])
					if match.any():
						conf_history[row_of_box[int(box_row)], epoch] = scores[match].max()
			print(f"  epoch {epoch + 1}/{self.n_epochs}  mean tracked confidence={conf_history[:, epoch].mean():.3f}")

		# Predictions on val + test, and AP50 on test
		held_out = np.sort(np.concatenate([split_indices["val"], split_indices["test"]]).astype(np.int64))
		test_set = set(int(index) for index in split_indices["test"])
		predictions: Dict[int, Dict[str, np.ndarray]] = {}
		test_predictions, test_truth = [], []
		for image_index, (boxes, labels, scores) in zip(
			held_out, self._predict(model, dataset, image_col, held_out, device, "Predicting val + test")
		):
			keep = nms(torch.as_tensor(boxes), torch.as_tensor(scores), NMS_IOU).numpy() if len(boxes) else np.zeros(0, dtype=np.int64)
			predictions[int(image_index)] = {"boxes": boxes[keep], "scores": scores[keep]}
			if int(image_index) in test_set:
				rows = box_table.subset([image_index])
				test_predictions.append((boxes, labels, scores))
				test_truth.append((box_table.boxes_xyxy[rows], box_table.class_id[rows]))

		ap50 = compute_ap50(test_predictions, test_truth, box_table.num_classes)
		print(f"  Detector AP50 on test = {ap50:.3f}")

		model.to("cpu")
		return DetectorOutput(conf_history=conf_history, box_order=box_order.astype(np.int64), predictions=predictions, ap50=ap50), model

	# Cache helpers
	@staticmethod
	def _save_predictions(path: str, predictions: Dict[int, Dict[str, np.ndarray]]) -> None:
		image_numbers = np.array(sorted(predictions), dtype=np.int64)
		counts = np.array([len(predictions[int(number)]["scores"]) for number in image_numbers], dtype=np.int64)
		boxes = [predictions[int(number)]["boxes"] for number in image_numbers]
		scores = [predictions[int(number)]["scores"] for number in image_numbers]
		np.savez(
			path,
			image_numbers=image_numbers,
			counts=counts,
			boxes=np.concatenate(boxes).reshape(-1, 4) if boxes else np.zeros((0, 4), dtype=np.float32),
			scores=np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32),
		)

	@staticmethod
	def _load_output(paths: Dict[str, str]) -> DetectorOutput:
		saved = np.load(paths["predictions"])
		edges = np.concatenate([[0], np.cumsum(saved["counts"])])
		predictions = {
			int(number): {"boxes": saved["boxes"][edges[position]:edges[position + 1]], "scores": saved["scores"][edges[position]:edges[position + 1]]}
			for position, number in enumerate(saved["image_numbers"])
		}
		with open(paths["summary"]) as handle:
			ap50 = json.load(handle)["ap50"]
		return DetectorOutput(np.load(paths["conf"]), np.load(paths["order"]), predictions, ap50)
