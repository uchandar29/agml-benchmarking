"""
DetectionDatasetAdapter
=======================
Loads an object detection dataset from Hugging Face and turns its `objects`
column into a BoxTable with absolute xyxy pixel boxes.

Formats seen in the Project-AgML datasets
-----------------------------------------
  objects = {bbox: [[...], ...], categories: [int, ...] or one scalar for all boxes}
  bbox as absolute COCO xywh (top-left), normalised COCO xywh (top-left, coco_normalized),
  normalised YOLO xywh (centre), or absolute xyxy (voc)

bbox_format "auto": any coordinate above 1.0 means coco. Otherwise the boxes are normalised, and
yolo or coco_normalized is picked by which reading keeps more boxes inside the image.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import datasets
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from .box_table import BoxTable
from .dataset_adapter import DatasetAdapter

BBOX_FORMATS = ("auto", "coco", "coco_normalized", "yolo", "voc")


@dataclass
class DetectionSchema:
	"""Same attribute names as DatasetSchema where the shared metrics read them."""
	image_col: str
	objects_col: str
	label_names: List[str]
	num_classes: int
	bbox_format: str


def detect_bbox_format(box_arrays: List[np.ndarray]) -> str:
	"""coco when any coordinate is above 1.0, else yolo or coco_normalized, whichever leaves fewer boxes outside the image."""
	non_empty = [np.asarray(boxes, dtype=np.float64).reshape(-1, 4) for boxes in box_arrays if len(boxes)]
	largest = max((float(boxes.max()) for boxes in non_empty), default=0.0)
	if 1.0 + 1e-3 < largest <= 1.5:
		print(f"  ⚠  Largest box coordinate is {largest:.3f}: treating boxes as coco pixels. Pass bbox_format if they are normalised.")
	if largest > 1.0 + 1e-3:
		return "coco"
	if not non_empty:
		return "yolo"
	boxes = np.concatenate(non_empty)
	slack = 0.01
	# Centre reading: the box must not cross the image edge around its centre
	outside_as_centre = ((boxes[:, 0] - boxes[:, 2] / 2 < -slack) | (boxes[:, 0] + boxes[:, 2] / 2 > 1 + slack)
		| (boxes[:, 1] - boxes[:, 3] / 2 < -slack) | (boxes[:, 1] + boxes[:, 3] / 2 > 1 + slack)).sum()
	outside_as_top_left = ((boxes[:, 0] + boxes[:, 2] > 1 + slack) | (boxes[:, 1] + boxes[:, 3] > 1 + slack)).sum()
	return "coco_normalized" if outside_as_top_left < outside_as_centre else "yolo"


def boxes_to_xyxy(boxes: np.ndarray, bbox_format: str, width: int, height: int) -> np.ndarray:
	"""Convert [k, 4] boxes to absolute xyxy pixels, clipped to the image. Degenerate boxes are kept here."""
	boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
	if bbox_format == "yolo":
		center_x, center_y = boxes[:, 0] * width, boxes[:, 1] * height
		half_width, half_height = boxes[:, 2] * width / 2.0, boxes[:, 3] * height / 2.0
		xyxy = np.stack([center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height], axis=1)
	elif bbox_format == "coco":
		xyxy = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], axis=1)
	elif bbox_format == "coco_normalized":
		left, top = boxes[:, 0] * width, boxes[:, 1] * height
		xyxy = np.stack([left, top, left + boxes[:, 2] * width, top + boxes[:, 3] * height], axis=1)
	elif bbox_format == "voc":
		xyxy = boxes.copy()
	else:
		raise ValueError(f"Unknown bbox_format '{bbox_format}'. Choose from {BBOX_FORMATS}")
	xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
	xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
	return xyxy.astype(np.float32)


def broadcast_categories(categories, count: int) -> np.ndarray:
	"""Categories are a list per box, or one scalar that applies to every box."""
	if np.isscalar(categories):
		return np.full(count, int(round(float(categories))), dtype=np.int64)
	values = np.asarray(categories, dtype=np.float64).reshape(-1)
	if len(values) != count:
		raise ValueError(f"Got {len(values)} categories for {count} boxes")
	return np.rint(values).astype(np.int64)


def _unwrap_feature(feature):
	"""Strip Sequence / List wrappers until the inner feature is reached."""
	while True:
		if isinstance(feature, list) and feature:
			feature = feature[0]
		elif hasattr(feature, "feature"):
			feature = feature.feature
		else:
			return feature


class DetectionDatasetAdapter(DatasetAdapter):
	"""
	Usage::

		adapter = DetectionDatasetAdapter("Project-AgML/GEMINI_cowpea_pod_detection")
		dataset = adapter.load()
		box_table = adapter.build_box_table(dataset)
		schema = adapter.schema()   # available after build_box_table
	"""

	def __init__(
		self,
		dataset_name: str,
		config_name: Optional[str] = None,
		image_col: str = "image",
		objects_col: str = "objects",
		bbox_key: str = "bbox",
		category_key: str = "categories",
		bbox_format: str = "auto",
	) -> None:
		if bbox_format not in BBOX_FORMATS:
			raise ValueError(f"bbox_format must be one of {BBOX_FORMATS}, got '{bbox_format}'")
		super().__init__(dataset_name, config_name=config_name, image_col=image_col)
		self.objects_col = objects_col
		self.bbox_key = bbox_key
		self.category_key = category_key
		self.bbox_format = bbox_format
		self._detection_schema: Optional[DetectionSchema] = None

	def download(self) -> None:
		"""Fetch the raw dataset into the local cache. Nothing else."""
		self._load_hf()

	def load(self) -> Dataset:
		if self._dataset is not None:
			return self._dataset
		dataset = self._load_hf()
		for column in (self._override_image_col, self.objects_col):
			if column not in dataset.features:
				raise ValueError(
					f"Column '{column}' not in '{self.dataset_name}'. Available: {list(dataset.features.keys())}"
				)
		self._dataset = dataset
		print(f"  Loaded {len(dataset):,} examples.")
		return dataset

	def schema(self) -> DetectionSchema:
		if self._detection_schema is None:
			raise RuntimeError("DetectionDatasetAdapter.schema() called before build_box_table().")
		return self._detection_schema

	def build_box_table(self, dataset: Dataset) -> BoxTable:
		image_col = self._override_image_col
		image_sizes = self._read_image_sizes(dataset, image_col)
		raw_objects = list(dataset[self.objects_col])

		raw_boxes: List[np.ndarray] = []
		raw_categories: List[np.ndarray] = []
		for entry in raw_objects:
			boxes, categories = self._parse_entry(entry)
			raw_boxes.append(boxes)
			raw_categories.append(categories)

		bbox_format = detect_bbox_format(raw_boxes) if self.bbox_format == "auto" else self.bbox_format
		print(f"  bbox format: {bbox_format}" + (" (auto-detected)" if self.bbox_format == "auto" else ""))

		image_index_parts, class_parts, box_parts = [], [], []
		dropped = 0
		for image_number, (boxes, categories) in enumerate(zip(raw_boxes, raw_categories)):
			if len(boxes) == 0:
				continue
			width, height = image_sizes[image_number]
			xyxy = boxes_to_xyxy(boxes, bbox_format, int(width), int(height))
			valid = ((xyxy[:, 2] - xyxy[:, 0]) > 0) & ((xyxy[:, 3] - xyxy[:, 1]) > 0)
			dropped += int((~valid).sum())
			image_index_parts.append(np.full(int(valid.sum()), image_number, dtype=np.int64))
			class_parts.append(categories[valid])
			box_parts.append(xyxy[valid])

		if not box_parts or sum(len(part) for part in box_parts) == 0:
			raise ValueError(f"No usable boxes found in '{self.dataset_name}' (dropped {dropped})")

		class_id = np.concatenate(class_parts)
		label_names = self._label_names(dataset, int(class_id.max()) + 1)
		if int(class_id.max()) >= len(label_names):
			raise ValueError("Found a class id larger than the number of label names")

		self._detection_schema = DetectionSchema(
			image_col=image_col,
			objects_col=self.objects_col,
			label_names=label_names,
			num_classes=len(label_names),
			bbox_format=bbox_format,
		)
		return BoxTable(
			image_index=np.concatenate(image_index_parts),
			class_id=class_id,
			boxes_xyxy=np.concatenate(box_parts).astype(np.float32),
			image_sizes=image_sizes,
			label_names=label_names,
			dropped_boxes=dropped,
		)

	# Internal helpers
	def _parse_entry(self, entry) -> Tuple[np.ndarray, np.ndarray]:
		"""One row of the objects column to (boxes [k, 4], categories [k])."""
		if isinstance(entry, list):  # list of per-box dicts
			boxes = [item[self.bbox_key] for item in entry]
			categories = [item[self.category_key] for item in entry]
		else:
			boxes = entry[self.bbox_key]
			categories = entry[self.category_key]
		boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
		return boxes, broadcast_categories(categories, len(boxes))

	def _read_image_sizes(self, dataset: Dataset, image_col: str, batch_size: int = 64) -> np.ndarray:
		"""Image (width, height) from the file headers, so no image is decoded."""
		from PIL import Image as PILImage

		undecoded = dataset.cast_column(image_col, datasets.Image(decode=False))
		sizes = np.zeros((len(undecoded), 2), dtype=np.int64)
		for start in tqdm(range(0, len(undecoded), batch_size), desc="Reading image sizes"):
			batch = undecoded[start : start + batch_size][image_col]
			for offset, image in enumerate(batch):
				source = io.BytesIO(image["bytes"]) if image.get("bytes") is not None else image["path"]
				with PILImage.open(source) as pil_image:
					sizes[start + offset] = pil_image.size
		return sizes

	def _label_names(self, dataset: Dataset, needed: int) -> List[str]:
		"""ClassLabel names when the dataset has them, otherwise class_<id>."""
		category_feature = _unwrap_feature(dataset.features[self.objects_col])
		if isinstance(category_feature, dict) and self.category_key in category_feature:
			category_feature = _unwrap_feature(category_feature[self.category_key])
			if isinstance(category_feature, datasets.ClassLabel):
				return list(category_feature.names)
		return [f"class_{class_id}" for class_id in range(needed)]
