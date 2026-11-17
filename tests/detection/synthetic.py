"""Small in-memory detection datasets and BoxTables for the tests."""

from __future__ import annotations

import numpy as np
from datasets import ClassLabel, Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage, ImageDraw

from core.data import BoxTable
from core.embeddings import EncoderSpec, ImageEncoder

CLASS_COLORS = {0: (220, 30, 30), 1: (30, 30, 220)}


def make_box_table(image_sizes, boxes_by_image, class_ids_by_image, label_names) -> BoxTable:
	"""boxes_by_image[i] is a list of xyxy boxes, class_ids_by_image[i] the matching class ids."""
	image_index, class_id, boxes = [], [], []
	for number, (image_boxes, image_classes) in enumerate(zip(boxes_by_image, class_ids_by_image)):
		for box, class_number in zip(image_boxes, image_classes):
			image_index.append(number)
			class_id.append(class_number)
			boxes.append(box)
	return BoxTable(
		image_index=np.array(image_index, dtype=np.int64),
		class_id=np.array(class_id, dtype=np.int64),
		boxes_xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
		image_sizes=np.array(image_sizes, dtype=np.int64),
		label_names=list(label_names),
	)


def make_detection_dataset(num_images: int = 30, size: int = 96, seed: int = 0, bbox_mode: str = "coco") -> Dataset:
	"""Coloured rectangles on grey. Class 0 is red, class 1 is blue. Every 3rd image has no box."""
	rng = np.random.default_rng(seed)
	images, bboxes, categories = [], [], []
	for number in range(num_images):
		image = PILImage.new("RGB", (size, size), (128, 128, 128))
		draw = ImageDraw.Draw(image)
		image_boxes, image_classes = [], []
		if number % 3 != 2:
			for _ in range(rng.integers(2, 5)):
				width, height = rng.integers(14, 32, size=2)
				x = int(rng.integers(0, size - width))
				y = int(rng.integers(0, size - height))
				class_id = int(rng.integers(0, 2))
				draw.rectangle([x, y, x + width, y + height], fill=CLASS_COLORS[class_id])
				if bbox_mode == "yolo":
					image_boxes.append([(x + width / 2) / size, (y + height / 2) / size, width / size, height / size])
				else:
					image_boxes.append([float(x), float(y), float(width), float(height)])
				image_classes.append(class_id)
		images.append(image)
		bboxes.append(image_boxes)
		categories.append(image_classes)
	features = Features({
		"image": Image(),
		"objects": {
			"bbox": Sequence(Sequence(Value("float32"))),
			"categories": Sequence(ClassLabel(names=["red", "blue"])),
		},
	})
	return Dataset.from_dict(
		{"image": images, "objects": [{"bbox": b, "categories": c} for b, c in zip(bboxes, categories)]},
		features=features,
	)


class TinyEncoder(ImageEncoder):
	"""Colour-grid features on the CPU, no weights to download."""

	def __init__(self) -> None:
		super().__init__(EncoderSpec(name="tiny", library="stub", model_id="none"), "cpu")
		self.preprocess = self.to_tensor

	@staticmethod
	def to_tensor(image):
		import torch

		array = np.asarray(image.resize((16, 16)), dtype=np.float32) / 255.0
		return torch.from_numpy(array).permute(2, 0, 1)

	def encode(self, pixel_batch):
		import torch

		pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_batch, 4).flatten(1)
		pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-12)
		return pooled.numpy().astype(np.float32)
