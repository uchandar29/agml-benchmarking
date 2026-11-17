"""
BoxTable
========
Flat table of every ground-truth box in a detection dataset. Row m is one box:
the image it belongs to, its class and its corners in pixels. Image sizes are
stored once per image, so relative measures (box area / image area) are cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BoxTable:
	image_index: np.ndarray    # int64 [M], row of the image in the full dataset
	class_id: np.ndarray       # int64 [M]
	boxes_xyxy: np.ndarray     # float32 [M, 4], pixels
	image_sizes: np.ndarray    # int64 [N, 2], (width, height)
	label_names: List[str]
	dropped_boxes: int = 0
	_order: np.ndarray = field(default=None, init=False, repr=False, compare=False)
	_offsets: np.ndarray = field(default=None, init=False, repr=False, compare=False)

	@property
	def num_images(self) -> int:
		return len(self.image_sizes)

	@property
	def num_boxes(self) -> int:
		return len(self.image_index)

	@property
	def num_classes(self) -> int:
		return len(self.label_names)

	def boxes_per_image(self) -> np.ndarray:
		return np.bincount(self.image_index, minlength=self.num_images)

	def widths(self) -> np.ndarray:
		return self.boxes_xyxy[:, 2] - self.boxes_xyxy[:, 0]

	def heights(self) -> np.ndarray:
		return self.boxes_xyxy[:, 3] - self.boxes_xyxy[:, 1]

	def areas(self) -> np.ndarray:
		return self.widths() * self.heights()

	def image_areas(self) -> np.ndarray:
		"""Area in pixels of the image each box sits in, one value per box."""
		sizes = self.image_sizes[self.image_index].astype(np.float64)
		return sizes[:, 0] * sizes[:, 1]

	def aspect_ratios(self) -> np.ndarray:
		"""Width over height."""
		return self.widths() / np.maximum(self.heights(), 1e-6)

	def centers(self) -> np.ndarray:
		return np.stack([
			(self.boxes_xyxy[:, 0] + self.boxes_xyxy[:, 2]) / 2.0,
			(self.boxes_xyxy[:, 1] + self.boxes_xyxy[:, 3]) / 2.0,
		], axis=1)

	def subset(self, image_indices) -> np.ndarray:
		"""Row numbers (ascending) of the boxes that belong to the given images."""
		if self._order is None:
			self._order = np.argsort(self.image_index, kind="stable")
			self._offsets = np.concatenate([[0], np.cumsum(self.boxes_per_image())])
		image_indices = np.atleast_1d(np.asarray(image_indices, dtype=np.int64))
		parts = [self._order[self._offsets[i]:self._offsets[i + 1]] for i in image_indices]
		if not parts:
			return np.zeros(0, dtype=np.int64)
		return np.sort(np.concatenate(parts)) if len(parts) > 1 else parts[0]
