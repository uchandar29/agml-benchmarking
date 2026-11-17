"""
Stratified splits for detection datasets.

An image has several boxes and so no single label. Each image is stratified on
the rarest class it contains (rarest by total box count), so rare classes show
up in every split. Images without boxes get their own label, num_classes.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset

from .box_table import BoxTable
from .split_manager import SplitManager, SplitResult


def proxy_labels(box_table: BoxTable) -> np.ndarray:
	class_counts = np.bincount(box_table.class_id, minlength=box_table.num_classes)
	labels = np.full(box_table.num_images, box_table.num_classes, dtype=np.int64)
	# sort boxes by image, then by class rarity, so the first box of each image is its rarest class
	order = np.lexsort((box_table.class_id, class_counts[box_table.class_id], box_table.image_index))
	sorted_images = box_table.image_index[order]
	first_of_image = np.concatenate([[True], sorted_images[1:] != sorted_images[:-1]])
	labels[sorted_images[first_of_image]] = box_table.class_id[order][first_of_image]
	return labels


def split_detection_dataset(
	dataset: Dataset,
	box_table: BoxTable,
	train_ratio: float = 0.70,
	val_ratio: float = 0.15,
	seed: int = 42,
) -> SplitResult:
	manager = SplitManager(train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
	return manager.split_labels(dataset, proxy_labels(box_table))
