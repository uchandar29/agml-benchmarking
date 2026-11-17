"""
Embeddings of whole images and of box crops, using any ImageEncoder from core/embeddings/encoders.py.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from core.data import BoxTable, to_rgb

from .encoders import ImageEncoder


def _encode_pending(encoder: ImageEncoder, pending: list) -> np.ndarray:
	import torch

	return encoder.encode(torch.stack(pending))


def embed_images(encoder: ImageEncoder, dataset: Dataset, image_col: str, batch_size: int = 64) -> np.ndarray:
	"""One embedding per row of the dataset, in row order."""
	chunks: List[np.ndarray] = []
	for start in tqdm(range(0, len(dataset), batch_size), desc="Embedding images"):
		images = dataset[start : start + batch_size][image_col]
		chunks.append(_encode_pending(encoder, [encoder.preprocess(to_rgb(image)) for image in images]))
	return np.concatenate(chunks, axis=0).astype(np.float32)


def sample_boxes_per_class(box_table: BoxTable, max_per_class: int, max_total: int, seed: int = 42) -> np.ndarray:
	"""Sorted box row numbers: at most max_per_class per class, then at most max_total overall."""
	rng = np.random.default_rng(seed)
	chosen = []
	for class_id in range(box_table.num_classes):
		rows = np.flatnonzero(box_table.class_id == class_id)
		if len(rows) > max_per_class:
			rows = rng.choice(rows, size=max_per_class, replace=False)
		chosen.append(rows)
	rows = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
	if len(rows) > max_total:
		rows = rng.choice(rows, size=max_total, replace=False)
	return np.sort(rows).astype(np.int64)


def embed_crops(
	encoder: ImageEncoder,
	dataset: Dataset,
	image_col: str,
	box_table: BoxTable,
	box_indices: np.ndarray,
	batch_size: int = 64,
	context_pad: float = 0.1,
	min_side: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Embed one crop per box. Each crop gets context_pad of its own width and height
	on every side. Boxes whose own width or height is under min_side pixels are skipped.
	Boxes are grouped by image so every image is decoded once.

	Returns (embeddings [K, D], kept_box_indices [K]), rows in the same order.
	"""
	box_indices = np.asarray(box_indices, dtype=np.int64)
	order = np.argsort(box_table.image_index[box_indices], kind="stable")
	box_indices = box_indices[order]
	images_of_boxes = box_table.image_index[box_indices]
	group_starts = np.flatnonzero(np.concatenate([[True], images_of_boxes[1:] != images_of_boxes[:-1]]))
	group_ends = np.concatenate([group_starts[1:], [len(box_indices)]])

	chunks: List[np.ndarray] = []
	kept: List[int] = []
	pending: list = []

	def flush() -> None:
		if pending:
			chunks.append(_encode_pending(encoder, pending))
			pending.clear()

	for start, end in tqdm(list(zip(group_starts, group_ends)), desc="Embedding crops"):
		image_number = int(images_of_boxes[start])
		image = to_rgb(dataset[image_number][image_col])
		width, height = image.size
		for box_index in box_indices[start:end]:
			x_min, y_min, x_max, y_max = box_table.boxes_xyxy[box_index]
			box_width, box_height = x_max - x_min, y_max - y_min
			if box_width < min_side or box_height < min_side:
				continue
			left = max(0, int(np.floor(x_min - context_pad * box_width)))
			top = max(0, int(np.floor(y_min - context_pad * box_height)))
			right = min(width, int(np.ceil(x_max + context_pad * box_width)))
			bottom = min(height, int(np.ceil(y_max + context_pad * box_height)))
			if right <= left or bottom <= top:
				continue
			pending.append(encoder.preprocess(image.crop((left, top, right, bottom))))
			kept.append(int(box_index))
			if len(pending) >= batch_size:
				flush()
	flush()

	if not chunks:
		return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
	return np.concatenate(chunks, axis=0).astype(np.float32), np.array(kept, dtype=np.int64)
