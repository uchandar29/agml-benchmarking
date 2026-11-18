"""
BoxOcclusionMetric
==================
Box Occlusion Ratio (BOR): how much the boxes of an image cover each other.

	occ(box) = area of the box covered by the union of the other boxes in its image / area of the box
	image score = mean occ of its boxes (images with 0 or 1 box score 0)

Output keys
-----------
mean_image_bor, median_image_bor, p90_image_bor : image scores over all images
pct_boxes_occ_above_half     : boxes with occ > OCCLUDED_ABOVE, in percent
mean_high_iou_pairs_per_image: box pairs with IoU >= HIGH_IOU, averaged over all images
images_evaluated             : all images, including those with fewer than two boxes
boxes_evaluated
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric

OCCLUDED_ABOVE = 0.5
HIGH_IOU = 0.5


def _intersections(boxes: np.ndarray):
	"""Pairwise intersection widths, heights (both clipped at 0) for boxes [n, 4]."""
	widths = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])
	heights = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1])
	return np.clip(widths, 0, None), np.clip(heights, 0, None)


def _union_area(clipped: np.ndarray) -> float:
	"""Exact area of the union of boxes [k, 4], by coordinate compression."""
	xs = np.unique(np.concatenate([clipped[:, 0], clipped[:, 2]]))
	ys = np.unique(np.concatenate([clipped[:, 1], clipped[:, 3]]))
	counts = np.zeros((len(xs), len(ys)), dtype=np.int32)
	x_low = np.searchsorted(xs, clipped[:, 0])
	x_high = np.searchsorted(xs, clipped[:, 2])
	y_low = np.searchsorted(ys, clipped[:, 1])
	y_high = np.searchsorted(ys, clipped[:, 3])
	# 2D difference array: +1 at the low corner, -1 at the high edges, then prefix sums
	np.add.at(counts, (x_low, y_low), 1)
	np.add.at(counts, (x_high, y_low), -1)
	np.add.at(counts, (x_low, y_high), -1)
	np.add.at(counts, (x_high, y_high), 1)
	covered = counts.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0
	return float((covered * np.outer(np.diff(xs), np.diff(ys))).sum())


def box_occlusion_scores(boxes_xyxy: np.ndarray) -> np.ndarray:
	"""occ per box: share of its area covered by the union of the other boxes."""
	boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
	count = len(boxes)
	scores = np.zeros(count)
	if count < 2:
		return scores
	widths, heights = _intersections(boxes)
	touching = (widths > 0) & (heights > 0)
	np.fill_diagonal(touching, False)
	areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	for number in np.flatnonzero(touching.any(axis=1) & (areas > 0)):
		others = boxes[touching[number]]
		own = boxes[number]
		clipped = np.stack([
			np.maximum(others[:, 0], own[0]), np.maximum(others[:, 1], own[1]),
			np.minimum(others[:, 2], own[2]), np.minimum(others[:, 3], own[3]),
		], axis=1)
		scores[number] = _union_area(clipped) / areas[number]
	return np.clip(scores, 0.0, 1.0)


def high_iou_pair_count(boxes_xyxy: np.ndarray, threshold: float = HIGH_IOU) -> int:
	"""Number of unordered box pairs with IoU >= threshold."""
	boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
	if len(boxes) < 2:
		return 0
	widths, heights = _intersections(boxes)
	inter = widths * heights
	areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	union = areas[:, None] + areas[None, :] - inter
	iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
	return int((np.triu(iou, k=1) >= threshold).sum())


class BoxOcclusionMetric(BaseMetric):
	name = "box_occlusion"
	phase = 1

	def run(self, box_table: BoxTable) -> Dict[str, Any]:  # type: ignore[override]
		if box_table.num_boxes == 0:
			return {"skipped": True, "reason": "The dataset has no boxes.", "images_evaluated": 0}

		image_scores = np.zeros(box_table.num_images)
		pair_counts = np.zeros(box_table.num_images)
		occluded_boxes = 0
		for image in range(box_table.num_images):
			rows = box_table.subset(image)
			if len(rows) < 2:
				continue
			boxes = box_table.boxes_xyxy[rows]
			scores = box_occlusion_scores(boxes)
			image_scores[image] = scores.mean()
			pair_counts[image] = high_iou_pair_count(boxes, HIGH_IOU)
			occluded_boxes += int((scores > OCCLUDED_ABOVE).sum())

		return {
			"images_evaluated": int(box_table.num_images),
			"boxes_evaluated": int(box_table.num_boxes),
			"mean_image_bor": round(float(image_scores.mean()), 4),
			"median_image_bor": round(float(np.median(image_scores)), 4),
			"p90_image_bor": round(float(np.percentile(image_scores, 90)), 4),
			"pct_boxes_occ_above_half": round(occluded_boxes / box_table.num_boxes * 100, 2),
			"mean_high_iou_pairs_per_image": round(float(pair_counts.mean()), 4),
			"occluded_above": OCCLUDED_ABOVE,
			"high_iou": HIGH_IOU,
		}
