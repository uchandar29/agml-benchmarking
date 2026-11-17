"""
MissingAnnotationMetric
=======================
Estimates how many objects the ground truth may have missed. A reference detector
predicts on held-out images. A confident prediction (score >= score_threshold) that
overlaps no ground-truth box (max IoU < iou_threshold) is a candidate missing box.
Class is ignored. The estimate is unverified: a candidate can also be a detector error.

Output keys
-----------
candidate_count, gt_count, missing_rate_estimate = candidates / (gt + candidates)
pct_images_with_candidates, images_evaluated
review_sample : up to 50 candidates {image_index, box, score}, seed 42, for manual review
note          : reminder that the estimate is unverified
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric

REVIEW_SAMPLE_SIZE = 50


class MissingAnnotationMetric(BaseMetric):
	name = "missing_annotation"
	phase = 3

	def __init__(self, score_threshold: float = 0.7, iou_threshold: float = 0.5, seed: int = 42) -> None:
		self.score_threshold = score_threshold
		self.iou_threshold = iou_threshold
		self.seed = seed

	def run(  # type: ignore[override]
		self,
		predictions: Dict[int, Dict[str, np.ndarray]],
		box_table: BoxTable,
		image_indices: Iterable[int],
	) -> Dict[str, Any]:
		import torch
		from torchvision.ops import box_iou

		candidates = []
		gt_count = 0
		images_with_candidates = 0
		image_indices = [int(index) for index in image_indices]

		for image_index in image_indices:
			ground_truth = box_table.boxes_xyxy[box_table.subset([image_index])]
			gt_count += len(ground_truth)
			predicted = predictions[image_index]
			confident = predicted["scores"] >= self.score_threshold
			boxes = predicted["boxes"][confident]
			scores = predicted["scores"][confident]
			if len(boxes) == 0:
				continue
			if len(ground_truth) == 0:
				unmatched = np.ones(len(boxes), dtype=bool)
			else:
				best_iou = box_iou(torch.as_tensor(boxes), torch.as_tensor(ground_truth)).max(dim=1).values.numpy()
				unmatched = best_iou < self.iou_threshold
			if unmatched.any():
				images_with_candidates += 1
			for box, score in zip(boxes[unmatched], scores[unmatched]):
				candidates.append({
					"image_index": image_index,
					"box": [round(float(value), 1) for value in box],
					"score": round(float(score), 4),
				})

		rng = np.random.default_rng(self.seed)
		if len(candidates) > REVIEW_SAMPLE_SIZE:
			chosen = rng.choice(len(candidates), size=REVIEW_SAMPLE_SIZE, replace=False)
			review_sample = [candidates[position] for position in sorted(chosen)]
		else:
			review_sample = candidates

		total = gt_count + len(candidates)
		return {
			"candidate_count": len(candidates),
			"gt_count": gt_count,
			"missing_rate_estimate": round(len(candidates) / total, 4) if total else 0.0,
			"pct_images_with_candidates": round(images_with_candidates / max(len(image_indices), 1) * 100, 2),
			"images_evaluated": len(image_indices),
			"score_threshold": self.score_threshold,
			"iou_threshold": self.iou_threshold,
			"review_sample": review_sample,
			"note": "Unverified estimate: candidates are confident detector boxes that match no ground truth and may be detector errors.",
		}
