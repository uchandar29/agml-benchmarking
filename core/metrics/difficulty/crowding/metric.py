"""
CrowdingMetric
==============
Clark-Evans nearest neighbour ratio of the box centres, per image with at least two boxes:

	R = mean nearest-neighbour distance / (0.5 / sqrt(n / image_area))

R is about 1 for randomly placed boxes, below 1 when boxes cluster, above 1 when they are evenly spread.

Output keys
-----------
images_evaluated   : images with >= 2 boxes
mean_R, median_R
pct_clustered      : images with R < CLUSTERED_BELOW, in percent
pct_regular        : images with R > REGULAR_ABOVE, in percent
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial import cKDTree

from core.data import BoxTable
from core.metrics.base import BaseMetric

CLUSTERED_BELOW = 1.0
REGULAR_ABOVE = 1.2


def clark_evans_ratio(centers: np.ndarray, image_area: float) -> float:
	count = len(centers)
	distances, _ = cKDTree(centers).query(centers, k=2)
	expected = 0.5 / np.sqrt(count / image_area)
	return float(distances[:, 1].mean() / expected)


class CrowdingMetric(BaseMetric):
	name = "crowding"
	phase = 1

	def run(self, box_table: BoxTable) -> Dict[str, Any]:  # type: ignore[override]
		order = np.argsort(box_table.image_index, kind="stable")
		sorted_images = box_table.image_index[order]
		centers = box_table.centers()[order]
		starts = np.flatnonzero(np.concatenate([[True], sorted_images[1:] != sorted_images[:-1]]))
		ends = np.concatenate([starts[1:], [len(order)]])

		ratios = []
		for start, end in zip(starts, ends):
			if end - start < 2:
				continue
			width, height = box_table.image_sizes[sorted_images[start]]
			ratios.append(clark_evans_ratio(centers[start:end], float(width) * float(height)))

		if not ratios:
			return {"skipped": True, "reason": "No image has two or more boxes.", "images_evaluated": 0}
		ratios = np.array(ratios)
		return {
			"images_evaluated": int(len(ratios)),
			"mean_R": round(float(ratios.mean()), 4),
			"median_R": round(float(np.median(ratios)), 4),
			"pct_clustered": round(float((ratios < CLUSTERED_BELOW).mean() * 100), 2),
			"pct_regular": round(float((ratios > REGULAR_ABOVE).mean() * 100), 2),
			"clustered_below": CLUSTERED_BELOW,
			"regular_above": REGULAR_ABOVE,
		}
