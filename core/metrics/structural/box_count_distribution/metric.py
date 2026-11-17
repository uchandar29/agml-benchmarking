"""
BoxCountDistributionMetric
==========================
How many boxes each image has, over the full dataset (images without boxes count as 0).

Output keys
-----------
mean, median, std, min, max, p90   -- boxes per image
pct_empty_images                   -- images with no box, in percent
histogram                          -- {"bin_edges": [...], "counts": [...]}
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric


class BoxCountDistributionMetric(BaseMetric):
	name = "box_count_distribution"
	phase = 1

	def __init__(self, histogram_bins: int = 20) -> None:
		self.histogram_bins = histogram_bins

	def run(self, box_table: BoxTable) -> Dict[str, Any]:  # type: ignore[override]
		per_image = box_table.boxes_per_image()
		counts, bin_edges = np.histogram(per_image, bins=min(self.histogram_bins, max(int(per_image.max()), 1)))
		return {
			"mean": round(float(per_image.mean()), 4),
			"median": float(np.median(per_image)),
			"std": round(float(per_image.std()), 4),
			"min": int(per_image.min()),
			"max": int(per_image.max()),
			"p90": float(np.percentile(per_image, 90)),
			"pct_empty_images": round(float((per_image == 0).mean() * 100), 2),
			"histogram": {
				"bin_edges": [round(float(edge), 2) for edge in bin_edges],
				"counts": [int(count) for count in counts],
			},
		}
