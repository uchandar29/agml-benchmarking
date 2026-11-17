"""
BoxSizeDistributionMetric
=========================
Box sizes in two ways.

coco      absolute pixels, COCO buckets: small < 32^2, medium 32^2 to 96^2, large > 96^2
relative  box area / image area. Buckets are the terciles of this dataset, so they
          say how sizes are spread inside the dataset whatever the image resolution

Output keys
-----------
coco                  : {n_small, n_medium, n_large, pct_small, pct_medium, pct_large}
relative_area         : {p5, p25, p50, p75, p95}
relative_terciles     : {cutoff_low, cutoff_high, n_small, n_medium, n_large, pct_small, pct_medium, pct_large}
per_class_median_relative_area : {class_name: float}
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric

SMALL_AREA = 32 ** 2
LARGE_AREA = 96 ** 2


def _bucket_summary(small: int, medium: int, large: int) -> Dict[str, Any]:
	total = max(small + medium + large, 1)
	return {
		"n_small": small,
		"n_medium": medium,
		"n_large": large,
		"pct_small": round(small / total * 100, 2),
		"pct_medium": round(medium / total * 100, 2),
		"pct_large": round(large / total * 100, 2),
	}


class BoxSizeDistributionMetric(BaseMetric):
	name = "box_size_distribution"
	phase = 1

	def run(self, box_table: BoxTable) -> Dict[str, Any]:  # type: ignore[override]
		areas = box_table.areas()
		small = int((areas < SMALL_AREA).sum())
		large = int((areas > LARGE_AREA).sum())
		medium = len(areas) - small - large

		relative = areas / box_table.image_areas()
		percentiles = np.percentile(relative, [5, 25, 50, 75, 95])
		cutoff_low, cutoff_high = np.percentile(relative, [100 / 3, 200 / 3])
		tercile_small = int((relative < cutoff_low).sum())
		tercile_large = int((relative > cutoff_high).sum())

		per_class = {}
		for class_id, class_name in enumerate(box_table.label_names):
			class_values = relative[box_table.class_id == class_id]
			if len(class_values):
				per_class[class_name] = round(float(np.median(class_values)), 6)

		return {
			"coco": _bucket_summary(small, medium, large),
			"relative_area": {
				name: round(float(value), 6) for name, value in zip(["p5", "p25", "p50", "p75", "p95"], percentiles)
			},
			"relative_terciles": {
				"cutoff_low": round(float(cutoff_low), 6),
				"cutoff_high": round(float(cutoff_high), 6),
				**_bucket_summary(tercile_small, len(relative) - tercile_small - tercile_large, tercile_large),
			},
			"per_class_median_relative_area": per_class,
		}
