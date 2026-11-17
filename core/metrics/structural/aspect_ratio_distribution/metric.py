"""
AspectRatioDistributionMetric
=============================
Box aspect ratio (width / height), overall and per class.

Per entry: n, median, p5, p95, std_log_ratio, pct_extreme.
pct_extreme is the share of boxes with a ratio above 3 or below 1/3, in percent.

Output keys
-----------
overall    : entry over all boxes
per_class  : {class_name: entry}
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric

EXTREME_RATIO = 3.0


def _summarise(ratios: np.ndarray) -> Dict[str, Any]:
	extreme = (ratios > EXTREME_RATIO) | (ratios < 1.0 / EXTREME_RATIO)
	return {
		"n": int(len(ratios)),
		"median": round(float(np.median(ratios)), 4),
		"p5": round(float(np.percentile(ratios, 5)), 4),
		"p95": round(float(np.percentile(ratios, 95)), 4),
		"std_log_ratio": round(float(np.log(ratios).std()), 4),
		"pct_extreme": round(float(extreme.mean() * 100), 2),
	}


class AspectRatioDistributionMetric(BaseMetric):
	name = "aspect_ratio_distribution"
	phase = 1

	def run(self, box_table: BoxTable) -> Dict[str, Any]:  # type: ignore[override]
		ratios = box_table.aspect_ratios()
		per_class = {}
		for class_id, class_name in enumerate(box_table.label_names):
			class_ratios = ratios[box_table.class_id == class_id]
			if len(class_ratios):
				per_class[class_name] = _summarise(class_ratios)
		return {"overall": _summarise(ratios), "per_class": per_class}
