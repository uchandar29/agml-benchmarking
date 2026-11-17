"""
ScaleVarianceMetric
===================
How much box size varies inside each class: the standard deviation of log(area),
and its spread factor exp(std). A factor of 2 means one standard deviation
covers a 2x change in box area. Classes with fewer than 2 boxes are skipped.

Output keys
-----------
per_class              : {class_name: {n, std_log_area, spread_factor}}
mean_std_log_area, mean_spread_factor   : averages over the classes evaluated
classes_evaluated      : int
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric


class ScaleVarianceMetric(BaseMetric):
	name = "scale_variance"
	phase = 1

	def run(self, box_table: BoxTable) -> Dict[str, Any]:  # type: ignore[override]
		log_areas = np.log(box_table.areas().astype(np.float64))
		per_class: Dict[str, Any] = {}
		for class_id, class_name in enumerate(box_table.label_names):
			values = log_areas[box_table.class_id == class_id]
			if len(values) < 2:
				continue
			spread = float(values.std())
			per_class[class_name] = {
				"n": int(len(values)),
				"std_log_area": round(spread, 4),
				"spread_factor": round(float(np.exp(spread)), 4),
			}
		if not per_class:
			return {"skipped": True, "reason": "No class has two or more boxes.", "classes_evaluated": 0}
		return {
			"per_class": per_class,
			"mean_std_log_area": round(float(np.mean([entry["std_log_area"] for entry in per_class.values()])), 4),
			"mean_spread_factor": round(float(np.mean([entry["spread_factor"] for entry in per_class.values()])), 4),
			"classes_evaluated": len(per_class),
		}
