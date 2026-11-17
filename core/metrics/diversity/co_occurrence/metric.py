"""
CoOccurrenceMetric
==================
Which classes appear together in the same image. For classes i and j:

	expected = n_i * n_j / N      (n = images containing the class, N = all images)
	lift     = observed / expected

A pair is flagged when lift > LIFT_THRESHOLD and observed >= MIN_OBSERVED. A model can
use such a pair as a shortcut (find class i, guess class j) instead of looking at the object.

Output keys
-----------
pairs_evaluated      : pairs where both classes occur in at least one image
flagged_pairs        : pairs that pass both thresholds
shortcut_share       : flagged_pairs / pairs_evaluated
top_pairs            : up to 10 pairs by lift, each {class_a, class_b, observed, expected, lift, flagged}
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric

LIFT_THRESHOLD = 2.0
MIN_OBSERVED = 5
TOP_PAIRS = 10


class CoOccurrenceMetric(BaseMetric):
	name = "co_occurrence"
	phase = 1

	def run(self, box_table: BoxTable) -> Dict[str, Any]:  # type: ignore[override]
		if box_table.num_classes < 2:
			return {"skipped": True, "reason": "Co-occurrence needs at least 2 classes."}

		presence = np.zeros((box_table.num_images, box_table.num_classes), dtype=np.int64)
		presence[box_table.image_index, box_table.class_id] = 1
		images_per_class = presence.sum(axis=0)
		observed = presence.T @ presence

		pairs = []
		for first in range(box_table.num_classes):
			for second in range(first + 1, box_table.num_classes):
				if images_per_class[first] == 0 or images_per_class[second] == 0:
					continue
				expected = images_per_class[first] * images_per_class[second] / box_table.num_images
				lift = observed[first, second] / expected
				pairs.append({
					"class_a": box_table.label_names[first],
					"class_b": box_table.label_names[second],
					"observed": int(observed[first, second]),
					"expected": round(float(expected), 4),
					"lift": round(float(lift), 4),
					"flagged": bool(lift > LIFT_THRESHOLD and observed[first, second] >= MIN_OBSERVED),
				})

		flagged = sum(pair["flagged"] for pair in pairs)
		pairs.sort(key=lambda pair: (-pair["lift"], -pair["observed"]))
		return {
			"pairs_evaluated": len(pairs),
			"flagged_pairs": flagged,
			"shortcut_share": round(flagged / len(pairs), 4) if pairs else 0.0,
			"lift_threshold": LIFT_THRESHOLD,
			"min_observed": MIN_OBSERVED,
			"top_pairs": pairs[:TOP_PAIRS],
		}
