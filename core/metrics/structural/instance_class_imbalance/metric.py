"""
InstanceClassImbalanceMetric
============================
Class imbalance counted in boxes instead of images, on the training split only.
Same output keys as ClassImbalanceMetric, with total_train_boxes instead of total_train_examples.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.data import BoxTable
from core.metrics.base import BaseMetric
from core.metrics.common import imbalance_stats


class InstanceClassImbalanceMetric(BaseMetric):
	name = "instance_class_imbalance"
	phase = 1

	def run(self, box_table: BoxTable, train_image_indices: np.ndarray) -> Dict[str, Any]:  # type: ignore[override]
		rows = box_table.subset(train_image_indices)
		counts = np.bincount(box_table.class_id[rows], minlength=box_table.num_classes)
		named = {box_table.label_names[class_id]: int(count) for class_id, count in enumerate(counts) if count > 0}
		return {
			**imbalance_stats(named, num_classes=box_table.num_classes),
			"total_train_boxes": int(counts.sum()),
		}
