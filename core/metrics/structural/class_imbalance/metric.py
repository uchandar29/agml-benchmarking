"""
ClassImbalanceMetric
====================
Measures label distribution skew across all classes in the training split.

Why training split only
-----------------------
Class balance should be assessed on what a model will actually learn from.
Measuring imbalance on the full dataset or held-out sets would mix signal
from the learning regime with evaluation artefacts.

Output keys
-----------
counts               : {class_name: int}  — examples per class
imbalance_ratio      : float              — max_count / min_count
                                            (1.0 = perfectly balanced)
normalized_entropy   : float in [0, 1]   — H / log(K); 1.0 = uniform
most_frequent_class  : str
least_frequent_class : str
total_train_examples : int
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from datasets import Dataset

from core.metrics.base import BaseMetric
from core.metrics.common import imbalance_stats
from core.data import DatasetSchema


class ClassImbalanceMetric(BaseMetric):
    name = "class_imbalance"
    phase = 1

    def run(  # type: ignore[override]
        self,
        train_dataset: Dataset,
        schema: DatasetSchema,
    ) -> Dict[str, Any]:
        # Fetch the label column directly — avoids decoding any images.
        labels = list(train_dataset[schema.label_col])
        counts = Counter(labels)

        # Map integer keys → class name strings
        named: Dict[str, int] = {
            schema.label_names[k]: v
            for k, v in sorted(counts.items())
        }

        stats = imbalance_stats(named, num_classes=schema.num_classes)
        return {
            **stats,
            "total_train_examples": sum(named.values()),
        }
