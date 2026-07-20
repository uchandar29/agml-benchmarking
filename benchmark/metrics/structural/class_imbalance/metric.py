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

import math
from collections import Counter
from typing import Any, Dict

from datasets import Dataset

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class ClassImbalanceMetric(BaseMetric):
    name = "class_imbalance"
    phase = 1

    def run(  # type: ignore[override]
        self,
        train_dataset: Dataset,
        schema: DatasetSchema,
    ) -> Dict[str, Any]:
        # Fetch the label column directly — avoids decoding any images.
        labels = train_dataset[schema.label_col]
        counts = Counter(labels)

        # Map integer keys → class name strings
        named: Dict[str, int] = {
            schema.label_names[k]: v
            for k, v in sorted(counts.items())
        }

        values = list(named.values())
        total = sum(values)
        K = schema.num_classes
        max_count = max(values)
        min_count = min(values)

        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        # Shannon entropy, normalised to [0, 1]
        entropy = 0.0
        for c in values:
            p = c / total
            if p > 0:
                entropy -= p * math.log(p)
        norm_entropy = entropy / math.log(K) if K > 1 else 1.0

        return {
            "counts": named,
            "imbalance_ratio": round(imbalance_ratio, 4),
            "normalized_entropy": round(norm_entropy, 4),
            "most_frequent_class": max(named, key=named.get),
            "least_frequent_class": min(named, key=named.get),
            "total_train_examples": total,
        }
