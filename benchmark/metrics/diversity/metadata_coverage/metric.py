"""
MetadataCoverageMetric
======================
Builds per-class contingency tables for every detected metadata column and
reports how uniformly each class is represented across metadata values.

Why this matters
----------------
A class that was only photographed under one condition (single variety, fixed
scale, one lighting setup) produces a narrow visual template.  Models trained
on it generalise poorly.  This metric makes that narrowness explicit.

Graceful no-op
--------------
If the dataset has no metadata columns (e.g. rice_leaf_disease has only
'image' + 'label') this metric returns {"skipped": true, ...} without
raising an error.  The calling code needs no special-casing.

Computed on the full dataset.

Output keys (when not skipped)
-------------------------------
skipped           : false
metadata_columns  : [str, ...]
coverage          : { col → { class_name → { value → count } } }
normalized_entropy: { col → { class_name → float } }
                    Entropy of the metadata-value distribution within each
                    class, normalised to [0, 1].
                    1.0 = perfectly spread across all values.
                    0.0 = all images in that class share the same value.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict

from datasets import Dataset

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class MetadataCoverageMetric(BaseMetric):
    name = "metadata_coverage"
    phase = 1

    def run(  # type: ignore[override]
        self,
        full_dataset: Dataset,
        schema: DatasetSchema,
    ) -> Dict[str, Any]:
        if not schema.metadata_cols:
            return {
                "skipped": True,
                "reason": "No metadata columns detected in dataset schema.",
            }

        label_col   = schema.label_col
        label_names = schema.label_names

        # Pull only the columns we need — no images decoded
        cols = [label_col] + schema.metadata_cols
        data = {col: full_dataset[col] for col in cols}

        coverage: Dict[str, Dict[str, Dict[str, int]]] = {}

        for meta_col in schema.metadata_cols:
            # { class_name → { meta_value → count } }
            tally: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for label_int, meta_val in zip(data[label_col], data[meta_col]):
                class_name = label_names[label_int]
                tally[class_name][str(meta_val)] += 1

            coverage[meta_col] = {cls: dict(vals) for cls, vals in tally.items()}

        # Normalised entropy per (metadata_col, class) pair
        entropy_map: Dict[str, Dict[str, float]] = {}
        for meta_col, class_dist in coverage.items():
            entropy_map[meta_col] = {}
            for class_name, val_counts in class_dist.items():
                total = sum(val_counts.values())
                K = len(val_counts)
                H = 0.0
                for count in val_counts.values():
                    p = count / total
                    if p > 0:
                        H -= p * math.log(p)
                norm_H = H / math.log(K) if K > 1 else 1.0
                entropy_map[meta_col][class_name] = round(norm_H, 4)

        return {
            "skipped":            False,
            "metadata_columns":   schema.metadata_cols,
            "coverage":           coverage,
            "normalized_entropy": entropy_map,
        }
