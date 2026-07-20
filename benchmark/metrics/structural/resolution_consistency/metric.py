"""
ResolutionConsistencyMetric
===========================
Characterises the spread of image dimensions across the full dataset.

Why this matters
----------------
A high coefficient of variation (CV) on image area reveals that images come
from heterogeneous capture pipelines — different cameras, scraping sources,
or sensors.  This introduces implicit difficulty (a model must handle wildly
different scales) that has nothing to do with the agricultural task itself.
A low CV suggests a controlled, uniform capture setup.

Implementation note
-------------------
PIL Image.size returns (width, height) by reading only the image header —
it does not decode the full pixel buffer.  This makes the metric fast even
on the 26 GB agarwood dataset where per-image sizes are large.

Computed on the full dataset.

Output keys
-----------
total_images     : int
width            : {mean, std, min, max}   (pixels)
height           : {mean, std, min, max}   (pixels)
aspect_ratio     : {mean, std, min, max}   (width / height)
area_cv          : float  — coefficient of variation of image area (std / mean)
                           lower → more consistent
mode_distribution: {mode_str: count}       (e.g. {"RGB": 5472})
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List

from datasets import Dataset
from tqdm import tqdm

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class ResolutionConsistencyMetric(BaseMetric):
    name = "resolution_consistency"
    phase = 1

    def run(  # type: ignore[override]
        self,
        full_dataset: Dataset,
        schema: DatasetSchema,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        image_col = schema.image_col

        widths:  List[int]   = []
        heights: List[int]   = []
        modes:   List[str]   = []

        n = len(full_dataset)
        for start in tqdm(range(0, n, batch_size), desc="Measuring resolutions"):
            batch = full_dataset[start: start + batch_size]
            for img in batch[image_col]:
                w, h = img.size          # header-only read; no full decode
                widths.append(w)
                heights.append(h)
                modes.append(img.mode)

        aspect_ratios = [w / h for w, h in zip(widths, heights)]
        areas         = [w * h for w, h in zip(widths, heights)]

        def _stats(values: List) -> Dict[str, float]:
            return {
                "mean": round(statistics.mean(values), 2),
                "std":  round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
                "min":  min(values),
                "max":  max(values),
            }

        area_mean = statistics.mean(areas)
        area_std  = statistics.stdev(areas) if len(areas) > 1 else 0.0
        area_cv   = round(area_std / area_mean, 4) if area_mean > 0 else 0.0

        return {
            "total_images":      n,
            "width":             _stats(widths),
            "height":            _stats(heights),
            "aspect_ratio":      _stats(aspect_ratios),
            "area_cv":           area_cv,
            "mode_distribution": dict(Counter(modes)),
        }
