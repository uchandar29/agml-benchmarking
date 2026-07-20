"""
IntraClassDiversityMetric
=========================
Measures how visually varied each class is by computing the mean L2 distance
from each image's DINOv2 embedding to its class centroid.

Why this matters
----------------
A class with low intra-class diversity consists largely of near-identical
shots — same angle, same background, same lighting.  A model trained on
such data learns a narrow visual template and generalises poorly to field
conditions where the same disease presents differently.

Agricultural relevance: disease symptoms appear differently depending on
growth stage, variety, lighting, and camera distance.  A high-diversity
class covers more of that variation; a low-diversity class covers very
little of it.

Computation
-----------
For each class c:
  centroid_c = mean(embeddings[labels == c], axis=0)    (not re-normalised)
  diversity_c = mean(||emb_i − centroid_c||₂)  for i ∈ class c

L2 distance on unit-norm vectors ranges [0, 2]:
  0 = identical,  √2 ≈ 1.41 = orthogonal,  2 = antipodal

Output keys
-----------
embed_model          : str
per_class_diversity  : {class_name: float}   — mean L2 to centroid
mean_diversity       : float                  — macro-average across classes
min_diversity_class  : str                    — narrowest class (flag for review)
max_diversity_class  : str                    — widest class
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class IntraClassDiversityMetric(BaseMetric):
    name = "intra_class_diversity"
    phase = 2

    def run(  # type: ignore[override]
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        schema: DatasetSchema,
        embed_model: str,
    ) -> Dict[str, Any]:
        per_class: Dict[str, float] = {}

        for cls_idx, cls_name in enumerate(schema.label_names):
            mask = labels == cls_idx
            cls_embeddings = embeddings[mask]

            if len(cls_embeddings) == 0:
                per_class[cls_name] = 0.0
                continue

            centroid = cls_embeddings.mean(axis=0)          # (D,)
            diffs    = cls_embeddings - centroid             # (n_cls, D)
            dists    = np.linalg.norm(diffs, axis=1)        # (n_cls,)
            per_class[cls_name] = round(float(dists.mean()), 4)

        # Sort by name for deterministic output
        per_class = dict(sorted(per_class.items()))

        mean_div = round(float(np.mean(list(per_class.values()))), 4)
        min_cls  = min(per_class, key=per_class.get)
        max_cls  = max(per_class, key=per_class.get)

        return {
            "embed_model":         embed_model,
            "per_class_diversity": per_class,
            "mean_diversity":      mean_div,
            "min_diversity_class": min_cls,
            "max_diversity_class": max_cls,
        }
