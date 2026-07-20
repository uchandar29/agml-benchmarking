"""
FeatureSeparabilityMetric
=========================
Measures how well the ground-truth label classes are separated in the DINOv2
embedding space, without training any model on this dataset.

Two complementary scores are computed:

Silhouette Score  (sklearn)
---------------------------
  Range  : [-1, 1]  — higher is better.
  Formula: s(i) = (b(i) − a(i)) / max(a(i), b(i))
             a(i) = mean intra-cluster distance for sample i
             b(i) = mean distance to nearest other cluster for sample i
  Metric : cosine distance (1 − cosine_similarity) since embeddings are
           L2-normalised — this is equivalent to and faster than Euclidean
           on the unit hypersphere.
  Sampling: O(N²) in memory, so capped at max_silhouette_samples.
            Davies-Bouldin uses the full dataset (it is O(K·N)).

Davies-Bouldin Index  (sklearn)
--------------------------------
  Range  : [0, ∞)  — lower is better.
  Measures the average ratio of within-cluster scatter to between-cluster
  distance.  Uses Euclidean distance internally; on L2-normalised vectors
  this is monotonically related to cosine distance.

Interpretation note
-------------------
Low separability in CLIP/DINOv2 space can mean two different things:
  (a) The classes are genuinely hard to distinguish visually — a real
      challenge that the benchmark should capture.
  (b) The labels are noisy — visually similar images have different labels.
Phase 3 (LabelNoiseMetric) will disambiguate.

Output keys
-----------
embed_model              : str
n_total                  : int
n_silhouette_samples     : int    — actual samples used (≤ max_silhouette_samples)
silhouette_score         : float
silhouette_interpretation: str
davies_bouldin_index     : float
davies_bouldin_interpretation: str
per_class_silhouette     : {class_name: float}   — per-class mean silhouette
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class FeatureSeparabilityMetric(BaseMetric):
    name = "feature_separability"
    phase = 2

    def run(  # type: ignore[override]
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        schema: DatasetSchema,
        embed_model: str,
        max_silhouette_samples: int = 10_000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        try:
            from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
        except ImportError:
            raise ImportError(
                "Phase 2 requires scikit-learn.  "
                "Install with:  pip install scikit-learn"
            )

        N = len(embeddings)
        rng = np.random.default_rng(seed)

        # ── Sample for silhouette (O(N²) memory) ─────────────────────────────
        if N > max_silhouette_samples:
            # Stratified sample: preserve class proportions
            sample_idx = _stratified_sample(labels, max_silhouette_samples, rng)
            X_sil = embeddings[sample_idx]
            y_sil = labels[sample_idx]
        else:
            X_sil = embeddings
            y_sil = labels
            sample_idx = np.arange(N)

        n_sampled = len(X_sil)

        # ── Silhouette ────────────────────────────────────────────────────────
        sil_global = float(silhouette_score(X_sil, y_sil, metric="cosine"))

        # Per-class mean silhouette
        sil_samples = silhouette_samples(X_sil, y_sil, metric="cosine")
        per_class_sil: Dict[str, float] = {}
        for cls_idx, cls_name in enumerate(schema.label_names):
            mask = y_sil == cls_idx
            per_class_sil[cls_name] = (
                round(float(sil_samples[mask].mean()), 4) if mask.any() else 0.0
            )

        # ── Davies-Bouldin (O(K·N) — use full dataset) ───────────────────────
        db = float(davies_bouldin_score(embeddings, labels))

        return {
            "embed_model":                    embed_model,
            "n_total":                        N,
            "n_silhouette_samples":           n_sampled,
            "silhouette_score":               round(sil_global, 4),
            "silhouette_interpretation":      _interpret_silhouette(sil_global),
            "davies_bouldin_index":           round(db, 4),
            "davies_bouldin_interpretation":  _interpret_davies_bouldin(db),
            "per_class_silhouette":           per_class_sil,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stratified_sample(
    labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices of a stratified random sample of size ≤ n."""
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    idx_list = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(labels == cls)[0]
        k = max(1, round(n * cnt / total))
        k = min(k, cnt)
        chosen = rng.choice(cls_idx, size=k, replace=False)
        idx_list.append(chosen)
    return np.concatenate(idx_list)


def _interpret_silhouette(s: float) -> str:
    if s > 0.70:  return "strong separation"
    if s > 0.50:  return "reasonable separation"
    if s > 0.25:  return "moderate separation"
    if s > 0.00:  return "weak separation"
    return "overlapping or no separation"


def _interpret_davies_bouldin(db: float) -> str:
    if db < 0.50:  return "well-separated clusters"
    if db < 1.00:  return "moderate separation"
    if db < 2.00:  return "some overlap between clusters"
    return "high overlap — classes not well-separated"
