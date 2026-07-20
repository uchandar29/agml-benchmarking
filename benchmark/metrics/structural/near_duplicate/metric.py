"""
NearDuplicateMetric
===================
Detects near-duplicate images using FAISS approximate nearest-neighbour
search over L2-normalised DINOv2 embeddings.

Why not hash-based detection?
------------------------------
The Phase 1 ExactDuplicateMetric catches pixel-identical images.  Near-
duplicates are visually nearly identical images that differ in compression,
cropping, minor brightness shifts, or slight camera angle.  These are
invisible to MD5 hashing but cluster tightly in embedding space.

FAISS index selection
---------------------
  N < faiss_ivf_threshold  → IndexFlatIP   (exact search, no training needed)
  N ≥ faiss_ivf_threshold  → IndexIVFFlat  (approximate, ANN via Voronoi cells)

Both use inner product (IP) metric, which equals cosine similarity for
L2-normalised vectors.

Cross-split near-duplicates
----------------------------
Pairs that straddle train/val or train/test are reported separately —
they are a data-leakage risk stronger than exact duplicates because they
are harder to detect manually.

Output keys
-----------
total_images              : int
embed_model               : str
threshold                 : float
near_duplicate_count      : int   — images in at least one near-dup pair
near_duplicate_rate       : float
near_duplicate_groups     : int   — distinct near-dup clusters
cross_split_near_duplicates : int — images in pairs that span ≥ 2 splits
faiss_index_type          : str   — 'IndexFlatIP' or 'IndexIVFFlat'
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Set

import numpy as np

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class NearDuplicateMetric(BaseMetric):
    name = "near_duplicate"
    phase = 2

    def __init__(self, threshold: float = 0.98, ivf_threshold: int = 100_000) -> None:
        self.threshold = threshold
        self.ivf_threshold = ivf_threshold

    def run(  # type: ignore[override]
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        train_indices: Set[int],
        val_indices: Set[int],
        test_indices: Set[int],
        schema: DatasetSchema,
        embed_model: str,
        k_neighbors: int = 50,
    ) -> Dict[str, Any]:
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for Phase 2.  "
                "Install with:  pip install faiss-gpu  (or faiss-cpu)"
            )

        N, D = embeddings.shape

        # ── Build FAISS index ─────────────────────────────────────────────────
        if N < self.ivf_threshold:
            index = faiss.IndexFlatIP(D)
            index.add(embeddings)
            index_type = "IndexFlatIP"
        else:
            nlist = max(int(np.sqrt(N)), 64)
            nlist = min(nlist, N // 39)   # FAISS requirement: N ≥ 39 * nlist
            nlist = max(nlist, 1)
            quantizer = faiss.IndexFlatIP(D)
            index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(nlist, 64)
            index_type = "IndexIVFFlat"

        # ── Query every vector against its top-k neighbours ───────────────────
        k = min(k_neighbors, N)
        similarities, indices = index.search(embeddings, k)   # (N, k)

        # ── Collect near-dup pairs (i < j) above threshold ───────────────────
        # Use Union-Find to group transitively connected pairs into clusters
        parent = list(range(N))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        pair_found = set()
        for i in range(N):
            for j_pos in range(1, k):          # skip self (position 0)
                j = int(indices[i, j_pos])
                sim = float(similarities[i, j_pos])
                if sim < self.threshold:
                    break                       # sorted descending; no more
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair not in pair_found:
                    pair_found.add(pair)
                    union(i, j)

        # ── Analyse clusters ──────────────────────────────────────────────────
        cluster_map: Dict[int, list] = defaultdict(list)
        for i in range(N):
            root = find(i)
            if root in {find(p) for p in pair_found if p == i or (p[0] == i or p[1] == i)} or \
               any(find(a) == find(i) and find(b) == find(i) for a, b in pair_found if a != b):
                cluster_map[root].append(i)

        # Rebuild cluster_map cleanly from pair_found
        cluster_map = defaultdict(list)
        in_dup = set()
        for a, b in pair_found:
            in_dup.add(a)
            in_dup.add(b)
        for i in in_dup:
            cluster_map[find(i)].append(i)

        dup_groups = {root: members for root, members in cluster_map.items()
                      if len(members) > 1}

        near_dup_count = len(in_dup)

        def split_of(idx: int) -> str:
            if idx in train_indices: return "train"
            if idx in val_indices:   return "val"
            if idx in test_indices:  return "test"
            return "unknown"

        cross_split_count = 0
        for members in dup_groups.values():
            splits_present = {split_of(m) for m in members}
            if len(splits_present) > 1:
                cross_split_count += len(members)

        return {
            "total_images":               N,
            "embed_model":                embed_model,
            "threshold":                  self.threshold,
            "near_duplicate_count":       near_dup_count,
            "near_duplicate_rate":        round(near_dup_count / N, 4) if N > 0 else 0.0,
            "near_duplicate_groups":      len(dup_groups),
            "cross_split_near_duplicates": cross_split_count,
            "faiss_index_type":           index_type,
        }
