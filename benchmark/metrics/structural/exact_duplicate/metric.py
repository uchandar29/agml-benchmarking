"""
ExactDuplicateMetric
====================
Detects exact duplicate images by hashing decoded pixel data (MD5).

Why MD5 on decoded pixels, not on raw bytes
--------------------------------------------
Hashing the compressed bytes (JPEG/PNG) would miss duplicates that were
re-encoded at different quality settings or with different metadata (EXIF).
Hashing PIL's tobytes() output — raw, uncompressed RGB pixels — catches true
pixel-level duplicates regardless of how the source file was encoded.

Cross-split duplicates
----------------------
A duplicate pair that straddles the train / val or train / test boundary is a
data-leakage risk: the model may have seen (a pixel-identical copy of) a test
image during training.  These are reported separately so they can be
prioritised for removal or re-splitting.

Computed on the full dataset (all splits combined).

Output keys
-----------
total_images          : int
exact_duplicate_count : int   — images that are copies of at least one other
exact_duplicate_rate  : float — duplicate_count / total_images
duplicate_groups      : int   — number of distinct groups with ≥ 2 copies
cross_split_duplicates: int   — images in groups that span ≥ 2 splits
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, Set

from datasets import Dataset
from tqdm import tqdm

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class ExactDuplicateMetric(BaseMetric):
    name = "exact_duplicate"
    phase = 1

    def run(  # type: ignore[override]
        self,
        full_dataset: Dataset,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        schema: DatasetSchema,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        image_col = schema.image_col

        # Build split membership sets keyed by original index
        train_idx: Set[int] = set(list(train_dataset["_orig_idx"]))
        val_idx:   Set[int] = set(list(val_dataset["_orig_idx"]))
        test_idx:  Set[int] = set(list(test_dataset["_orig_idx"]))

        def split_of(orig_idx: int) -> str:
            if orig_idx in train_idx:
                return "train"
            if orig_idx in val_idx:
                return "val"
            if orig_idx in test_idx:
                return "test"
            return "unknown"

        # ---------------------------- Hash every image ----------------------------
        # hash → list of original indices that produced that hash
        hash_to_orig: Dict[str, list] = defaultdict(list)

        n = len(full_dataset)
        for start in tqdm(range(0, n, batch_size), desc="Hashing images (exact)"):
            batch = full_dataset[start: start + batch_size]
            images       = list(batch[image_col])
            orig_indices = list(batch["_orig_idx"])

            for img, orig_idx in zip(images, orig_indices):
                digest = hashlib.md5(img.tobytes()).hexdigest()
                hash_to_orig[digest].append(orig_idx)

        # ---------------------------- Analyse results ----------------------------
        dup_groups = {
            h: idxs
            for h, idxs in hash_to_orig.items()
            if len(idxs) > 1
        }

        # Count images that are duplicates of at least one other image.
        # For a group of size N, every member is a duplicate (N images, not N-1).
        exact_duplicate_count = sum(len(idxs) for idxs in dup_groups.values())

        # Cross-split: groups where at least two different splits are present
        cross_split_count = 0
        for idxs in dup_groups.values():
            splits_present = {split_of(i) for i in idxs}
            if len(splits_present) > 1:
                cross_split_count += len(idxs)

        return {
            "total_images": n,
            "exact_duplicate_count": exact_duplicate_count,
            "exact_duplicate_rate": round(exact_duplicate_count / n, 4) if n > 0 else 0.0,
            "duplicate_groups": len(dup_groups),
            "cross_split_duplicates": cross_split_count,
        }
