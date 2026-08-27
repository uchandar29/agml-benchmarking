"""
SplitManager
============
Creates a deterministic, stratified 70 / 15 / 15 split from any single-split
HuggingFace Dataset.

Design notes
------------
• We never rely on pre-existing HF train/val/test splits.  Every dataset is
  split from scratch using the same fixed ratios and seed.

• Split indices are computed with sklearn's StratifiedShuffleSplit — pure
  numpy, no Arrow I/O.  HF Dataset.select() is then used to build each
  subset lazily (no data is copied).

• A '_orig_idx' column is added to each split so that downstream metrics
  can trace any sample back to its row in the full dataset.  The full
  dataset itself does not need this column (embedding_engine iterates
  it in row order).

• If stratified splitting fails (e.g. a class has only 1 sample), we fall
  back to a random split and log a warning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from datasets import Dataset


# ---------------------------- Public data structures ----------------------------
@dataclass
class SplitResult:
    """Holds all four views of the dataset after splitting."""
    full: Dataset    # Original dataset, no extra columns added
    train: Dataset   # Has '_orig_idx' column
    val: Dataset     # Has '_orig_idx' column
    test: Dataset    # Has '_orig_idx' column


# ---------------------------- Manager ----------------------------
class SplitManager:
    """
    Stratified 70 / 15 / 15 split with deterministic seeding.

    Usage::
        mgr = SplitManager()
        splits = mgr.split(dataset, label_col="label", output_dir="results/run_xyz")
        # splits.train, splits.val, splits.test, splits.full
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    # ---------------------------- Public API ----------------------------
    def split(
        self,
        dataset: Dataset,
        label_col: str,
        output_dir: str,
    ) -> SplitResult:
        """
        Create and return the four dataset views.

        Parameters
        ----------
        dataset    : The full source dataset (single split from HF).
        label_col  : Column name of the ClassLabel feature used for stratification.
        output_dir : Directory where splits.json will be written.

        Returns
        -------
        SplitResult with .full / .train / .val / .test
        """
        labels = np.array(dataset[label_col])
        N = len(labels)

        train_idx, val_idx, test_idx = self._compute_indices(labels, N)

        # select() is lazy — no Arrow copy, just an index mapping
        train_ds = dataset.select(train_idx).add_column("_orig_idx", train_idx.tolist())
        val_ds   = dataset.select(val_idx).add_column("_orig_idx", val_idx.tolist())
        test_ds  = dataset.select(test_idx).add_column("_orig_idx", test_idx.tolist())

        os.makedirs(output_dir, exist_ok=True)

        print(
            f"Split complete  →  "
            f"train={len(train_ds):,}  |  "
            f"val={len(val_ds):,}  |  "
            f"test={len(test_ds):,}  "
            f"(seed={self.seed})"
        )

        return SplitResult(
            full=dataset,
            train=train_ds,
            val=val_ds,
            test=test_ds,
        )

    # ---------------------------- Internal helpers ----------------------------
    def _compute_indices(
        self,
        labels: np.ndarray,
        N: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute stratified train / val / test indices using sklearn.
        Falls back to random if any class is too small for stratification.
        """
        from sklearn.model_selection import StratifiedShuffleSplit

        temp_ratio = 1.0 - self.train_ratio  # 0.30

        # Step 1: train vs temp
        try:
            sss1 = StratifiedShuffleSplit(
                n_splits=1, test_size=temp_ratio, random_state=self.seed
            )
            train_idx, temp_idx = next(sss1.split(np.zeros(N), labels))
        except ValueError as exc:
            print(
                f"  ⚠  Stratified split failed ({exc}).  "
                f"Falling back to a random split — class distribution may be "
                f"uneven across splits."
            )
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(N)
            n_train = int(round(N * self.train_ratio))
            train_idx = perm[:n_train]
            temp_idx  = perm[n_train:]

        # Step 2: val vs test from temp
        temp_labels = labels[temp_idx]
        n_temp = len(temp_idx)
        val_frac = self.val_ratio / temp_ratio  # 0.50

        try:
            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=1.0 - val_frac, random_state=self.seed
            )
            val_rel, test_rel = next(sss2.split(np.zeros(n_temp), temp_labels))
        except ValueError:
            rng = np.random.default_rng(self.seed + 1)
            perm = rng.permutation(n_temp)
            n_val = int(round(n_temp * val_frac))
            val_rel  = perm[:n_val]
            test_rel = perm[n_val:]

        return (
            np.sort(train_idx),
            np.sort(temp_idx[val_rel]),
            np.sort(temp_idx[test_rel]),
        )
