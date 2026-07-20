"""
SplitManager
============
Creates a deterministic, stratified 70 / 15 / 15 split from any single-split
HuggingFace Dataset.

Design notes
------------
• We never rely on pre-existing HF train/val/test splits.  Every dataset is
  split from scratch using the same fixed ratios and seed.

• A synthetic '_orig_idx' column (0 … N-1) is added to the dataset before
  splitting.  All returned Dataset objects carry this column so that any
  downstream metric can map a sample back to its position in the full dataset
  and determine which split it belongs to — without relying on HF internals.

• Split indices are written to splits.json in the run output directory so
  that every pipeline run is fully reproducible.

• If stratified splitting fails for a class (e.g. the class has only 1
  sample), we fall back to a non-stratified split and log a warning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset


# ---------------------------- Public data structures ----------------------------
@dataclass
class SplitResult:
    """Holds all four views of the dataset after splitting."""
    full: Dataset    # Complete dataset with '_orig_idx' column
    train: Dataset
    val: Dataset
    test: Dataset


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
        # Attach original index so every downstream metric can trace samples
        dataset_with_idx = dataset.add_column(
            "_orig_idx", list(range(len(dataset)))
        )

        # ---------------------------- Step 1: train (70%) vs temp (30%) ----------------------------
        temp_ratio = 1.0 - self.train_ratio   # 0.30
        split1 = self._stratified_split(
            dataset_with_idx,
            test_size=temp_ratio,
            label_col=label_col,
        )
        train_ds = split1["train"]
        temp_ds  = split1["test"]

        # ---------------------------- Step 2: val (15%) vs test (15%) from the temp 30% ----------------------------
        # Within the 30% temp block, val takes 50% → 15% of total
        val_fraction_of_temp = self.val_ratio / temp_ratio   # 0.50
        split2 = self._stratified_split(
            temp_ds,
            test_size=1.0 - val_fraction_of_temp,
            label_col=label_col,
        )
        val_ds  = split2["train"]
        test_ds = split2["test"]

        # ---------------------------- Persist indices ----------------------------
        os.makedirs(output_dir, exist_ok=True)
        self._save_indices(train_ds, val_ds, test_ds, output_dir)

        print(
            f"Split complete  →  "
            f"train={len(train_ds):,}  |  "
            f"val={len(val_ds):,}  |  "
            f"test={len(test_ds):,}  "
            f"(seed={self.seed})"
        )

        return SplitResult(
            full=dataset_with_idx,
            train=train_ds,
            val=val_ds,
            test=test_ds,
        )

    # ---------------------------- Internal helpers ----------------------------
    def _stratified_split(
        self,
        dataset: Dataset,
        test_size: float,
        label_col: str,
    ) -> dict:
        """
        Attempt a stratified split; fall back to a random split if any class
        has too few examples for stratification to succeed.
        """
        try:
            return dataset.train_test_split(
                test_size=test_size,
                stratify_by_column=label_col,
                seed=self.seed,
            )
        except Exception as exc:
            print(
                f"  ⚠  Stratified split failed ({exc}).  "
                f"Falling back to a random split — class distribution may be "
                f"uneven across splits."
            )
            return dataset.train_test_split(
                test_size=test_size,
                seed=self.seed,
            )

    def _save_indices(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset,
        output_dir: str,
    ) -> None:
        """Write split membership to splits.json for reproducibility."""
        
        payload = {
            "seed": self.seed,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "train_indices": list(train_ds["_orig_idx"]),
            "val_indices":   list(val_ds["_orig_idx"]),
            "test_indices":  list(test_ds["_orig_idx"]),
        }
        path = os.path.join(output_dir, "splits.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=4)
        print(f"  Split indices → {path}")
