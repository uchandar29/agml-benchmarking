"""
SplitManager
============
Creates a deterministic, stratified 70 / 15 / 15 split from any single-split
HuggingFace Dataset. Used by the benchmark and every experiment, so they all
see the same train, val and test rows.

Design notes
------------
• Pre-existing HF train/val/test splits are never used. Every dataset is split
  from scratch with the same ratios and seed.

• Split indices come from sklearn's StratifiedShuffleSplit. Each split is a
  lazy HF Dataset.select(), so no images are copied. Copying the images of a
  split (for example by adding a column to it) can overflow Arrow's 2 GB
  per-column limit on datasets with large images.

• The row numbers of each split in the full dataset are kept as arrays in
  SplitResult.indices instead of as a column on the dataset.

• If stratified splitting fails (e.g. a class has only 1 sample), we fall
  back to a random split and log a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from datasets import Dataset

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitResult:
	"""The full dataset, its three splits, and each split's row numbers in the full dataset."""
	full: Dataset
	train: Dataset
	val: Dataset
	test: Dataset
	indices: Dict[str, np.ndarray]

	@property
	def sizes(self) -> Dict[str, int]:
		return {split_name: len(self.indices[split_name]) for split_name in SPLIT_NAMES}

	def index_sets(self) -> Dict[str, set]:
		"""Row numbers of each split as Python sets, for fast membership checks."""
		return {split_name: set(self.indices[split_name].tolist()) for split_name in SPLIT_NAMES}


class SplitManager:
	"""
	Stratified 70 / 15 / 15 split with deterministic seeding.

	Usage::
		splits = SplitManager().split(dataset, label_col="label")
		splits.train, splits.val, splits.test, splits.full, splits.indices["test"]
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

	def split(self, dataset: Dataset, label_col: str) -> SplitResult:
		labels = np.array(dataset[label_col])
		train_indices, val_indices, test_indices = self.compute_indices(labels)
		indices = {"train": train_indices, "val": val_indices, "test": test_indices}

		splits = SplitResult(
			full=dataset,
			train=dataset.select(train_indices),
			val=dataset.select(val_indices),
			test=dataset.select(test_indices),
			indices=indices,
		)
		print(
			f"Split complete  →  train={len(train_indices):,}  |  val={len(val_indices):,}  |  "
			f"test={len(test_indices):,}  (seed={self.seed})"
		)
		return splits

	def compute_indices(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Compute stratified train / val / test row indices using sklearn.
		Falls back to random if any class is too small for stratification.
		"""
		from sklearn.model_selection import StratifiedShuffleSplit

		labels = np.asarray(labels)
		N = len(labels)

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
			temp_idx = perm[n_train:]

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
			val_rel = perm[:n_val]
			test_rel = perm[n_val:]

		return (
			np.sort(train_idx),
			np.sort(temp_idx[val_rel]),
			np.sort(temp_idx[test_rel]),
		)
