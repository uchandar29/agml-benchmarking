"""
LabelNoiseMetric
================
Estimates the percentage of mislabeled samples in the dataset using
confident learning (Northcutt et al., 2021 — https://arxiv.org/abs/1911.00068).

Method
------
1. Run stratified k-fold cross-validation on the DINOv2 embeddings using
   logistic regression to obtain out-of-fold (OOF) predicted probabilities
   for every sample in the dataset.
2. Pass the OOF probabilities and ground-truth labels to cleanlab's
   find_label_issues(), which identifies samples where the model's confident
   prediction disagrees with the assigned label.

Why this combination works well
--------------------------------
DINOv2 embeddings are rich enough that logistic regression achieves strong
OOF accuracy without any fine-tuning.  Cleanlab's confident learning is
most reliable when OOF probabilities span the full dataset (all N samples),
which k-fold CV guarantees.  This is the exact setup recommended in the
cleanlab documentation for pre-computed feature vectors.

Output keys
-----------
  estimated_noise_rate      -- fraction of samples likely mislabeled (0–1)
  n_noisy_samples           -- absolute count of flagged samples
  n_total_samples           -- total samples evaluated
  flagged_orig_indices      -- list of _orig_idx values for flagged samples,
                               sorted by cleanlab's self-confidence score
                               (most suspicious first)
  per_class_noise_counts    -- {class_name: n_flagged} breakdown
  cv_folds                  -- number of CV folds used
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from benchmark.core.dataset_adapter import DatasetSchema
from benchmark.metrics.base import BaseMetric


class LabelNoiseMetric(BaseMetric):
	name = "label_noise"
	phase = 3

	def __init__(self, cv_folds: int = 5) -> None:
		"""
		Parameters
		----------
		cv_folds : Number of stratified CV folds.  5 is the standard choice.
		"""
		self.cv_folds = cv_folds

	def run(
		self,
		embeddings: np.ndarray,
		labels: np.ndarray,
		orig_idx_order: np.ndarray,
		schema: DatasetSchema,
		**kwargs,
	) -> Dict[str, Any]:
		"""
		Parameters
		----------
		embeddings     : float32 (N, D) L2-normalised DINOv2 embeddings for the
		                 full dataset (from EmbeddingEngine).
		labels         : int64 (N,) ground-truth class indices.
		orig_idx_order : int64 (N,) mapping row → _orig_idx (0 … N-1 for the
		                 full dataset, so this is just np.arange(N)).
		schema         : Resolved DatasetSchema.

		Returns
		-------
		JSON-serialisable dict.
		"""
		try:
			from cleanlab.filter import find_label_issues
		except ImportError:
			raise ImportError(
				"LabelNoiseMetric requires cleanlab.  "
				"Install with:  pip install cleanlab"
			)

		from sklearn.linear_model import LogisticRegression
		from sklearn.model_selection import StratifiedKFold

		N = len(embeddings)
		n_classes = schema.num_classes
		label_names = schema.label_names

		print(f"  Label noise: {self.cv_folds}-fold CV on {N} samples …")

		oof_probs = np.zeros((N, n_classes), dtype=np.float64)

		skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
		for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
			clf = LogisticRegression(
				max_iter=1000,
				C=1.0,
				solver="lbfgs",
				multi_class="multinomial",
				random_state=42,
				n_jobs=-1,
			)
			clf.fit(embeddings[train_idx], labels[train_idx])
			oof_probs[val_idx] = clf.predict_proba(embeddings[val_idx])

		# Cleanlab expects probabilities that sum to 1 per row — ensure this
		row_sums = oof_probs.sum(axis=1, keepdims=True).clip(min=1e-12)
		oof_probs /= row_sums

		print("  Running cleanlab confident learning …")
		label_issues = find_label_issues(
			labels=labels,
			pred_probs=oof_probs,
			return_indices_ranked_by="self_confidence",
		)

		# label_issues is an array of row indices into embeddings/labels
		flagged_orig = orig_idx_order[label_issues].tolist()
		flagged_labels = labels[label_issues].tolist()

		# Per-class noise breakdown
		per_class: Dict[str, int] = {name: 0 for name in label_names}
		for lbl in flagged_labels:
			per_class[label_names[int(lbl)]] += 1

		noise_rate = round(len(label_issues) / max(N, 1), 4)

		print(
			f"  Estimated noise rate: {noise_rate:.1%}  "
			f"({len(label_issues)} of {N} samples flagged)"
		)

		return {
			"estimated_noise_rate":   noise_rate,
			"n_noisy_samples":        int(len(label_issues)),
			"n_total_samples":        int(N),
			"flagged_orig_indices":   [int(i) for i in flagged_orig],
			"per_class_noise_counts": per_class,
			"cv_folds":               self.cv_folds,
		}
