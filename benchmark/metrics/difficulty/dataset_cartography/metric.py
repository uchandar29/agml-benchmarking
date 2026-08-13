"""
DatasetCartographyMetric
========================
Classifies every training sample as easy, ambiguous, or hard-to-learn
by analysing the confidence trajectory the reference model assigns to
each sample across training epochs.

Method
------
Based on "Dataset Cartography: Mapping and Diagnosing Datasets with
Training Dynamics" (Swayamdipta et al., 2020).

For each training sample we compute:

  confidence  -- mean of the per-epoch max softmax probability assigned
                 to the sample.  High confidence = model learned the
                 sample quickly and reliably.

  variability -- std deviation of the per-epoch max softmax probability.
                 High variability = the model is uncertain / oscillates
                 on this sample throughout training.

Classification thresholds
--------------------------
  easy       -- confidence >= 0.7  AND  variability <= 0.1
  hard       -- confidence <  0.3  AND  variability <= 0.1
  ambiguous  -- everything else (high variability or mid-confidence)

These thresholds are configurable via constructor arguments.

Output keys
-----------
  n_easy, n_ambiguous, n_hard            -- sample counts per category
  pct_easy, pct_ambiguous, pct_hard      -- percentages
  mean_confidence                         -- mean confidence across all samples
  mean_variability                        -- mean variability across all samples
  easy_threshold, hard_threshold,
  variability_threshold                   -- thresholds used
  n_epochs                               -- number of training epochs used
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from benchmark.metrics.base import BaseMetric
from benchmark.core.dataset_adapter import DatasetSchema


class DatasetCartographyMetric(BaseMetric):
	name = "dataset_cartography"
	phase = 3

	def __init__(
		self,
		easy_threshold: float = 0.7,
		hard_threshold: float = 0.3,
		variability_threshold: float = 0.1,
	) -> None:
		"""
		Parameters
		----------
		easy_threshold       : Samples with confidence >= this are 'easy'.
		hard_threshold       : Samples with confidence <  this are 'hard'.
		variability_threshold: Samples with variability > this are 'ambiguous'
		                       regardless of confidence.
		"""
		self.easy_threshold = easy_threshold
		self.hard_threshold = hard_threshold
		self.variability_threshold = variability_threshold

	def run(
		self,
		conf_history: np.ndarray,
		orig_idx_order: np.ndarray,
		schema: DatasetSchema,
		**kwargs,
	) -> Dict[str, Any]:
		"""
		Parameters
		----------
		conf_history    : float32 (N_train, n_epochs) from ReferenceModelTrainer.
		orig_idx_order  : int64 (N_train,) mapping row → _orig_idx.
		schema          : DatasetSchema (used for num_classes context).

		Returns
		-------
		JSON-serialisable dict with counts, percentages, and per-sample map.
		"""
		N, n_epochs = conf_history.shape

		confidence = conf_history.mean(axis=1)
		variability = conf_history.std(axis=1)

		categories = self._classify(confidence, variability)

		counts = {"easy": 0, "ambiguous": 0, "hard": 0}
		for cat in categories:
			counts[cat] += 1

		return {
			"n_easy":                counts["easy"],
			"n_ambiguous":           counts["ambiguous"],
			"n_hard":                counts["hard"],
			"pct_easy":              round(counts["easy"] / N * 100, 2),
			"pct_ambiguous":         round(counts["ambiguous"] / N * 100, 2),
			"pct_hard":              round(counts["hard"] / N * 100, 2),
			"mean_confidence":       round(float(confidence.mean()), 4),
			"mean_variability":      round(float(variability.mean()), 4),
			"easy_threshold":        self.easy_threshold,
			"hard_threshold":        self.hard_threshold,
			"variability_threshold": self.variability_threshold,
			"n_epochs":              n_epochs,
		}

	def _classify(self, confidence: np.ndarray, variability: np.ndarray) -> List[str]:
		categories = []
		for conf, var in zip(confidence, variability):
			if var > self.variability_threshold:
				categories.append("ambiguous")
			elif conf >= self.easy_threshold:
				categories.append("easy")
			elif conf < self.hard_threshold:
				categories.append("hard")
			else:
				categories.append("ambiguous")
		return categories
