"""
ClassConfusabilityMetric
========================
Evaluates the trained reference model on the held-out test split and
surfaces which pairs of classes the model most frequently confuses.

This metric answers a practical question: "Which classes in this dataset
are visually hard to tell apart?"  High confusion between two classes is
a signal to either collect more discriminative examples or to reconsider
whether those classes should be merged or split more carefully.

Method
------
1. Run the trained ResNet on the test split and collect predictions.
2. Build a normalised confusion matrix (rows = true class, cols = predicted).
3. Identify the top-N off-diagonal entries — these are the most confused pairs.

Output keys
-----------
  accuracy              -- overall accuracy on the test split
  per_class_accuracy    -- {class_name: accuracy} for each class
  confusion_matrix      -- list[list[float]], normalised by true class count
  top_confused_pairs    -- list of {true_class, predicted_as, confusion_rate}
                           sorted descending by confusion_rate
  n_top_pairs           -- number of confused pairs returned
  n_test_samples        -- total test samples evaluated
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset

from benchmark.core.dataset_adapter import DatasetSchema
from benchmark.metrics.base import BaseMetric


class ClassConfusabilityMetric(BaseMetric):
	name = "class_confusability"
	phase = 3

	def __init__(self, n_top_pairs: int = 10) -> None:
		"""
		Parameters
		----------
		n_top_pairs : Number of most-confused class pairs to include in the output.
		"""
		self.n_top_pairs = n_top_pairs

	def run(
		self,
		model,
		test_dataset: Dataset,
		schema: DatasetSchema,
		batch_size: int = 64,
		device: Optional[str] = None,
		**kwargs,
	) -> Dict[str, Any]:
		"""
		Parameters
		----------
		model        : Trained nn.Module from ReferenceModelTrainer (eval mode, CPU).
		test_dataset : HF Dataset with _orig_idx column (splits.test).
		schema       : Resolved DatasetSchema.
		batch_size   : Images per forward pass.
		device       : Target device.  Auto-detected when None.

		Returns
		-------
		JSON-serialisable dict.
		"""
		import torch
		from torchvision import transforms
		from torch.utils.data import DataLoader

		from benchmark.core.reference_trainer import _HFImageDataset

		resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

		transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
			                     std=[0.229, 0.224, 0.225]),
		])

		hf_ds = _HFImageDataset(
			test_dataset, schema.image_col, schema.label_col, transform
		)
		loader = DataLoader(hf_ds, batch_size=batch_size, shuffle=False, num_workers=0)

		model = model.to(resolved_device)
		model.eval()

		all_preds: List[int] = []
		all_labels: List[int] = []

		with torch.no_grad():
			for images, labels, _ in loader:
				logits = model(images.to(resolved_device))
				preds = logits.argmax(dim=1)
				all_preds.extend(preds.cpu().tolist())
				all_labels.extend(labels.tolist())

		model.to("cpu")

		n_classes = schema.num_classes
		label_names = schema.label_names

		cm = np.zeros((n_classes, n_classes), dtype=np.int64)
		for true, pred in zip(all_labels, all_preds):
			cm[true, pred] += 1

		# Normalise each row by the true class count
		row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
		cm_norm = (cm / row_sums).round(4)

		# Per-class accuracy = diagonal of normalised confusion matrix
		per_class_acc = {
			label_names[i]: round(float(cm_norm[i, i]), 4)
			for i in range(n_classes)
		}

		# Overall accuracy
		accuracy = round(float(np.diag(cm).sum() / max(len(all_labels), 1)), 4)

		# Top confused pairs — highest off-diagonal normalised values
		confused_pairs = []
		for true_i in range(n_classes):
			for pred_j in range(n_classes):
				if true_i == pred_j:
					continue
				rate = float(cm_norm[true_i, pred_j])
				if rate > 0:
					confused_pairs.append({
						"true_class":      label_names[true_i],
						"predicted_as":    label_names[pred_j],
						"confusion_rate":  round(rate, 4),
					})

		confused_pairs.sort(key=lambda x: x["confusion_rate"], reverse=True)
		top_pairs = confused_pairs[: self.n_top_pairs]

		return {
			"accuracy":           accuracy,
			"per_class_accuracy": per_class_acc,
			"confusion_matrix":   cm_norm.tolist(),
			"top_confused_pairs": top_pairs,
			"n_top_pairs":        len(top_pairs),
			"n_test_samples":     len(all_labels),
		}
