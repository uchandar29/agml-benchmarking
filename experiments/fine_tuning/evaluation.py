from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def compute_classification_metrics(
	true_labels: Sequence[int],
	predicted_labels: Sequence[int],
	class_names: List[str],
) -> Dict[str, Any]:
	from sklearn.metrics import accuracy_score, f1_score

	true_labels = np.asarray(true_labels)
	predicted_labels = np.asarray(predicted_labels)
	class_indices = list(range(len(class_names)))

	per_class_f1 = f1_score(
		true_labels, predicted_labels, labels=class_indices, average=None, zero_division=0
	)
	return {
		"macro_f1": round(float(f1_score(
			true_labels, predicted_labels, labels=class_indices, average="macro", zero_division=0
		)), 4),
		"weighted_f1": round(float(f1_score(
			true_labels, predicted_labels, labels=class_indices, average="weighted", zero_division=0
		)), 4),
		"accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
		"number_of_test_samples": int(len(true_labels)),
		"per_class_f1": {
			class_name: round(float(score), 4)
			for class_name, score in zip(class_names, per_class_f1)
		},
	}
