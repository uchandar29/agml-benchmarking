"""
Imbalance statistics shared by class imbalance (images per class) and
instance class imbalance (boxes per class).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def imbalance_stats(counts: Dict[str, int], num_classes: Optional[int] = None) -> Dict[str, Any]:
	"""
	counts       : {class_name: count} for the classes that occur.
	num_classes  : K used to normalise the entropy. Defaults to len(counts).
	"""
	values = list(counts.values())
	total = sum(values)
	class_count = num_classes if num_classes is not None else len(values)
	max_count = max(values)
	min_count = min(values)

	imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

	# Shannon entropy, normalised to [0, 1]
	entropy = 0.0
	for count in values:
		share = count / total
		if share > 0:
			entropy -= share * math.log(share)
	normalized_entropy = entropy / math.log(class_count) if class_count > 1 else 1.0

	return {
		"counts": counts,
		"imbalance_ratio": round(imbalance_ratio, 4),
		"normalized_entropy": round(normalized_entropy, 4),
		"most_frequent_class": max(counts, key=counts.get),
		"least_frequent_class": min(counts, key=counts.get),
	}
