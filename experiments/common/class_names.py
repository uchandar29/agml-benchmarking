from __future__ import annotations

import re
from typing import Dict, List, Optional


def make_readable_class_name(raw_class_name: str) -> str:
	"""Turns dataset labels like 'Leaf_Rust' or 'Bottle Gourd, Downy_Mildew' into plain words."""
	readable_name = raw_class_name.replace("_", " ").replace("-", " ")
	return re.sub(r"\s+", " ", readable_name).strip()


def build_display_class_names(
	raw_class_names: List[str],
	class_name_overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
	"""Readable names in label index order. Overrides from config.yaml win over the automatic cleanup."""
	class_name_overrides = class_name_overrides or {}
	unknown_override_keys = set(class_name_overrides) - set(raw_class_names)
	if unknown_override_keys:
		raise ValueError(f"class_name_overrides has labels that are not in the dataset: {sorted(unknown_override_keys)}")

	display_names = [
		class_name_overrides.get(raw_name, make_readable_class_name(raw_name))
		for raw_name in raw_class_names
	]
	# Two labels collapsing to the same text would make predictions ambiguous
	lowered_names = [name.lower() for name in display_names]
	if len(set(lowered_names)) != len(lowered_names):
		duplicates = sorted({name for name in lowered_names if lowered_names.count(name) > 1})
		raise ValueError(f"Class names are not unique after cleanup: {duplicates}. Add class_name_overrides.")
	return display_names
