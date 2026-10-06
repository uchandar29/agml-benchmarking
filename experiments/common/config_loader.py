"""Reads the shared experiments/config.yaml used by Experiment 1 and Experiment 2."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@dataclass
class DatasetEntry:
	name: str
	label_column: Optional[str] = None
	image_column: Optional[str] = None
	hf_config_name: Optional[str] = None
	compound_label_columns: Optional[List[str]] = None
	class_name_overrides: Optional[Dict[str, str]] = None

	@property
	def short_name(self) -> str:
		return self.name.split("/")[-1]


def load_experiment_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
	resolved_path = Path(os.path.expandvars(os.path.expanduser(str(config_path or DEFAULT_CONFIG_PATH))))
	with open(resolved_path) as config_file:
		experiment_config = yaml.safe_load(config_file)
	experiment_config["shared"]["output_root"] = os.path.expandvars(
		os.path.expanduser(experiment_config["shared"]["output_root"])
	)
	return experiment_config


def parse_dataset_entries(
	experiment_config: Dict[str, Any],
	selected_dataset_names: Optional[List[str]] = None,
) -> List[DatasetEntry]:
	"""Merges dataset_defaults into every dataset entry. Selection accepts full or short names."""
	dataset_defaults = experiment_config.get("dataset_defaults") or {}
	dataset_entries = []
	for raw_entry in experiment_config["datasets"]:
		merged_entry = {**dataset_defaults, **raw_entry}
		dataset_entries.append(DatasetEntry(
			name=merged_entry["name"],
			label_column=merged_entry.get("label_column") or None,
			image_column=merged_entry.get("image_column") or None,
			hf_config_name=merged_entry.get("hf_config_name") or None,
			compound_label_columns=merged_entry.get("compound_label_columns") or None,
			class_name_overrides=merged_entry.get("class_name_overrides") or None,
		))

	if not selected_dataset_names:
		return dataset_entries

	wanted_names = set(selected_dataset_names)
	selected_entries = [
		entry for entry in dataset_entries
		if entry.name in wanted_names or entry.short_name in wanted_names
	]
	known_names = {entry.name for entry in dataset_entries} | {entry.short_name for entry in dataset_entries}
	unknown_names = wanted_names - known_names
	if unknown_names:
		raise ValueError(f"Datasets not found in config.yaml: {sorted(unknown_names)}")
	return selected_entries
