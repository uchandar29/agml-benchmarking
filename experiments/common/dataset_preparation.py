"""
Loads a dataset and rebuilds the exact 70 / 15 / 15 split used by the benchmark
pipeline (core SplitManager), so experiment results line up with the quality
scores row for row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from datasets import Dataset

from core.data import DatasetAdapter, DatasetSchema, SplitManager, SplitResult
from experiments.common.config_loader import DatasetEntry


@dataclass
class PreparedDataset:
	entry: DatasetEntry
	schema: DatasetSchema
	splits: SplitResult

	@property
	def class_names(self):
		return self.schema.label_names

	@property
	def split_sizes(self) -> Dict[str, int]:
		return self.splits.sizes

	@property
	def split_indices(self) -> Dict[str, np.ndarray]:
		"""Row numbers of each split in the full dataset, same numbering as the benchmark."""
		return self.splits.indices


def download_dataset(dataset_entry: DatasetEntry) -> None:
	"""Prefetch: only puts the raw dataset in the cache. No label handling and no splits."""
	DatasetAdapter(dataset_name=dataset_entry.name, config_name=dataset_entry.hf_config_name).download()


def load_dataset_with_schema(dataset_entry: DatasetEntry) -> Tuple[Dataset, DatasetSchema]:
	"""Downloads the dataset into the HF cache (or reads it from there) and resolves its columns."""
	adapter = DatasetAdapter(
		dataset_name=dataset_entry.name,
		config_name=dataset_entry.hf_config_name,
		label_col=dataset_entry.label_column,
		image_col=dataset_entry.image_column,
		compound_label_cols=dataset_entry.compound_label_columns,
	)
	full_dataset = adapter.load()
	return full_dataset, adapter.schema()


def build_split_manager(split_settings: Dict[str, Any]) -> SplitManager:
	return SplitManager(
		train_ratio=split_settings["train_ratio"],
		val_ratio=split_settings["val_ratio"],
		seed=split_settings["seed"],
	)


def prepare_dataset(
	dataset_entry: DatasetEntry,
	split_settings: Dict[str, Any],
) -> PreparedDataset:
	full_dataset, schema = load_dataset_with_schema(dataset_entry)
	splits = build_split_manager(split_settings).split(full_dataset, schema.label_col)
	return PreparedDataset(entry=dataset_entry, schema=schema, splits=splits)
