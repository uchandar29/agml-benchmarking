"""
Loads a dataset and rebuilds the exact 70 / 15 / 15 split used by the benchmark
pipeline, so experiment results line up with the quality scores row for row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from datasets import Dataset

from benchmark.core.dataset_adapter import DatasetAdapter, DatasetSchema
from benchmark.core.split_manager import SplitManager, SplitResult
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
		return {
			"train": len(self.splits.train),
			"val": len(self.splits.val),
			"test": len(self.splits.test),
		}


def prepare_dataset(
	dataset_entry: DatasetEntry,
	split_settings: Dict[str, Any],
	split_output_dir: str,
) -> PreparedDataset:
	adapter = DatasetAdapter(
		dataset_name=dataset_entry.name,
		config_name=dataset_entry.hf_config_name,
		label_col=dataset_entry.label_column,
		image_col=dataset_entry.image_column,
		compound_label_cols=dataset_entry.compound_label_columns,
	)
	full_dataset = adapter.load()
	schema = adapter.schema()

	split_manager = SplitManager(
		train_ratio=split_settings["train_ratio"],
		val_ratio=split_settings["val_ratio"],
		seed=split_settings["seed"],
	)
	splits = split_manager.split(full_dataset, schema.label_col, split_output_dir)
	return PreparedDataset(entry=dataset_entry, schema=schema, splits=splits)


class ImageClassificationDataset:
	"""Torch-style dataset over a HuggingFace split. Returns (image_tensor, label)."""

	def __init__(
		self,
		hf_split: Dataset,
		image_column: str,
		label_column: str,
		image_transform: Callable,
	) -> None:
		self.hf_split = hf_split
		self.image_column = image_column
		self.label_column = label_column
		self.image_transform = image_transform

	def __len__(self) -> int:
		return len(self.hf_split)

	def __getitem__(self, row_index: int):
		record = self.hf_split[row_index]
		image = record[self.image_column]
		if image.mode != "RGB":
			image = image.convert("RGB")
		return self.image_transform(image), int(record[self.label_column])
