"""
DatasetAdapter
==============
Loads a HuggingFace image-classification dataset and exposes a stable
DatasetSchema regardless of how the underlying columns are named.

Schema detection rules (applied in order; any rule can be overridden via
constructor arguments):
  image_col   → first column whose dtype is datasets.Image()
  label_col   → first column whose dtype is datasets.ClassLabel(...)
  metadata_cols → every remaining column (may be an empty list)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import datasets
from datasets import Dataset


# Public data structures
@dataclass
class DatasetSchema:
    """Resolved column mapping and class metadata for one dataset."""
    image_col: str
    label_col: str
    label_names: List[str]   # ordered list, index == integer label value
    num_classes: int
    metadata_cols: List[str]  # empty when no auxiliary columns exist


# Adapter
class DatasetAdapter:
    """
    Thin wrapper around datasets.load_dataset() that:
      1. Resolves multi-config datasets to a single config.
      2. Collapses DatasetDicts to their 'train' split (we always build our
         own splits via SplitManager — pre-existing splits are ignored).
      3. Infers image_col, label_col, and metadata_cols from feature dtypes.
      4. Allows caller to override any auto-detected value.
    """

    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        label_col: Optional[str] = None,
        image_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.config_name = config_name
        self._override_image_col = image_col
        self._override_label_col = label_col
        self._override_metadata_cols = metadata_cols

        self._dataset: Optional[Dataset] = None
        self._schema: Optional[DatasetSchema] = None

    # Public API
    def load(self) -> Dataset:
        """
        Download (or restore from HF cache) and return the raw dataset.
        Subsequent calls return the cached object without re-downloading.
        """
        if self._dataset is not None:
            return self._dataset

        config = self._resolve_config_name()
        tag = f" (config='{config}')" if config else ""
        print(f"Loading '{self.dataset_name}'{tag} …")

        raw = datasets.load_dataset(self.dataset_name, config)

        if isinstance(raw, datasets.DatasetDict):
            if "train" in raw:
                dataset = raw["train"]
            else:
                first_key = next(iter(raw))
                print(
                    f"  ⚠  No 'train' split found in DatasetDict; "
                    f"using '{first_key}' instead."
                )
                dataset = raw[first_key]
        else:
            dataset = raw

        self._dataset = dataset
        self._schema = self._detect_schema(dataset)

        print(f"  Loaded {len(dataset):,} examples.")
        return dataset

    def schema(self) -> DatasetSchema:
        """Return the resolved schema.  Raises if load() has not been called."""
        if self._schema is None:
            raise RuntimeError(
                "DatasetAdapter.schema() called before load().  "
                "Call load() first."
            )
        return self._schema

    # ----------------- Internal helpers -----------------
    def _resolve_config_name(self) -> Optional[str]:
        """
        Return the config name to pass to load_dataset().

        If the caller supplied config_name, use it directly.
        If the dataset has only one config (or a config named 'default'),
        return None (load_dataset handles the default automatically).
        If the dataset has multiple named configs, use the first one listed
        and print a notice so the caller knows.
        """
        if self.config_name is not None:
            return self.config_name

        try:
            builder = datasets.load_dataset_builder(self.dataset_name)
            configs = builder.builder_configs

            # Single config → no name needed
            if len(configs) <= 1:
                return None

            # Find HF-marked default
            for cfg in configs:
                if getattr(cfg, "name", None) in (None, "default", ""):
                    return None

            # Multiple named configs with no explicit default → use first
            chosen = configs[0].name
            all_names = [c.name for c in configs]
            print(
                f"  ℹ  Multiple configs detected {all_names}.  "
                f"Defaulting to '{chosen}'.  "
                f"Override with config_name= if needed."
            )
            return chosen

        except Exception:
            # If inspection fails, let load_dataset sort it out
            return None

    def _detect_schema(self, dataset: Dataset) -> DatasetSchema:
        features = dataset.features

        image_col = self._override_image_col
        label_col = self._override_label_col

        # Walk columns in declaration order for determinism
        for col, dtype in features.items():
            if image_col is None and isinstance(dtype, datasets.Image):
                image_col = col
            elif label_col is None and isinstance(dtype, datasets.ClassLabel):
                label_col = col

        if image_col is None:
            raise ValueError(
                f"No Image column found in '{self.dataset_name}'.  "
                f"Available columns: {list(features.keys())}.  "
                f"Pass image_col= explicitly."
            )
        if label_col is None:
            raise ValueError(
                f"No ClassLabel column found in '{self.dataset_name}'.  "
                f"Available columns: {list(features.keys())}.  "
                f"Pass label_col= explicitly."
            )

        label_names: List[str] = features[label_col].names

        if self._override_metadata_cols is not None:
            metadata_cols = self._override_metadata_cols
        else:
            metadata_cols = [
                col for col in features
                if col not in (image_col, label_col)
            ]

        return DatasetSchema(
            image_col=image_col,
            label_col=label_col,
            label_names=label_names,
            num_classes=len(label_names),
            metadata_cols=metadata_cols,
        )
