"""
DatasetAdapter
==============
Loads an image-classification dataset (HuggingFace or agml) and exposes a
stable DatasetSchema regardless of how the underlying columns are named.

Source routing
--------------
  iNatAg/<name> or iNatAg-mini/<name>  → agml.data.AgMLDataLoader
  anything else                         → datasets.load_dataset (HuggingFace)

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
      5. Optionally combines multiple columns into a compound label column
         (e.g. crop_type + label → "tomato, bruised").
    """

    _COMPOUND_COL = "_compound_label"

    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        label_col: Optional[str] = None,
        image_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None,
        compound_label_cols: Optional[List[str]] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.config_name = config_name
        self._override_image_col = image_col
        self._override_label_col = label_col
        self._override_metadata_cols = metadata_cols
        self._compound_label_cols = compound_label_cols

        self._dataset: Optional[Dataset] = None
        self._schema: Optional[DatasetSchema] = None

    _AGML_PREFIXES = ("iNatAg/", "iNatAg-mini/")

    # Public API
    def load(self) -> Dataset:
        """
        Download (or restore from cache) and return the raw dataset.
        Subsequent calls return the cached object without re-downloading.

        iNatAg / iNatAg-mini datasets are loaded via agml; all others via HF.
        """
        if self._dataset is not None:
            return self._dataset

        if any(self.dataset_name.startswith(p) for p in self._AGML_PREFIXES):
            dataset = self._load_agml()
        else:
            dataset = self._load_hf()

        if self._compound_label_cols:
            dataset = self._create_compound_label(dataset)
            self._override_label_col = self._COMPOUND_COL

        self._dataset = dataset
        self._schema = self._detect_schema(dataset)

        print(f"  Loaded {len(dataset):,} examples.")
        return dataset

    def _load_hf(self) -> Dataset:
        config = self._resolve_config_name()
        tag = f" (config='{config}')" if config else ""
        print(f"Loading '{self.dataset_name}'{tag} …")

        raw = datasets.load_dataset(self.dataset_name, config)

        if isinstance(raw, datasets.DatasetDict):
            if "train" in raw:
                return raw["train"]
            first_key = next(iter(raw))
            print(
                f"  ⚠  No 'train' split found in DatasetDict; "
                f"using '{first_key}' instead."
            )
            return raw[first_key]
        return raw

    def _load_agml(self) -> Dataset:
        """Load an iNatAg / iNatAg-mini dataset via agml and convert to HF Dataset.

        Uses Dataset.from_generator() so images are written to Arrow one at a
        time — memory usage stays bounded regardless of dataset size.
        """
        try:
            import agml
        except ImportError:
            raise ImportError(
                "agml is required for iNatAg datasets.  "
                "Install with: pip install agml"
            )

        from PIL import Image as PILImage

        import os
        cache_dir = os.environ.get("AGML_DATA_DIR")
        print(f"Loading '{self.dataset_name}' via agml …")
        loader = agml.data.AgMLDataLoader(
            self.dataset_name,
            **({"dataset_path": cache_dir} if cache_dir else {}),
        )
        class_names: list[str] = list(loader.classes)

        features = datasets.Features({
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=class_names),
        })

        def _gen():
            for img_arr, label in loader:
                # img_arr is already uint8 HWC; fromarray avoids an extra copy
                yield {
                    "image": PILImage.fromarray(img_arr),
                    "label": int(label),
                }

        return Dataset.from_generator(_gen, features=features)

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

    def _create_compound_label(self, dataset: Dataset) -> Dataset:
        """
        Combine multiple columns into a single compound ClassLabel column.

        Values are joined with ", " in the order the columns are specified.
        ClassLabel (integer) columns are decoded to their string names first.
        The resulting column is named ``_compound_label`` and cast to
        ClassLabel so SplitManager can stratify on it.

        Example
        -------
        compound_label_cols=["crop_type", "label"]
        → new column values like "tomato, bruised", "pepper, healthy", …
        """
        from datasets import ClassLabel as HFClassLabel

        cols = self._compound_label_cols
        missing = [c for c in cols if c not in dataset.features] # pyright: ignore[reportOptionalIterable]
        if missing:
            raise ValueError(
                f"compound_label_cols references columns not in dataset: {missing}.  "
                f"Available columns: {list(dataset.features.keys())}"
            )

        # Decode each column to strings (handles both ClassLabel and raw strings)
        col_strings: dict[str, list[str]] = {}
        for col in cols:
            raw = list(dataset[col])
            feat = dataset.features[col]
            if isinstance(feat, HFClassLabel):
                col_strings[col] = [feat.names[i] for i in raw]
            else:
                col_strings[col] = [str(v) for v in raw]

        n = len(dataset)
        compound_values = [
            ", ".join(col_strings[col][i] for col in cols)
            for i in range(n)
        ]

        unique_labels = sorted(set(compound_values))
        label_to_int = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        int_values = [label_to_int[v] for v in compound_values]

        print(
            f"  Compound label from {cols} → "
            f"{len(unique_labels)} classes: {unique_labels}"
        )

        # Add the integer column, then update the feature schema to ClassLabel.
        # We avoid cast() because it triggers a full Arrow rewrite over all
        # columns including images, which overflows PyArrow's 2GB binary offset
        # limit on large image datasets. ClassLabel is stored as int64 internally
        # so no data transformation is needed — only the metadata changes.
        import copy
        dataset = dataset.add_column(self._COMPOUND_COL, int_values)
        new_info = copy.deepcopy(dataset._info)  # type: ignore[attr-defined]
        new_info.features[self._COMPOUND_COL] = HFClassLabel(names=unique_labels) # type: ignore
        dataset._info = new_info  # type: ignore[attr-defined]

        return dataset

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
