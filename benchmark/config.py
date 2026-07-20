from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    """
    Infrastructure and tuning settings for the AgML benchmark pipeline.

    This config captures settings that are **stable across datasets** — compute
    resource limits, split reproducibility, output location, and Phase-2/3
    model choices.  Dataset-specific fields (dataset_name, label_col, etc.) are
    passed at runtime, not stored here.

    Config discovery (highest → lowest precedence)
    -----------------------------------------------
    1. ``PipelineConfig.from_json(path)`` — explicit path
    2. ``$AGML_CONFIG`` environment variable pointing to a JSON file
    3. ``benchmark/config.json`` in the current working directory
    4. Built-in defaults (no file required)

    Use :py:meth:`load` to auto-discover::

        cfg = PipelineConfig.load()          # auto-discovers
        cfg = PipelineConfig.load("benchmark/configs/farm.json")  # explicit

    Save the exact settings used in a run for reproducibility::

        cfg.to_json(os.path.join(run_dir, "config_used.json"))
    """

    # ── Splits ─────────────────────────────────────────────────────────────────
    split_seed: int = 42
    """Random seed for the 70 / 15 / 15 stratified split.  Change to test
    sensitivity to the specific split used."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio is implicit: 1.0 − train_ratio − val_ratio

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir: str = "results"
    """Root directory for all run artefacts.  On FARM point this to group
    storage, e.g. /group/jmearlesgrp/$USER/benchmark_results."""

    # ── Phase 2 — embeddings ───────────────────────────────────────────────────
    embed_model: str = "facebook/dinov2-base"
    """DINOv2 model passed to ``AutoModel.from_pretrained()``.
    Options: dinov2-small (384-d) | dinov2-base (768-d) | dinov2-large (1024-d)
    """

    embed_batch_size: int = 64
    """Images per GPU forward pass.  Reduce for very high-resolution datasets
    (e.g. agarwood ~4.75 MB/image) or when GPU memory is constrained."""

    near_dup_threshold: float = 0.98
    """Cosine similarity threshold for near-duplicate detection.
    1.0 = exact pixel match after embedding; lower values flag visually very
    similar pairs."""

    faiss_ivf_threshold: int = 100_000
    """Dataset size above which FAISS switches from exact (IndexFlatIP) to
    approximate (IndexIVFFlat) search to keep memory bounded."""

    max_silhouette_samples: int = 10_000
    """Silhouette score is O(N²) in memory.  When N exceeds this value a
    stratified random sample is drawn instead of using the full matrix."""

    # ── Phase 3 (reserved) ─────────────────────────────────────────────────────
    backbone: str = "resnet18"
    cv_folds: int = 5

    # ── Derived ────────────────────────────────────────────────────────────────
    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.val_ratio, 10)

    # ── Validation ─────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio
        if total >= 1.0:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1.0, got {total:.2f}.  "
                f"The remainder becomes the test split."
            )

    # ── Discovery & JSON I/O ───────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[str] = None) -> "PipelineConfig":
        """
        Auto-discover and load a config, falling back to built-in defaults.

        Precedence:
        1. ``path`` argument (explicit override)
        2. ``$AGML_CONFIG`` environment variable
        3. ``benchmark/config.json`` relative to the current working directory
        4. All defaults (no file required)

        Example — typical SLURM job::

            # In the .sbatch script:
            #   export AGML_CONFIG=/group/jmearlesgrp/$USER/configs/farm.json
            cfg = PipelineConfig.load()   # picks up $AGML_CONFIG automatically
        """
        resolved = path
        if resolved is None:
            resolved = os.environ.get("AGML_CONFIG")
        if resolved is None and os.path.exists("benchmark/config.json"):
            resolved = "benchmark/config.json"
        if resolved is not None:
            return cls.from_json(resolved)
        return cls()

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """
        Load from an explicit JSON file path.

        Unknown keys in the file are silently ignored so that old config files
        remain forward-compatible as new fields are added.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path) as fh:
            raw: Dict[str, Any] = json.load(fh)
        valid = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        return cls(**filtered)

    def to_json(self, path: str) -> None:
        """
        Serialise the current config to JSON for reproducibility logging::

            cfg.to_json(os.path.join(run_dir, "config_used.json"))
        """
        path = os.path.expandvars(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Return config as a plain JSON-serialisable dict."""
        return dataclasses.asdict(self)
