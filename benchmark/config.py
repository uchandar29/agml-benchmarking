from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "benchmark" / "slurm" / "config.json"


@dataclass
class ClassificationConfig:
    """Settings for the image classification benchmark (the "image_classification" section of config.json)."""

    max_silhouette_samples: int = 10_000
    """Silhouette score is O(N²) in memory.  When N exceeds this value a
    stratified random sample is drawn instead of using the full matrix."""

    backbone: str = "resnet18"
    """Torchvision model used as the reference classifier for Dataset Cartography
    and Class Confusability.  resnet18 is the default — fast and well-understood."""

    cartography_epochs: int = 30
    """Number of epochs to fine-tune the reference model.  More epochs give
    smoother confidence trajectories but cost more compute."""

    cartography_lr: float = 1e-3
    """Learning rate for the reference model fine-tuning run."""


@dataclass
class DetectionConfig:
    """Settings for the object detection benchmark (the "object_detection" section of config.json)."""

    arch: str = "fasterrcnn_mobilenet_v3_large_fpn"
    n_epochs: int = 12
    lr: float = 0.01
    batch_size: int = 8
    image_min_size: int = 640
    max_eval_images: int = 2000
    score_threshold: float = 0.05
    missing_score_threshold: float = 0.7
    missing_iou: float = 0.5
    crop_max_per_class: int = 2000
    crop_max_total: int = 20000
    crop_context_pad: float = 0.1
    pretrained: bool = True
    """False builds the detector without COCO weights (offline tests only)."""


@dataclass
class PipelineConfig:
    """
    Infrastructure and tuning settings for the AgML benchmark pipelines.

    config.json has three sections: ``common`` (the fields of this class),
    ``image_classification`` and ``object_detection`` (one nested dataclass each).

    This config captures settings that are **stable across datasets** — compute
    resource limits, split reproducibility, output location, and Phase-2/3
    model choices.  Dataset-specific fields (dataset_name, label_col, etc.) are
    passed at runtime, not stored here.

    Config discovery (highest → lowest precedence)
    -----------------------------------------------
    1. ``PipelineConfig.from_json(path)`` — explicit path
    2. ``$AGML_CONFIG`` environment variable pointing to a JSON file
    3. ``benchmark/slurm/config.json``
    4. Built-in defaults (no file required)

    Use :py:meth:`load` to auto-discover::

        cfg = PipelineConfig.load()          # auto-discovers
        cfg = PipelineConfig.load("benchmark/slurm/config.json")  # explicit

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
    output_dir: str = "benchmark/results"
    """Root directory for all run artefacts.  Relative paths are resolved
    against the repo root, so results always land in benchmark/results."""

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

    cv_folds: int = 5
    """Number of cross-validation folds for Label Noise Rate (cleanlab OOF
    probabilities).  5 is the standard choice in the cleanlab paper."""

    # ── Per-task sections ──────────────────────────────────────────────────────
    image_classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    object_detection: DetectionConfig = field(default_factory=DetectionConfig)

    # ── Derived ────────────────────────────────────────────────────────────────
    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.val_ratio, 10)

    # ── Validation ─────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        output_path = Path(os.path.expandvars(os.path.expanduser(self.output_dir)))
        self.output_dir = str(output_path if output_path.is_absolute() else REPO_ROOT / output_path)

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
        3. ``benchmark/slurm/config.json``
        4. All defaults (no file required)

        Example — typical SLURM job::

            # In the .sbatch script:
            #   export AGML_CONFIG=/path/to/config.json
            cfg = PipelineConfig.load()   # picks up $AGML_CONFIG automatically
        """
        resolved = path
        if resolved is None:
            resolved = os.environ.get("AGML_CONFIG")
        if resolved is None and DEFAULT_CONFIG_PATH.exists():
            resolved = str(DEFAULT_CONFIG_PATH)
        if resolved is not None:
            return cls.from_json(resolved)
        return cls()

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """
        Load from an explicit JSON file path.

        Unknown keys in a section are silently ignored so that old config files
        remain forward-compatible as new fields are added.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path) as fh:
            raw: Dict[str, Any] = json.load(fh)
        common_valid = {f.name for f in dataclasses.fields(cls)} - {"image_classification", "object_detection"}
        filtered = {k: v for k, v in raw.get("common", {}).items() if k in common_valid}
        for section, section_class in (("image_classification", ClassificationConfig), ("object_detection", DetectionConfig)):
            section_valid = {f.name for f in dataclasses.fields(section_class)}
            filtered[section] = section_class(
                **{k: v for k, v in raw.get(section, {}).items() if k in section_valid}
            )
        return cls(**filtered)

    def to_json(self, path: str) -> None:
        """
        Serialise the current config to JSON for reproducibility logging::

            cfg.to_json(os.path.join(run_dir, "config_used.json"))
        """
        path = os.path.expandvars(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Return config in the nested config.json form."""
        everything = dataclasses.asdict(self)
        sections = {name: everything.pop(name) for name in ("image_classification", "object_detection")}
        return {"common": everything, **sections}
