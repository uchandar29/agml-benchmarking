"""
Image classification benchmark pipeline (library class, started by benchmark/run.py).

    from benchmark.pipelines.image_classification import AgMLBenchmarkPipeline

    pipeline = AgMLBenchmarkPipeline("Project-AgML/rice_leaf_disease_classification")
    pipeline.run(phases=[1, 2])

    # Dataset with a specific HF config name, explicit columns or a compound label
    AgMLBenchmarkPipeline("Project-AgML/watermelon_disease_classification", config_name="raw")
    AgMLBenchmarkPipeline("Project-AgML/banana_grade_variety_classification", label_col="label", image_col="image")

Settings come from a PipelineConfig: pass cfg=..., or it is auto-loaded from
$AGML_CONFIG, then benchmark/slurm/config.json, then built-in defaults.
Results go to <output_dir>/image_classification/<dataset>/<timestamp>/.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from benchmark.config import PipelineConfig
from core.data import DatasetAdapter, SplitManager
from core.embeddings import EncoderSpec, download_encoder_weights
from benchmark.components.embedding_engine import EmbeddingEngine
from benchmark.components.reference_trainer import ReferenceModelTrainer
from benchmark.components.umap_projector import UMAPProjector
from core.metrics.annotation import LabelNoiseMetric
from core.metrics.difficulty import (
    ClassConfusabilityMetric,
    DatasetCartographyMetric,
    FeatureSeparabilityMetric,
)
from core.metrics.diversity import IntraClassDiversityMetric, MetadataCoverageMetric
from core.metrics.structural import (
    ClassImbalanceMetric,
    ExactDuplicateMetric,
    NearDuplicateMetric,
    ResolutionConsistencyMetric,
)
from benchmark.output.writer import ReportWriter


TASK = "image_classification"


class AgMLBenchmarkPipeline:
    """
    Orchestrates the benchmark pipeline for any HuggingFace image-classification
    dataset across one or more phases.

    ``dataset_name`` is the only required argument — everything else is optional::

        pipeline = AgMLBenchmarkPipeline("Project-AgML/rice_leaf_disease_classification")
        pipeline.run(phases=[1, 2])

    Infrastructure settings (batch size, thresholds, output dir, …) come from a
    ``PipelineConfig`` that is auto-discovered at construction time.  Pass ``cfg``
    explicitly to override::

        cfg = PipelineConfig.from_json("benchmark/slurm/config.json")
        pipeline = AgMLBenchmarkPipeline("Project-AgML/...", cfg=cfg)
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        config_name: Optional[str] = None,
        label_col: Optional[str] = None,
        image_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None,
        compound_label_cols: Optional[List[str]] = None,
        cfg: Optional[PipelineConfig] = None,
        show_banner: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dataset_name:
            HuggingFace dataset identifier, e.g.
            ``'Project-AgML/rice_leaf_disease_classification'``.
        config_name:
            HF dataset config name (e.g. ``'raw'``).  None → HF default.
        label_col:
            Column with ClassLabel dtype.  Auto-detected when None.
            Ignored when compound_label_cols is set.
        image_col:
            Column with Image dtype.  Auto-detected when None.
        metadata_cols:
            Columns to treat as metadata for MetadataCoverageMetric.
            Auto-detected (all non-image, non-label columns) when None.
            Pass ``[]`` to explicitly disable metadata analysis.
        compound_label_cols:
            Two or more column names to combine into a single compound label.
            Values are joined with ", " in the given order, e.g.
            ``['crop_type', 'label']`` → ``"tomato, bruised"``.
            When set, overrides label_col.
        cfg:
            Infrastructure config.  When None, auto-loaded via
            ``PipelineConfig.load()`` (checks ``$AGML_CONFIG``, then
            ``benchmark/slurm/config.json``, then built-in defaults).
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.label_col = label_col
        self.image_col = image_col
        self.metadata_cols = metadata_cols
        self.compound_label_cols = compound_label_cols
        self.cfg = cfg if cfg is not None else PipelineConfig.load()
        self.show_banner = show_banner
        self._writer = None   # created lazily so the timestamp is stable

    # ── Public API ────────────────────────────────────────────────────────────

    def _get_writer(self):
        """Return the ReportWriter, creating it on first call."""
        if self._writer is None:
            self._writer = ReportWriter(
                dataset_name=self.dataset_name,
                output_dir=self.cfg.output_dir,
                task=TASK,
            )
        return self._writer

    def report_path(self) -> str:
        """
        Return the path where the JSON report will be written.
        Safe to call before run() — the timestamp is fixed on first call.
        """
        return self._get_writer().path()

    def _adapter(self) -> DatasetAdapter:
        return DatasetAdapter(
            dataset_name=self.dataset_name,
            config_name=self.config_name,
            label_col=self.label_col,
            image_col=self.image_col,
            metadata_cols=self.metadata_cols,
            compound_label_cols=self.compound_label_cols,
        )

    def prefetch(self) -> None:
        """Download the dataset and the DINOv2 weights into the caches. Nothing else."""
        self._adapter().download()
        download_encoder_weights(EncoderSpec(
            name="benchmark", library="transformers_cls",
            model_id=self.cfg.embed_model, batch_size=self.cfg.embed_batch_size,
        ))

    def run(self, phases: Optional[List[int]] = None) -> str:
        """
        Execute the requested phases in order.

        Parameters
        ----------
        phases:
            Phase numbers to run, e.g. ``[1]`` or ``[1, 2]``.
            Defaults to ``[1]`` when omitted.

        Returns
        -------
        str
            Absolute path to the JSON report file.
        """
        if phases is None:
            phases = [1]

        writer = self._get_writer()

        # Embed key reproducibility fields directly in the report so the file
        # is self-contained, and persist the full config alongside it.
        writer.set_reproducibility(self.cfg)
        self.cfg.to_json(os.path.join(writer.run_dir, "config_used.json"))

        if self.show_banner:
            _banner(self.dataset_name, phases, writer.path())

        adapter = self._adapter()
        full_dataset = adapter.load()
        schema = adapter.schema()
        _print_schema(schema, len(full_dataset))

        split_mgr = SplitManager(
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
            seed=self.cfg.split_seed,
        )
        splits = split_mgr.split(full_dataset, schema.label_col)
        split_index_sets = splits.index_sets()

        # ── Phase 1 ───────────────────────────────────────────────────────────
        if 1 in phases:
            print("\n── Phase 1: Structural Quality + Metadata Coverage ──\n")

            result = ClassImbalanceMetric().run(
                train_dataset=splits.train, schema=schema,
            )
            writer.add("class_imbalance", result)
            print(
                f"  [✓] class_imbalance        "
                f"ratio={result['imbalance_ratio']}  "
                f"entropy={result['normalized_entropy']}"
            )

            result = ExactDuplicateMetric().run(
                full_dataset=splits.full,
                split_indices=split_index_sets,
                schema=schema,
            )
            writer.add("exact_duplicate", result)
            print(
                f"  [✓] exact_duplicate        "
                f"rate={result['exact_duplicate_rate']}  "
                f"cross_split={result['cross_split_duplicates']}"
            )

            result = ResolutionConsistencyMetric().run(
                full_dataset=splits.full, schema=schema,
            )
            writer.add("resolution_consistency", result)
            print(f"  [✓] resolution_consistency area_cv={result['area_cv']}")

            result = MetadataCoverageMetric().run(
                full_dataset=splits.full, schema=schema,
            )
            writer.add("metadata_coverage", result)
            if result.get("skipped"):
                print("  [–] metadata_coverage      skipped (no metadata columns)")
            else:
                print(f"  [✓] metadata_coverage      columns={result['metadata_columns']}")

            writer.complete_phase(1)

        # ── Phase 2 ───────────────────────────────────────────────────────────
        if 2 in phases:
            print("\n── Phase 2: Embedding-Based Metrics (DINOv2) ──\n")

            engine = EmbeddingEngine(
                model_name=self.cfg.embed_model,
                batch_size=self.cfg.embed_batch_size,
            )
            embeddings, emb_labels = engine.run(
                full_dataset=splits.full,
                schema=schema,
                run_dir=writer.run_dir,
                reuse=True,
            )

            train_idx = split_index_sets["train"]
            val_idx   = split_index_sets["val"]
            test_idx  = split_index_sets["test"]

            result = NearDuplicateMetric(
                threshold=self.cfg.near_dup_threshold,
                ivf_threshold=self.cfg.faiss_ivf_threshold,
            ).run(
                embeddings=embeddings,
                labels=emb_labels,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                schema=schema,
                embed_model=self.cfg.embed_model,
            )
            writer.add("near_duplicate", result)
            print(
                f"  [✓] near_duplicate         "
                f"rate={result['near_duplicate_rate']}  "
                f"cross_split={result['cross_split_near_duplicates']}  "
                f"index={result['faiss_index_type']}"
            )

            result = FeatureSeparabilityMetric().run(
                embeddings=embeddings,
                labels=emb_labels,
                schema=schema,
                embed_model=self.cfg.embed_model,
                max_silhouette_samples=self.cfg.image_classification.max_silhouette_samples,
            )
            writer.add("feature_separability", result)
            if result.get("skipped"):
                print(f"  [–] feature_separability   skipped ({result['reason']})")
            else:
                print(
                    f"  [✓] feature_separability   "
                    f"silhouette={result['silhouette_score']}  "
                    f"({result['silhouette_interpretation']})  "
                    f"davies_bouldin={result['davies_bouldin_index']}"
                )

            result = IntraClassDiversityMetric().run(
                embeddings=embeddings,
                labels=emb_labels,
                schema=schema,
                embed_model=self.cfg.embed_model,
            )
            writer.add("intra_class_diversity", result)
            print(
                f"  [✓] intra_class_diversity  "
                f"mean={result['mean_diversity']}  "
                f"narrowest={result['min_diversity_class']}"
            )

            writer.complete_phase(2)

            # ── UMAP projection → JSON for website ────────────────────────────
            UMAPProjector().run(
                embeddings=embeddings,
                labels=emb_labels,
                label_names=schema.label_names,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                run_dir=writer.run_dir,
                dataset_name=self.dataset_name,
            )

        # ── Phase 3 ───────────────────────────────────────────────────────────
        if 3 in phases and schema.num_classes < 2:
            print("\n── Phase 3: Skipped — requires ≥ 2 classes ──\n")
            skip_reason = f"Dataset has only {schema.num_classes} class — Phase 3 metrics require ≥ 2."
            writer.add("dataset_cartography",  {"skipped": True, "reason": skip_reason})
            writer.add("class_confusability",  {"skipped": True, "reason": skip_reason})
            writer.add("label_noise",          {"skipped": True, "reason": skip_reason})
            writer.complete_phase(3)

        if 3 in phases and schema.num_classes >= 2:
            print("\n── Phase 3: Training Dynamics + Annotation Reliability ──\n")

            # Phase 2 embeddings are required for Label Noise — load from cache
            # if Phase 2 was not run in this session.
            engine = EmbeddingEngine(
                model_name=self.cfg.embed_model,
                batch_size=self.cfg.embed_batch_size,
            )
            embeddings, emb_labels = engine.run(
                full_dataset=splits.full,
                schema=schema,
                run_dir=writer.run_dir,
                reuse=True,
            )

            # Train reference model (ResNet-18) — shared by Cartography and
            # Confusability so we only pay the training cost once.
            trainer = ReferenceModelTrainer(
                backbone=self.cfg.image_classification.backbone,
                n_epochs=self.cfg.image_classification.cartography_epochs,
                lr=self.cfg.image_classification.cartography_lr,
                batch_size=self.cfg.embed_batch_size,
            )
            ref_model, conf_history, orig_idx_order = trainer.run(
                train_dataset=splits.train,
                train_indices=splits.indices["train"],
                schema=schema,
                run_dir=writer.run_dir,
                reuse=True,
            )

            result = DatasetCartographyMetric().run(
                conf_history=conf_history,
                orig_idx_order=orig_idx_order,
                schema=schema,
            )
            writer.add("dataset_cartography", result)
            print(
                f"  [✓] dataset_cartography    "
                f"easy={result['pct_easy']}%  "
                f"ambiguous={result['pct_ambiguous']}%  "
                f"hard={result['pct_hard']}%"
            )

            result = ClassConfusabilityMetric().run(
                model=ref_model,
                test_dataset=splits.test,
                schema=schema,
                batch_size=self.cfg.embed_batch_size,
            )
            writer.add("class_confusability", result)
            top_pair = result["top_confused_pairs"][0] if result["top_confused_pairs"] else {}
            print(
                f"  [✓] class_confusability    "
                f"accuracy={result['accuracy']}  "
                f"most_confused={top_pair.get('true_class', 'n/a')} → "
                f"{top_pair.get('predicted_as', 'n/a')} "
                f"({top_pair.get('confusion_rate', 0):.1%})"
            )

            # Full-dataset embeddings and labels for Label Noise k-fold CV
            full_orig_idx = np.arange(len(embeddings), dtype=np.int64)
            result = LabelNoiseMetric(cv_folds=self.cfg.cv_folds).run(
                embeddings=embeddings,
                labels=emb_labels,
                orig_idx_order=full_orig_idx,
                schema=schema,
            )
            writer.add("label_noise", result)
            print(
                f"  [✓] label_noise            "
                f"estimated_rate={result['estimated_noise_rate']:.1%}  "
                f"flagged={result['n_noisy_samples']}/{result['n_total_samples']}"
            )

            writer.complete_phase(3)

        print(f"\n  ✓ Successfully generated report at: {writer.path()}")

        return writer.path()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(dataset_name: str, phases: List[int], report_path: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  AgML Benchmark Pipeline")
    print(f"  Dataset : {dataset_name}")
    print(f"  Phases  : {phases}")
    print(f"  Report  : {report_path}")
    print(f"{'═' * 60}")


def _print_schema(schema, total: int) -> None:
    print(f"\nSchema")
    print(f"  image_col     : {schema.image_col}")
    print(f"  label_col     : {schema.label_col}")
    print(f"  num_classes   : {schema.num_classes}")
    print(f"  metadata_cols : {schema.metadata_cols or '(none)'}")
    print(f"  total images  : {total:,}")
