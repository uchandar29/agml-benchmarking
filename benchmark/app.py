#!/usr/bin/env python3
"""
AgML Benchmark Pipeline — entry point.

Programmatic usage
------------------
    from benchmark.app import AgMLBenchmarkPipeline

    # Minimal — config auto-loaded from $AGML_CONFIG or benchmark/config.json
    pipeline = AgMLBenchmarkPipeline("Project-AgML/rice_leaf_disease_classification")
    pipeline.run(phases=[1, 2])

    # Dataset with a specific HF config name (e.g. multi-config dataset)
    pipeline = AgMLBenchmarkPipeline(
        "Project-AgML/watermelon_disease_classification",
        config_name="raw",
    )

    # Dataset with explicit column names (overrides auto-detection)
    pipeline = AgMLBenchmarkPipeline(
        "Project-AgML/banana_grade_variety_classification",
        label_col="label",
        image_col="image",
        metadata_cols=["variety", "scale"],
    )

    # Explicit PipelineConfig (e.g. loaded from a custom path)
    from benchmark.config import PipelineConfig
    cfg = PipelineConfig.from_json("benchmark/configs/farm.json")
    pipeline = AgMLBenchmarkPipeline(
        "Project-AgML/agarwood_leaf_disease_classification",
        cfg=cfg,
    )
    pipeline.run(phases=[1, 2])

CLI usage
---------
    # Minimal — config auto-loaded from $AGML_CONFIG or benchmark/config.json
    python -m benchmark.app --dataset Project-AgML/rice_leaf_disease_classification --phases 1 2

    # Dataset-specific overrides as CLI flags
    python -m benchmark.app \\
        --dataset Project-AgML/watermelon_disease_classification \\
        --hf-config-name raw \\
        --phases 1 2

    python -m benchmark.app \\
        --dataset Project-AgML/banana_grade_variety_classification \\
        --label-col label --image-col image \\
        --metadata-cols variety scale \\
        --phases 1 2

Config auto-discovery (no CLI flag needed)
------------------------------------------
    Set $AGML_CONFIG to point at a JSON file, or place benchmark/config.json
    in the working directory.  Fields not in the file fall back to defaults.

    On FARM, add to your .sbatch script:
        export AGML_CONFIG=/group/jmearlesgrp/$USER/configs/farm.json
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from benchmark.config import PipelineConfig
from benchmark.core.dataset_adapter import DatasetAdapter
from benchmark.core.embedding_engine import EmbeddingEngine
from benchmark.core.split_manager import SplitManager
from benchmark.metrics.difficulty import FeatureSeparabilityMetric
from benchmark.metrics.diversity import IntraClassDiversityMetric, MetadataCoverageMetric
from benchmark.metrics.structural import (
    ClassImbalanceMetric,
    ExactDuplicateMetric,
    NearDuplicateMetric,
    ResolutionConsistencyMetric,
)
from benchmark.output.writer import ReportWriter


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

        cfg = PipelineConfig.from_json("benchmark/configs/farm.json")
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
        cfg: Optional[PipelineConfig] = None,
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
        image_col:
            Column with Image dtype.  Auto-detected when None.
        metadata_cols:
            Columns to treat as metadata for MetadataCoverageMetric.
            Auto-detected (all non-image, non-label columns) when None.
            Pass ``[]`` to explicitly disable metadata analysis.
        cfg:
            Infrastructure config.  When None, auto-loaded via
            ``PipelineConfig.load()`` (checks ``$AGML_CONFIG``, then
            ``benchmark/config.json``, then built-in defaults).
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.label_col = label_col
        self.image_col = image_col
        self.metadata_cols = metadata_cols
        self.cfg = cfg if cfg is not None else PipelineConfig.load()

    # ── Public API ────────────────────────────────────────────────────────────

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

        writer = ReportWriter(
            dataset_name=self.dataset_name,
            output_dir=self.cfg.output_dir,
        )

        # Persist the exact config used alongside the report for reproducibility
        self.cfg.to_json(os.path.join(writer.run_dir, "config_used.json"))

        _banner(self.dataset_name, phases, writer.path())

        adapter = DatasetAdapter(
            dataset_name=self.dataset_name,
            config_name=self.config_name,
            label_col=self.label_col,
            image_col=self.image_col,
            metadata_cols=self.metadata_cols,
        )
        full_dataset = adapter.load()
        schema = adapter.schema()
        _print_schema(schema, len(full_dataset))

        split_mgr = SplitManager(
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
            seed=self.cfg.split_seed,
        )
        splits = split_mgr.split(full_dataset, schema.label_col, writer.run_dir)

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
                train_dataset=splits.train,
                val_dataset=splits.val,
                test_dataset=splits.test,
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

            train_idx = set(list(splits.train["_orig_idx"]))
            val_idx   = set(list(splits.val["_orig_idx"]))
            test_idx  = set(list(splits.test["_orig_idx"]))

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
                max_silhouette_samples=self.cfg.max_silhouette_samples,
            )
            writer.add("feature_separability", result)
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

        # ── Phase 3 (placeholder) ─────────────────────────────────────────────
        if 3 in phases:
            raise NotImplementedError("Phase 3 is not yet implemented.")

        print(f"\n{'─' * 60}")
        print(f"  Report → {writer.path()}")
        print(f"{'─' * 60}\n")

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agml-benchmark",
        description="AgML Dataset Benchmarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
config auto-discovery (no flag needed):
  1. $AGML_CONFIG environment variable
  2. benchmark/config.json in the current directory
  3. Built-in defaults

examples:
  # Minimal — config auto-loaded
  python -m benchmark.app --dataset Project-AgML/rice_leaf_disease_classification --phases 1 2

  # Multi-config HF dataset
  python -m benchmark.app \\
      --dataset Project-AgML/watermelon_disease_classification \\
      --hf-config-name raw --phases 1 2

  # Dataset with explicit column names and metadata
  python -m benchmark.app \\
      --dataset Project-AgML/banana_grade_variety_classification \\
      --label-col label --image-col image \\
      --metadata-cols variety scale \\
      --phases 1 2

  # FARM job: config picked up from $AGML_CONFIG set in .sbatch
  python -m benchmark.app \\
      --dataset Project-AgML/agarwood_leaf_disease_classification \\
      --phases 1 2
        """,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        required=True,
        metavar="HF_REPO",
        help="HuggingFace dataset identifier, e.g. Project-AgML/rice_leaf_disease_classification",
    )

    # ── Dataset-specific overrides ────────────────────────────────────────────
    parser.add_argument(
        "--hf-config-name",
        default=None,
        metavar="NAME",
        help="HF dataset config name (e.g. 'raw').  None → HF default.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Label column name.  Overrides auto-detection.",
    )
    parser.add_argument(
        "--image-col",
        default=None,
        help="Image column name.  Overrides auto-detection.",
    )
    parser.add_argument(
        "--metadata-cols",
        nargs="*",
        default=None,
        metavar="COL",
        help=(
            "Metadata column names for coverage analysis.  "
            "Pass no names (--metadata-cols) to disable metadata analysis."
        ),
    )

    # ── Run control ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=[1],
        help="Phases to run (default: 1).  Example: --phases 1 2",
    )

    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()

    # Infrastructure config: auto-discovered (no CLI flag)
    _cfg = PipelineConfig.load()

    AgMLBenchmarkPipeline(
        dataset_name=_args.dataset,
        config_name=_args.hf_config_name,
        label_col=_args.label_col,
        image_col=_args.image_col,
        metadata_cols=_args.metadata_cols,
        cfg=_cfg,
    ).run(phases=_args.phases)
