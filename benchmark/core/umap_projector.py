"""
UMAPProjector
=============
Reduces DINOv2 embeddings to 2D and 3D using PCA → UMAP and saves the
coordinates as JSON files for use in the Docusaurus website.

Output (written to run_dir/embeddings/)
----------------------------------------
  umap_2d.json  — list of {x, y, label, split, index}
  umap_3d.json  — list of {x, y, z, label, split, index}

Each point carries its class name, split membership, and original dataset
index so the website can filter/colour by any of these fields.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Set

import numpy as np


class UMAPProjector:
    """
    Usage::

        projector = UMAPProjector()
        projector.run(
            embeddings=embeddings,       # (N, D) float32, L2-normalised
            labels=labels,               # (N,) int64
            label_names=schema.label_names,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            run_dir=writer.run_dir,
        )
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
        n_pca_components: int = 50,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.n_pca_components = n_pca_components

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: List[str],
        train_idx: Set[int],
        val_idx: Set[int],
        test_idx: Set[int],
        run_dir: str,
    ) -> None:
        """
        Compute UMAP 2D and 3D projections and save to run_dir/embeddings/.

        Parameters
        ----------
        embeddings  : (N, D) L2-normalised float32 array.
        labels      : (N,) integer class indices.
        label_names : class name for each integer label.
        train_idx   : set of original indices in the train split.
        val_idx     : set of original indices in the val split.
        test_idx    : set of original indices in the test split.
        run_dir     : root run directory (embeddings/ subfolder used).
        """
        try:
            import umap as umap_lib
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "UMAP projection requires umap-learn and scikit-learn.  "
                "Install with:  pip install umap-learn scikit-learn"
            )

        emb_dir = os.path.join(run_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)

        N, D = embeddings.shape
        print(f"  UMAP projection: {N} samples × {D} dims")

        # ── PCA pre-reduction ─────────────────────────────────────────────────
        n_pca = min(self.n_pca_components, D, N)
        pca = PCA(n_components=n_pca, random_state=self.random_state)
        emb_pca = pca.fit_transform(embeddings)
        print(f"  PCA {D}d → {n_pca}d  ({pca.explained_variance_ratio_.sum():.1%} variance)")

        # ── UMAP 2D ───────────────────────────────────────────────────────────
        print("  UMAP 2D ...", end=" ", flush=True)
        coords_2d = umap_lib.UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        ).fit_transform(emb_pca)
        print("done")

        # ── UMAP 3D ───────────────────────────────────────────────────────────
        print("  UMAP 3D ...", end=" ", flush=True)
        coords_3d = umap_lib.UMAP(
            n_components=3,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        ).fit_transform(emb_pca)
        print("done")

        # ── Build split lookup ────────────────────────────────────────────────
        def split_of(i: int) -> str:
            if i in train_idx: return "train"
            if i in val_idx:   return "val"
            if i in test_idx:  return "test"
            return "unknown"

        # ── Serialise ─────────────────────────────────────────────────────────
        records_2d = [
            {
                "x":     round(float(coords_2d[i, 0]), 4),
                "y":     round(float(coords_2d[i, 1]), 4),
                "label": label_names[int(labels[i])],
                "split": split_of(i),
                "index": i,
            }
            for i in range(N)
        ]

        records_3d = [
            {
                "x":     round(float(coords_3d[i, 0]), 4),
                "y":     round(float(coords_3d[i, 1]), 4),
                "z":     round(float(coords_3d[i, 2]), 4),
                "label": label_names[int(labels[i])],
                "split": split_of(i),
                "index": i,
            }
            for i in range(N)
        ]

        path_2d = os.path.join(emb_dir, "umap_2d.json")
        path_3d = os.path.join(emb_dir, "umap_3d.json")

        with open(path_2d, "w") as fh:
            json.dump(records_2d, fh, separators=(",", ":"))
        with open(path_3d, "w") as fh:
            json.dump(records_3d, fh, separators=(",", ":"))

        print(f"  UMAP JSON saved → {emb_dir}/umap_2d.json + umap_3d.json")
