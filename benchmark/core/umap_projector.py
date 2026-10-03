"""
UMAPProjector
=============
Reduces DINOv2 embeddings to 2D and 3D using PCA → UMAP and saves the
projections as a single UEB2 binary file.

UEB2 Format
-----------
A compact binary format holding both 2D and 3D UMAP projections in one file.
All multi-byte values are little-endian.

	offset 0   4 bytes   magic = ASCII "UEB2"
	offset 4   1 byte    uint8  flags        (bit0 = has 2D, bit1 = has 3D)
	offset 5   2 bytes   uint16 labelCount
	offset 7   1 byte    uint8  splitCount
	offset 8   4 bytes   uint32 pointCount

	then: label table — labelCount entries, each:
	    1 byte    uint8 len
	    len bytes UTF-8 string

	then: split table — splitCount entries, each:
	    1 byte    uint8 len
	    len bytes UTF-8 string  (e.g. "train" / "val" / "test")

	then: pointCount metadata records (7 bytes each):
	    2 bytes   uint16 labelIdx
	    1 byte    uint8  splitIdx
	    4 bytes   uint32 index     (original sample index in the full dataset)

	then, if flags bit0 set: pointCount × 2D coords (8 bytes each):
	    4 bytes   float32 x
	    4 bytes   float32 y

	then, if flags bit1 set: pointCount × 3D coords (12 bytes each):
	    4 bytes   float32 x
	    4 bytes   float32 y
	    4 bytes   float32 z

Critical invariant: 2D and 3D projections must cover the same samples in the
same order — the shared metadata block is invalid otherwise. The writer asserts
this before writing.

Use benchmark/utils/ueb2_to_json.py to convert back to JSON for the frontend.

Output (written to run_dir/embeddings/)
----------------------------------------
	umap.bin  — single UEB2 binary file
"""

from __future__ import annotations

import os
import struct
from typing import Callable, List, Set

import numpy as np


_MAGIC		= b"UEB2"
_FLAG_HAS_2D	= 0b01
_FLAG_HAS_3D	= 0b10


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
		self.n_neighbors	= n_neighbors
		self.min_dist		= min_dist
		self.metric		= metric
		self.random_state	= random_state
		self.n_pca_components	= n_pca_components

	def run(
		self,
		embeddings: np.ndarray,
		labels: np.ndarray,
		label_names: List[str],
		train_idx: Set[int],
		val_idx: Set[int],
		test_idx: Set[int],
		run_dir: str,
		dataset_name: str = "",
	) -> None:
		"""
		Compute UMAP 2D and 3D projections and save to run_dir/embeddings/<name>_umap.bin.

		Parameters
		----------
		embeddings   : (N, D) L2-normalised float32 array.
		labels       : (N,) integer class indices.
		label_names  : class name for each integer label.
		train_idx    : set of original indices in the train split.
		val_idx      : set of original indices in the val split.
		test_idx     : set of original indices in the test split.
		run_dir      : root run directory (embeddings/ subfolder used).
		dataset_name : full dataset name (e.g. "iNatAg/abelmoschus_esculentus").
		               The bare species/dataset name is used as the filename prefix.
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

		# Both projections must cover the same N samples in the same order —
		# this is trivially true since both operate on the same emb_pca array,
		# but the assertion guards against future refactors that break this.
		assert len(coords_2d) == len(coords_3d) == N, (
			f"2D ({len(coords_2d)}) and 3D ({len(coords_3d)}) projections "
			f"have different point counts — shared metadata block would be misaligned."
		)

		# ── Build split lookup ────────────────────────────────────────────────
		def split_of(i: int) -> str:
			if i in train_idx: return "train"
			if i in val_idx:   return "val"
			if i in test_idx:  return "test"
			return "unknown"

		# ── Write UEB2 binary ─────────────────────────────────────────────────
		bare_name = dataset_name.split("/")[-1] if dataset_name else "umap"
		bin_path  = os.path.join(emb_dir, f"{bare_name}_umap.bin")
		self._write_ueb2(bin_path, coords_2d, coords_3d, labels, label_names, split_of, N)
		print(f"  UMAP binary saved → {bin_path}")

	# ── UEB2 writer ───────────────────────────────────────────────────────────

	def _write_ueb2(
		self,
		path: str,
		coords_2d: np.ndarray,
		coords_3d: np.ndarray,
		labels: np.ndarray,
		label_names: List[str],
		split_of: Callable[[int], str],
		N: int,
	) -> None:
		splits = [split_of(i) for i in range(N)]

		# Preserve insertion order so train=0, val=1, test=2 in typical runs
		unique_splits = list(dict.fromkeys(splits))
		split_to_idx  = {s: idx for idx, s in enumerate(unique_splits)}

		flags = _FLAG_HAS_2D | _FLAG_HAS_3D

		with open(path, "wb") as f:
			# ── Fixed header (12 bytes) ──────────────────────────────────────
			f.write(_MAGIC)
			f.write(struct.pack("<B", flags))
			f.write(struct.pack("<H", len(label_names)))
			f.write(struct.pack("<B", len(unique_splits)))
			f.write(struct.pack("<I", N))

			# ── Label table ─────────────────────────────────────────────────
			for name in label_names:
				encoded = name.encode("utf-8")
				assert len(encoded) <= 255, f"Label name too long (>255 bytes): {name!r}"
				f.write(struct.pack("<B", len(encoded)))
				f.write(encoded)

			# ── Split table ─────────────────────────────────────────────────
			for split in unique_splits:
				encoded = split.encode("utf-8")
				assert len(encoded) <= 255, f"Split name too long (>255 bytes): {split!r}"
				f.write(struct.pack("<B", len(encoded)))
				f.write(encoded)

			# ── Metadata records (7 bytes each) ──────────────────────────────
			# Structured numpy array allows a single bulk tobytes() call
			meta_dtype = np.dtype([
				("labelIdx", "<u2"),
				("splitIdx", "<u1"),
				("index",    "<u4"),
			])
			meta              = np.empty(N, dtype=meta_dtype)
			meta["labelIdx"]  = labels.astype(np.uint16)
			meta["splitIdx"]  = np.array([split_to_idx[s] for s in splits], dtype=np.uint8)
			meta["index"]     = np.arange(N, dtype=np.uint32)
			f.write(meta.tobytes())

			# ── 2D coordinate block ──────────────────────────────────────────
			f.write(coords_2d.astype(np.float32).tobytes())

			# ── 3D coordinate block ──────────────────────────────────────────
			f.write(coords_3d.astype(np.float32).tobytes())
