"""
EmbeddingEngine
===============
Runs DINOv2 inference over the full dataset in batches and caches the result
to disk so every Phase 2 metric can share a single embedding pass.

Model
-----
DINOv2 (facebook/dinov2-base by default), loaded through the shared encoder in
core/embeddings. The CLS token of the final hidden state is the image-level
representation, the standard extraction point for DINOv2.

Embeddings are L2-normalised, so cosine similarity equals inner product, which
the FAISS IndexFlatIP index in NearDuplicateMetric relies on.

Embedding dim by model variant
-------------------------------
  facebook/dinov2-small  → 384
  facebook/dinov2-base   → 768   ← default
  facebook/dinov2-large  → 1024
  facebook/dinov2-giant  → 1536

Cache files (written to run_dir/embeddings)
--------------------------------------------
  embeddings.npy  — float32 array, shape (N, D)
  labels.npy      — int64 array,   shape (N,)

Row i of both arrays is row i of the full dataset.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from core.data import DatasetSchema, to_rgb
from core.embeddings import EncoderSpec, build_encoder


class EmbeddingEngine:
	"""
	Usage::

		engine = EmbeddingEngine()
		embeddings, labels = engine.run(
			full_dataset=splits.full,
			schema=schema,
			run_dir=writer.run_dir,
		)
		# embeddings: np.ndarray (N, D), float32, L2-normalised
		# labels:     np.ndarray (N,),   int64
	"""

	def __init__(
		self,
		model_name: str = "facebook/dinov2-base",
		batch_size: int = 64,
		device: Optional[str] = None,
	) -> None:
		self.model_name = model_name
		self.batch_size = batch_size
		# Resolved when the model loads, so the constructor does not import torch
		self._device_override = device
		self._encoder = None

	def run(
		self,
		full_dataset: Dataset,
		schema: DatasetSchema,
		run_dir: str,
		reuse: bool = True,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Compute (or restore from cache) embeddings for the full dataset.

		Parameters
		----------
		full_dataset : The full dataset (splits.full).
		schema       : Resolved DatasetSchema.
		run_dir      : Directory to read/write embeddings.npy and labels.npy.
		reuse        : If True and cached files exist, skip inference.

		Returns
		-------
		embeddings : float32 ndarray of shape (N, D), L2-normalised.
		labels     : int64 ndarray of shape (N,).
		"""
		embedding_folder = os.path.join(run_dir, "embeddings")
		embedding_path = os.path.join(embedding_folder, "embeddings.npy")
		label_path = os.path.join(embedding_folder, "labels.npy")

		if reuse and os.path.exists(embedding_path) and os.path.exists(label_path):
			print(f"  Reusing cached embeddings from {embedding_folder}")
			return np.load(embedding_path), np.load(label_path)

		self._load_model()
		embeddings, labels = self._encode(full_dataset, schema)

		os.makedirs(embedding_folder, exist_ok=True)
		np.save(embedding_path, embeddings)
		np.save(label_path, labels)
		print(f"  Embeddings saved → {embedding_folder}  shape={embeddings.shape}")
		return embeddings, labels

	def _load_model(self) -> None:
		if self._encoder is not None:
			return
		import torch

		device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
		print(f"  Loading {self.model_name} on {device} …")
		spec = EncoderSpec(name="benchmark", library="transformers_cls", model_id=self.model_name, batch_size=self.batch_size)
		self._encoder = build_encoder(spec, device)
		self._encoder.load()
		print("  Model ready.")

	def _encode(self, full_dataset: Dataset, schema: DatasetSchema) -> Tuple[np.ndarray, np.ndarray]:
		import torch

		all_embeddings: list[np.ndarray] = []
		all_labels: list[int] = []

		for start in tqdm(range(0, len(full_dataset), self.batch_size), desc="Embedding (DINOv2)"):
			batch = full_dataset[start : start + self.batch_size]
			pixel_batch = torch.stack([self._encoder.preprocess(to_rgb(image)) for image in batch[schema.image_col]])
			all_embeddings.append(self._encoder.encode(pixel_batch))
			all_labels.extend(batch[schema.label_col])

		embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
		labels = np.array(all_labels, dtype=np.int64)
		return embeddings, labels
