"""
EmbeddingEngine
===============
Runs DINOv2 inference over the full dataset in batches and caches the result
to disk so every Phase 2 metric can share a single embedding pass.

Model
-----
DINOv2 (facebook/dinov2-base by default) via HuggingFace Transformers.
The CLS token from the final hidden state is used as the image-level
representation — this is the standard extraction point for DINOv2.

Embeddings are L2-normalised before saving.  This makes cosine similarity
equivalent to inner product, which is required by the FAISS IndexFlatIP
index used in NearDuplicateMetric.

Embedding dim by model variant
-------------------------------
  facebook/dinov2-small  → 384
  facebook/dinov2-base   → 768   ← default
  facebook/dinov2-large  → 1024
  facebook/dinov2-giant  → 1536

Cache files (written to run_dir)
---------------------------------
  embeddings.npy  — float32 array, shape (N, D)
  labels.npy      — int64 array,   shape (N,)

Both arrays are indexed 0 … N-1, where index i corresponds to
full_dataset[i] (which has _orig_idx == i).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from benchmark.core.dataset_adapter import DatasetSchema

# torch and transformers are imported lazily inside methods so that Phase 1
# remains importable on machines that do not have GPU dependencies installed.

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
        # Resolve device lazily so the constructor doesn't touch torch
        self._device_override = device

        self._processor = None
        self._model = None

    # ---------------------------- Public API ----------------------------
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
        full_dataset : Dataset with _orig_idx column (output of SplitManager).
        schema       : Resolved DatasetSchema.
        run_dir      : Directory to read/write embeddings.npy and labels.npy.
        reuse        : If True and cached files exist, skip inference.

        Returns
        -------
        embeddings : float32 ndarray of shape (N, D), L2-normalised.
        labels     : int64 ndarray of shape (N,).
        """
        emb_path   = os.path.join(run_dir, "embeddings.npy")
        label_path = os.path.join(run_dir, "labels.npy")

        if reuse and os.path.exists(emb_path) and os.path.exists(label_path):
            print(f"  Reusing cached embeddings from {run_dir}")
            return np.load(emb_path), np.load(label_path)

        self._load_model()

        embeddings, labels = self._encode(full_dataset, schema)

        os.makedirs(run_dir, exist_ok=True)
        np.save(emb_path,   embeddings)
        np.save(label_path, labels)
        print(f"  Embeddings saved → {emb_path}  shape={embeddings.shape}")

        return embeddings, labels

    #---------------------------- Internal ----------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "Phase 2 requires torch and transformers."
                "Install with:  pip install torch transformers"
            )

        self.device = self._device_override or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"  Loading {self.model_name} on {self.device} …")
        
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        
        # Switches the model to evaluation mode, disabling dropout and other training-specific layers.
        self._model.eval()
        
        # Move the model weights to the specified device for inference as 
        # model weight and input tensor must be on the same device.
        self._model.to(self.device)
        
        print(f"  Model ready.")

    def _encode(
        self,
        full_dataset: Dataset,
        schema: DatasetSchema,
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_col = schema.image_col
        label_col = schema.label_col
        N = len(full_dataset)

        all_embeddings: list[np.ndarray] = []
        all_labels: list[int] = []

        for start in tqdm(range(0, N, self.batch_size), desc="Embedding (DINOv2)"):
            batch = full_dataset[start : start + self.batch_size]
            pil_images = batch[image_col]
            batch_labels = batch[label_col]

            # Ensure RGB — DINOv2 processor expects 3-channel input
            pil_images = [
                img.convert("RGB") if img.mode != "RGB" else img
                for img in pil_images
            ]

            import torch  # already confirmed available by _load_model

            inputs = self._processor(
                images=pil_images,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # CLS token: last_hidden_state[:, 0, :] — shape (B, D)
                vecs = outputs.last_hidden_state[:, 0, :]
                # L2-normalise so inner product == cosine similarity
                vecs = vecs / vecs.norm(dim=-1, keepdim=True)

            all_embeddings.append(vecs.cpu().float().numpy())
            all_labels.extend(batch_labels)

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        labels     = np.array(all_labels, dtype=np.int64)
        return embeddings, labels
