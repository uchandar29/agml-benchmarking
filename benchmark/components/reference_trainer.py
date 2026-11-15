"""
ReferenceModelTrainer
=====================
Fine-tunes a lightweight CNN (ResNet-18 by default) on the training split
and records per-sample softmax confidence at every epoch.  The resulting
confidence history is used by DatasetCartographyMetric and the trained model
is passed to ClassConfusabilityMetric.

Design notes
------------
The model checkpoint and confidence history are cached to disk so that
Phase 3 can be re-run without retraining.  Set reuse=False to force a
fresh training run.

Output (written to run_dir/phase3/)
-------------------------------------
  reference_model.pt       -- final model state dict
  confidence_history.npy   -- float32 array of shape (N_train, n_epochs)
                              row i = per-epoch max softmax confidence for
                              the training image whose full-dataset row is orig_idx_order[i]
  orig_idx_order.npy       -- int64 array mapping row i → row number in the full dataset
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from core.data import DatasetSchema, ImageDataset, imagenet_evaluation_transform


class ReferenceModelTrainer:
	"""
	Fine-tunes a torchvision backbone on the training split and tracks
	per-sample confidence across all epochs.

	Usage::

		trainer = ReferenceModelTrainer(backbone="resnet18", n_epochs=30)
		model, conf_history, orig_idx_order = trainer.run(
			train_dataset=splits.train,
			train_indices=splits.indices["train"],
			schema=schema,
			run_dir=writer.run_dir,
		)
		# model            -- trained nn.Module in eval mode
		# conf_history     -- np.ndarray (N_train, n_epochs) float32
		# orig_idx_order   -- np.ndarray (N_train,) int64, maps row → full-dataset row
	"""

	def __init__(
		self,
		backbone: str = "resnet18",
		n_epochs: int = 30,
		lr: float = 1e-3,
		batch_size: int = 64,
		device: Optional[str] = None,
	) -> None:
		self.backbone = backbone
		self.n_epochs = n_epochs
		self.lr = lr
		self.batch_size = batch_size
		self._device_override = device

	def run(
		self,
		train_dataset: Dataset,
		train_indices: np.ndarray,
		schema: DatasetSchema,
		run_dir: str,
		reuse: bool = True,
	) -> Tuple[object, np.ndarray, np.ndarray]:
		"""
		Train (or restore from cache) the reference model.

		Parameters
		----------
		train_dataset : HF Dataset of the training split (splits.train).
		train_indices : Row numbers of the training split in the full dataset (splits.indices["train"]).
		schema        : Resolved DatasetSchema.
		run_dir       : Root run directory — phase3/ subfolder is used.
		reuse         : If True and cached files exist, skip training.

		Returns
		-------
		model          : Trained nn.Module in eval mode, on CPU.
		conf_history   : float32 (N_train, n_epochs) — per-sample per-epoch confidence.
		orig_idx_order : int64 (N_train,) — maps row index → row number in the full dataset.
		"""
		phase3_dir = os.path.join(run_dir, "phase3")
		model_path = os.path.join(phase3_dir, "reference_model.pt")
		conf_path = os.path.join(phase3_dir, "confidence_history.npy")
		order_path = os.path.join(phase3_dir, "orig_idx_order.npy")

		if reuse and all(os.path.exists(p) for p in [model_path, conf_path, order_path]):
			print(f"  Reusing cached reference model from {phase3_dir}")
			model = self._build_model(schema.num_classes)
			import torch
			model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
			model.eval()
			return model, np.load(conf_path), np.load(order_path)

		os.makedirs(phase3_dir, exist_ok=True)

		model, conf_history, orig_idx_order = self._train(train_dataset, train_indices, schema)

		import torch
		torch.save(model.state_dict(), model_path)
		np.save(conf_path, conf_history)
		np.save(order_path, orig_idx_order)
		print(f"  Reference model saved → {phase3_dir}")

		return model, conf_history, orig_idx_order

	def _build_model(self, num_classes: int):
		import torch.nn as nn
		import torchvision.models as tvm

		weights_map = {
			"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
			"resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2,
		}
		weights = weights_map.get(self.backbone)

		model_fn = getattr(tvm, self.backbone, None)
		if model_fn is None:
			raise ValueError(
				f"Unsupported backbone '{self.backbone}'.  "
				f"Supported: {list(weights_map.keys())}"
			)

		model = model_fn(weights=weights)

		# Replace the final classification layer for the target number of classes
		if hasattr(model, "fc"):
			in_features = model.fc.in_features
			model.fc = nn.Linear(in_features, num_classes)
		elif hasattr(model, "classifier"):
			in_features = model.classifier[-1].in_features
			model.classifier[-1] = nn.Linear(in_features, num_classes)

		return model

	def _train(
		self,
		train_dataset: Dataset,
		train_indices: np.ndarray,
		schema: DatasetSchema,
	) -> Tuple[object, np.ndarray, np.ndarray]:
		import torch
		import torch.nn as nn
		import torch.optim as optim
		from torch.utils.data import DataLoader

		device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
		print(f"  Training {self.backbone} on {device} for {self.n_epochs} epochs …")

		train_images = ImageDataset(
			train_dataset,
			schema.image_col,
			imagenet_evaluation_transform(),
			label_column=schema.label_col,
			original_indices=train_indices,
		)
		loader = DataLoader(
			train_images,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=0,
			pin_memory=(device == "cuda"),
		)

		model = self._build_model(schema.num_classes).to(device)
		model.train()

		optimizer = optim.Adam(model.parameters(), lr=self.lr)
		criterion = nn.CrossEntropyLoss()

		N = len(train_dataset)
		n_epochs = self.n_epochs

		# Map full-dataset row number → row position in conf_history
		orig_idx_order = np.asarray(train_indices, dtype=np.int64)
		idx_to_row: Dict[int, int] = {int(oi): i for i, oi in enumerate(orig_idx_order)}

		conf_history = np.zeros((N, n_epochs), dtype=np.float32)

		for epoch in tqdm(range(n_epochs), desc=f"Training {self.backbone}"):
			model.train()
			epoch_conf: Dict[int, float] = {}

			for images, labels, orig_indices in loader:
				images = images.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()
				logits = model(images)
				loss = criterion(logits, labels)
				loss.backward()
				optimizer.step()

				# Record max softmax confidence for each sample in the batch
				with torch.no_grad():
					probs = torch.softmax(logits, dim=1)
					max_conf, _ = probs.max(dim=1)
					for oi, conf in zip(orig_indices.tolist(), max_conf.cpu().tolist()):
						epoch_conf[oi] = conf

			# Fill conf_history for this epoch
			for oi, conf in epoch_conf.items():
				row = idx_to_row[oi]
				conf_history[row, epoch] = conf

		model.eval()
		model.to("cpu")

		print(
			f"  Training complete  "
			f"final mean confidence={conf_history[:, -1].mean():.3f}"
		)

		return model, conf_history, orig_idx_order
