"""
Image encoders shared by the benchmark (DINOv2 embeddings) and the encoder sensitivity experiment.

Every encoder turns one PIL image into a preprocessed tensor and a batch of
tensors into L2-normalised image embeddings, so the benchmark metrics can run
on any of them without changes.

Supported libraries:
	transformers_cls    DINOv2 / DINOv3 style backbones, CLS token of the last layer
	transformers_clip   CLIP / SigLIP style models, projected image embedding
	timm                Supervised timm backbones, pooled features
	open_clip           open_clip models (BioCLIP 2, Perception Encoder)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class EncoderSpec:
	name: str
	library: str
	model_id: str
	pretrained: Optional[str] = None
	batch_size: int = 64
	description: str = ""
	notes: List[str] = field(default_factory=list)


def parse_encoder_specs(raw_encoders: List[Dict[str, Any]]) -> List[EncoderSpec]:
	specs = []
	for raw_encoder in raw_encoders:
		specs.append(EncoderSpec(
			name=raw_encoder["name"],
			library=raw_encoder["library"],
			model_id=raw_encoder["model_id"],
			pretrained=raw_encoder.get("pretrained"),
			batch_size=int(raw_encoder.get("batch_size", 64)),
			description=raw_encoder.get("description", ""),
		))
	return specs


def l2_normalise(features):
	return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def pick_image_features(model_output):
	# transformers 4.x returns a tensor here, 5.x returns an output object with the projection in pooler_output
	if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
		return model_output.pooler_output
	if hasattr(model_output, "image_embeds") and model_output.image_embeds is not None:
		return model_output.image_embeds
	return model_output


class HuggingFacePreprocess:
	"""Picklable wrapper so data loader workers can use it under spawn or forkserver."""

	def __init__(self, image_processor) -> None:
		self.image_processor = image_processor

	def __call__(self, image):
		return self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]


class ImageEncoder:
	"""Common interface: preprocess(image) for data loader workers, encode(batch) on the GPU."""

	def __init__(self, spec: EncoderSpec, device: str) -> None:
		self.spec = spec
		self.device = device
		self.model = None
		self.preprocess: Optional[Callable] = None

	def load(self) -> None:
		raise NotImplementedError

	def forward(self, pixel_batch):
		raise NotImplementedError

	def encode(self, pixel_batch) -> np.ndarray:
		import torch

		with torch.no_grad():
			features = self.forward(pixel_batch.to(self.device))
			features = l2_normalise(features.float())
		return features.cpu().numpy().astype(np.float32)

	def unload(self) -> None:
		import gc

		import torch

		self.model = None
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()


class TransformersClsEncoder(ImageEncoder):
	"""CLS token of last_hidden_state, the embedding the benchmark uses for DINOv2."""

	def load(self) -> None:
		from transformers import AutoImageProcessor, AutoModel

		image_processor = AutoImageProcessor.from_pretrained(self.spec.model_id)
		self.model = AutoModel.from_pretrained(self.spec.model_id).eval().to(self.device)
		self.preprocess = HuggingFacePreprocess(image_processor)

	def forward(self, pixel_batch):
		return self.model(pixel_values=pixel_batch).last_hidden_state[:, 0, :]


class TransformersClipEncoder(ImageEncoder):
	def load(self) -> None:
		from transformers import AutoImageProcessor, AutoModel

		image_processor = AutoImageProcessor.from_pretrained(self.spec.model_id)
		self.model = AutoModel.from_pretrained(self.spec.model_id).eval().to(self.device)
		self.preprocess = HuggingFacePreprocess(image_processor)

	def forward(self, pixel_batch):
		return pick_image_features(self.model.get_image_features(pixel_values=pixel_batch))


class TimmEncoder(ImageEncoder):
	def load(self) -> None:
		import timm

		self.model = timm.create_model(self.spec.model_id, pretrained=True, num_classes=0).eval().to(self.device)
		data_config = timm.data.resolve_model_data_config(self.model)
		self.preprocess = timm.data.create_transform(**data_config, is_training=False)

	def forward(self, pixel_batch):
		return self.model(pixel_batch)


class OpenClipEncoder(ImageEncoder):
	def load(self) -> None:
		import open_clip

		model, _, evaluation_transform = open_clip.create_model_and_transforms(
			self.spec.model_id,
			pretrained=self.spec.pretrained,
		)
		self.model = model.eval().to(self.device)
		self.preprocess = evaluation_transform

	def forward(self, pixel_batch):
		return self.model.encode_image(pixel_batch)


ENCODER_CLASSES = {
	"transformers_cls": TransformersClsEncoder,
	"transformers_clip": TransformersClipEncoder,
	"timm": TimmEncoder,
	"open_clip": OpenClipEncoder,
}


def build_encoder(spec: EncoderSpec, device: str) -> ImageEncoder:
	if spec.library not in ENCODER_CLASSES:
		raise ValueError(f"Unknown encoder library '{spec.library}' for {spec.name}. Choose from {sorted(ENCODER_CLASSES)}")
	return ENCODER_CLASSES[spec.library](spec, device)


def download_encoder_weights(spec: EncoderSpec) -> None:
	"""Loads the encoder once on the CPU so its weights land in the local cache (for the login node)."""
	encoder = build_encoder(spec, device="cpu")
	encoder.load()
	encoder.unload()
