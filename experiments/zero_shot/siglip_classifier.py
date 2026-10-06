"""Contrastive zero-shot classifier: pick the class whose text embedding is closest to the image embedding."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# SigLIP was trained on text padded to 64 tokens; other padding lengths hurt accuracy
SIGLIP_TEXT_LENGTH = 64


def as_embedding_tensor(feature_output):
	# transformers 4.x returns the embedding tensor, 5.x returns an output object holding it in pooler_output
	if hasattr(feature_output, "pooler_output"):
		return feature_output.pooler_output
	return feature_output


class SiglipZeroShotClassifier:
	def __init__(self, model_settings: Dict[str, Any], model=None, processor=None, device=None) -> None:
		import torch

		self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.text_templates: List[str] = model_settings["text_templates"]
		self.batch_size: int = model_settings["batch_size"]
		if model is None or processor is None:
			from transformers import AutoModel, AutoProcessor

			model = AutoModel.from_pretrained(model_settings["model_id"])
			processor = AutoProcessor.from_pretrained(model_settings["model_id"])
		model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
		self.model = model.to(self.device, dtype=model_dtype).eval()
		self.processor = processor
		self.class_text_embeddings = None

	def set_class_names(self, display_class_names: List[str]) -> None:
		import torch

		per_class_embeddings = []
		with torch.no_grad():
			for class_name in display_class_names:
				prompts = [template.format(class_name=class_name) for template in self.text_templates]
				text_inputs = self.processor(
					text=prompts,
					padding="max_length",
					max_length=SIGLIP_TEXT_LENGTH,
					truncation=True,
					return_tensors="pt",
				).to(self.device)
				template_embeddings = as_embedding_tensor(self.model.get_text_features(**text_inputs)).float()
				template_embeddings = template_embeddings / template_embeddings.norm(dim=-1, keepdim=True)
				# Average over prompt templates, then re-normalise (standard prompt ensembling)
				class_embedding = template_embeddings.mean(dim=0)
				per_class_embeddings.append(class_embedding / class_embedding.norm())
		self.class_text_embeddings = torch.stack(per_class_embeddings)

	def predict(self, images: List[Any]) -> List[int]:
		import torch

		if self.class_text_embeddings is None:
			raise RuntimeError("set_class_names() must be called before predict()")
		predicted_labels: List[int] = []
		with torch.no_grad():
			for batch_start in range(0, len(images), self.batch_size):
				image_batch = images[batch_start:batch_start + self.batch_size]
				image_inputs = self.processor(images=image_batch, return_tensors="pt").to(self.device)
				image_inputs["pixel_values"] = image_inputs["pixel_values"].to(self.model.dtype)
				image_embeddings = as_embedding_tensor(self.model.get_image_features(**image_inputs)).float()
				image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
				# Sigmoid and logit scale are monotonic, so argmax over cosine similarity gives the same top-1
				similarity = image_embeddings @ self.class_text_embeddings.T
				predicted_labels.extend(similarity.argmax(dim=1).cpu().tolist())
		return predicted_labels

	def describe(self) -> Dict[str, Any]:
		return {"text_templates": self.text_templates, "text_length": SIGLIP_TEXT_LENGTH}
