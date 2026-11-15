"""
PyTorch view of a HuggingFace split, shared by the benchmark's reference model,
class confusability, fine-tuning and the encoder sensitivity experiment.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from datasets import Dataset


def to_rgb(image):
	"""Most models expect 3-channel input, so grayscale, RGBA and palette images are converted."""
	return image if image.mode == "RGB" else image.convert("RGB")


def imagenet_evaluation_transform():
	"""Resize 256, centre crop 224 and ImageNet normalisation, as used by the torchvision backbones."""
	from torchvision import transforms

	return transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


class ImageDataset:
	"""
	Returns the transformed image of each row, plus its label when label_column
	is given, plus its row number in the full dataset when original_indices is given:
		image  |  (image, label)  |  (image, label, original_index)
	"""

	def __init__(
		self,
		hf_split: Dataset,
		image_column: str,
		transform: Callable,
		label_column: Optional[str] = None,
		original_indices: Optional[Sequence[int]] = None,
		row_limit: Optional[int] = None,
	) -> None:
		if original_indices is not None and len(original_indices) != len(hf_split):
			raise ValueError(f"Got {len(original_indices)} original indices for {len(hf_split)} rows")
		self.hf_split = hf_split
		self.image_column = image_column
		self.transform = transform
		self.label_column = label_column
		self.original_indices = original_indices
		self.row_count = len(hf_split) if row_limit is None else min(row_limit, len(hf_split))

	def __len__(self) -> int:
		return self.row_count

	def __getitem__(self, row_index: int):
		record = self.hf_split[row_index]
		image = self.transform(to_rgb(record[self.image_column]))
		if self.label_column is None:
			return image
		label = int(record[self.label_column])
		if self.original_indices is None:
			return image, label
		return image, label, int(self.original_indices[row_index])
