from __future__ import annotations

from typing import Tuple


def build_classifier(timm_model_name: str, number_of_classes: int, image_size: int, use_pretrained_weights: bool = True):
	"""Pretrained timm backbone with a fresh classification head sized to the dataset."""
	import timm

	return timm.create_model(
		timm_model_name,
		pretrained=use_pretrained_weights,
		num_classes=number_of_classes,
		img_size=image_size,
	)


def get_normalization_statistics(model) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
	import timm

	model_data_config = timm.data.resolve_model_data_config(model)
	return tuple(model_data_config["mean"]), tuple(model_data_config["std"])


def download_pretrained_weights(timm_model_name: str) -> None:
	# Used by --prefetch-only so compute nodes can run with the hub cache offline
	import timm

	timm.create_model(timm_model_name, pretrained=True)
