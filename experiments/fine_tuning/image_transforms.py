from __future__ import annotations

from typing import Any, Dict, Sequence


def build_training_transform(
	image_size: int,
	augmentation_settings: Dict[str, Any],
	normalization_mean: Sequence[float],
	normalization_std: Sequence[float],
):
	from torchvision import transforms

	color_jitter_settings = augmentation_settings["color_jitter"]
	return transforms.Compose([
		transforms.RandomResizedCrop(image_size, scale=tuple(augmentation_settings["random_crop_scale"])),
		transforms.RandomHorizontalFlip(p=augmentation_settings["horizontal_flip_probability"]),
		transforms.ColorJitter(
			brightness=color_jitter_settings["brightness"],
			contrast=color_jitter_settings["contrast"],
			saturation=color_jitter_settings["saturation"],
			hue=color_jitter_settings["hue"],
		),
		transforms.ToTensor(),
		transforms.Normalize(normalization_mean, normalization_std),
	])


def build_evaluation_transform(
	image_size: int,
	normalization_mean: Sequence[float],
	normalization_std: Sequence[float],
):
	from torchvision import transforms

	# Plain resize instead of center crop so symptoms near the image border are not cut off
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(normalization_mean, normalization_std),
	])
