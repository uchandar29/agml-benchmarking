"""AgML classification dataset loading and image preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import agml
import numpy as np
from PIL import Image


@dataclass
class ImageSample:
    """One image plus its ground-truth label and a stable id."""

    imageId: str
    image: Image.Image
    trueLabel: str


def ToPilRgb(rawImage: object) -> Image.Image:
    """Convert an AgML image (PIL or numpy) into a PIL RGB image."""
    pilImage: Image.Image
    if isinstance(rawImage, Image.Image):
        pilImage = rawImage
    else:
        # AgML often yields numpy HWC uint8 arrays.
        npImage: np.ndarray = np.asarray(rawImage)
        pilImage = Image.fromarray(npImage)
    return pilImage.convert("RGB")


def CapResolution(image: Image.Image, maxPixels: int) -> Image.Image:
    """Downscale an image in place (proportionally) if it exceeds the pixel cap."""
    currentPixels: int = image.width * image.height
    if currentPixels <= maxPixels:
        return image
    scale: float = (maxPixels / currentPixels) ** 0.5
    newSize: tuple[int, int] = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(newSize, Image.BILINEAR)


def LabelToName(loader: "agml.data.AgMLDataLoader", label: object) -> str:
    """Map an integer/string label to its human-readable class name."""
    classList: List[str] = list(loader.classes)
    if isinstance(label, (int, np.integer)):
        return classList[int(label)]
    return str(label)


def LoadDataset(
    datasetName: str,
    dataDir: Optional[str] = None,
    maxPixels: int = 1048576,
    limit: Optional[int] = None,
) -> tuple[List[str], Iterator[ImageSample]]:
    """Load an AgML classification dataset; return (classList, sample iterator)."""
    loader: "agml.data.AgMLDataLoader" = agml.data.AgMLDataLoader(datasetName, dataset_path=dataDir)
    classList: List[str] = list(loader.classes)

    def SampleIterator() -> Iterator[ImageSample]:
        """Yield prepared ImageSample objects one at a time."""
        index: int = 0
        for rawImage, label in loader:
            if limit is not None and index >= limit:
                break
            preparedImage: Image.Image = CapResolution(ToPilRgb(rawImage), maxPixels)
            sample: ImageSample = ImageSample(
                imageId=f"{datasetName}:{index}",
                image=preparedImage,
                trueLabel=LabelToName(loader, label),
            )
            yield sample
            index += 1

    return classList, SampleIterator()
