"""The single preprocessing path used by training and encoding."""
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = (64, 80)  # width, height


def preprocess_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32).reshape(-1) / np.float32(255)


def array_to_image(values: np.ndarray) -> Image.Image:
    values = np.asarray(values)
    if values.size != IMAGE_SIZE[0] * IMAGE_SIZE[1]:
        raise ValueError(f"Expected {IMAGE_SIZE[0] * IMAGE_SIZE[1]} pixels")
    pixels = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels.reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]))
