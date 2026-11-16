"""Decode with only the payload and shared model; no source image needed."""
from pathlib import Path

import numpy as np

from .model import DEFAULT_MODEL, load_model, output_path, validate_size
from .preprocess import array_to_image


def decode_payload(payload: bytes, bundle: dict) -> np.ndarray:
    size = len(payload)
    validate_size(size, bundle)
    quantized = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
    ranges = bundle["metadata"]["quantization"]
    low = np.asarray(ranges["minimum"][:size], dtype=np.float32)
    high = np.asarray(ranges["maximum"][:size], dtype=np.float32)
    coefficients = low + (quantized / np.float32(255)) * (high - low)
    pca = bundle["pca"]
    reconstructed = coefficients @ pca.components_[:size] + pca.mean_
    return np.clip(reconstructed, 0, 1).astype(np.float32)


def decode_file(payload_path: str | Path, model_path=DEFAULT_MODEL,
                output: str | Path | None = None) -> np.ndarray:
    bundle = load_model(model_path)
    payload_path = Path(payload_path)
    validate_size(payload_path.stat().st_size, bundle)
    values = decode_payload(payload_path.read_bytes(), bundle)
    if output is not None:
        array_to_image(values).save(output_path(output), format="PNG")
    return values
