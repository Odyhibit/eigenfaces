"""One unsigned byte per coefficient; no per-image header or scales."""
from pathlib import Path

import numpy as np

from .model import DEFAULT_MODEL, load_model, output_path, validate_size
from .preprocess import preprocess_image


def encode_array(values: np.ndarray, bundle: dict, size: int = 64) -> bytes:
    validate_size(size, bundle)
    coefficients = bundle["pca"].transform(np.asarray(values, dtype=np.float32).reshape(1, -1))[0, :size]
    ranges = bundle["metadata"]["quantization"]
    low = np.asarray(ranges["minimum"][:size], dtype=np.float32)
    span = np.asarray(ranges["maximum"][:size], dtype=np.float32) - low
    normalized = np.divide(coefficients - low, span, out=np.zeros_like(low), where=span > 0)
    payload = np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8).tobytes()
    assert len(payload) == size
    return payload


def encode_image(image_path: str | Path, model_path=DEFAULT_MODEL, size=64,
                 output: str | Path | None = None) -> bytes:
    bundle = load_model(model_path)
    payload = encode_array(preprocess_image(image_path), bundle, size)
    if output is not None:
        output_path(output).write_bytes(payload)
    return payload
