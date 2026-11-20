"""Shared model format and project paths. Load only trusted joblib files."""
import json
from pathlib import Path

import joblib
import numpy as np

from .preprocess import IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(r"E:\eigenfaces\img_align_celeba")
DEFAULT_MODEL = ROOT / "models" / "eigenfaces.joblib"
DEFAULT_OUTPUT = ROOT / "output"
PAYLOAD_SIZES = (32, 48, 64, 80, 96)
FORMAT_VERSION = 1


def output_path(path: str | Path) -> Path:
    path = Path(path)
    if any(path.resolve().is_relative_to(root.resolve()) for root in (DEFAULT_DATASET, ROOT / "dataset")):
        raise ValueError("Output must be outside the original dataset directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_images(dataset: str | Path) -> list[Path]:
    dataset = Path(dataset)
    files = sorted(p for p in dataset.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"} and p.is_file())
    if not files:
        raise ValueError(f"No JPEG images found in {dataset}")
    return files


def load_model(path: str | Path = DEFAULT_MODEL) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run 'python main.py train' first.")
    bundle = joblib.load(path)
    metadata = bundle["metadata"]
    pca = bundle["pca"]
    count = metadata["components"]
    if metadata["format_version"] != FORMAT_VERSION or tuple(metadata["image_size"]) != IMAGE_SIZE:
        raise ValueError("Unsupported model format or image dimensions")
    low = np.asarray(metadata["quantization"]["minimum"], dtype=np.float32)
    high = np.asarray(metadata["quantization"]["maximum"], dtype=np.float32)
    if (low.shape != (count,) or high.shape != (count,) or
            not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or
            np.any(high < low) or pca.components_.shape != (count, 5120) or
            not np.isfinite(pca.components_).all() or pca.mean_.shape != (5120,) or
            not np.isfinite(pca.mean_).all()):
        raise ValueError("Invalid shared PCA model or quantization ranges")
    return bundle


def save_model(bundle: dict, path: str | Path) -> None:
    path = output_path(path)
    joblib.dump(bundle, path, compress=3)
    output_path(path.with_suffix(".json")).write_text(json.dumps(bundle["metadata"], indent=2) + "\n")


def validate_size(size: int, bundle: dict) -> None:
    if size not in PAYLOAD_SIZES:
        raise ValueError(f"Payload size must be one of {PAYLOAD_SIZES}; got {size}")
    if size > bundle["metadata"]["components"]:
        raise ValueError(f"Payload needs {size} components; model has {bundle['metadata']['components']}")
