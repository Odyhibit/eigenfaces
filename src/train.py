"""Reproducible randomized PCA training and shared quantization calibration."""
from pathlib import Path

import numpy as np
import joblib
import PIL
import sklearn
from sklearn.decomposition import PCA

from .model import DEFAULT_DATASET, DEFAULT_MODEL, FORMAT_VERSION, find_images, save_model
from .preprocess import IMAGE_SIZE, preprocess_image


def train_model(dataset=DEFAULT_DATASET, model_path=DEFAULT_MODEL, samples=5000,
                components=128, seed=42) -> dict:
    files = find_images(dataset)
    if samples < 2 or samples > len(files):
        raise ValueError(f"Training samples must be between 2 and {len(files)}")
    if not 1 <= components <= min(samples - 1, np.prod(IMAGE_SIZE)):
        raise ValueError("Components must be between 1 and min(samples - 1, 5120)")
    indices = np.random.default_rng(seed).choice(len(files), samples, replace=False)
    selected = [files[i] for i in indices]
    matrix = np.empty((samples, np.prod(IMAGE_SIZE)), dtype=np.float32)
    for i, path in enumerate(selected):
        matrix[i] = preprocess_image(path)
        if (i + 1) % 500 == 0 or i + 1 == samples:
            print(f"Preprocessed {i + 1}/{samples} images", flush=True)
    print(f"Training randomized PCA with {components} components...", flush=True)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(matrix)
    # Use the encoder's transform, not the approximate randomized fit_transform.
    coefficients = pca.transform(matrix)
    if not np.isfinite(coefficients).all() or not np.isfinite(pca.components_).all():
        raise ValueError("PCA produced nonfinite values; model was not saved")
    metadata = {
        "format_version": FORMAT_VERSION,
        "image_size": list(IMAGE_SIZE), "components": components,
        "normalization": "grayscale L / 255; float32; flattened row-major",
        "resize_filter": "LANCZOS", "samples": samples, "seed": seed,
        "solver": "randomized", "dataset_root": str(Path(dataset).resolve()),
        "software_versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                              "pillow": PIL.__version__, "joblib": joblib.__version__},
        "training_files": [p.relative_to(dataset).as_posix() for p in selected],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "quantization": {"method": "training_min_max", "bits": 8,
                         "minimum": coefficients.min(axis=0).tolist(),
                         "maximum": coefficients.max(axis=0).tolist()},
    }
    bundle = {"pca": pca, "metadata": metadata}
    save_model(bundle, model_path)
    print(f"Cumulative explained variance: {metadata['cumulative_explained_variance']:.2%}")
    print(f"Saved model: {model_path}")
    return bundle
