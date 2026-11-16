"""Held-out comparisons, metrics, and PCA visualizations."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from .decode import decode_payload
from .encode import encode_array
from .model import DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_OUTPUT, PAYLOAD_SIZES, find_images, load_model, output_path, validate_size
from .preprocess import array_to_image, preprocess_image


def quality_metrics(original, reconstruction) -> dict:
    original = np.asarray(original).reshape(80, 64)
    reconstruction = np.asarray(reconstruction).reshape(80, 64)
    return {"mse": float(mean_squared_error(original, reconstruction)),
            "psnr": float(peak_signal_noise_ratio(original, reconstruction, data_range=1)),
            "ssim": float(structural_similarity(original, reconstruction, data_range=1))}


def visualize_model(bundle, output=DEFAULT_OUTPUT):
    pca = bundle["pca"]
    array_to_image(pca.mean_).save(output_path(Path(output) / "average_face.png"))
    fig, axes = plt.subplots(4, 4, figsize=(8, 10))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(pca.components_):
            face = pca.components_[i].reshape(80, 64)
            limit = max(float(np.abs(face).max()), 1e-9)
            ax.imshow(face, cmap="gray", vmin=-limit, vmax=limit)
            ax.set_title(f"PC {i + 1}")
    fig.suptitle("Eigenfaces (independent symmetric contrast scales)")
    fig.tight_layout()
    fig.savefig(output_path(Path(output) / "eigenfaces.png"), dpi=140)
    plt.close(fig)


def compare_images(dataset=DEFAULT_DATASET, model_path=DEFAULT_MODEL,
                   output=DEFAULT_OUTPUT, samples=10, seed=43):
    bundle = load_model(model_path)
    for size in PAYLOAD_SIZES:
        validate_size(size, bundle)
    training = set(bundle["metadata"]["training_files"])
    files = [p for p in find_images(dataset) if p.relative_to(dataset).as_posix() not in training]
    if samples < 1 or samples > len(files):
        raise ValueError(f"Comparison samples must be between 1 and {len(files)} held-out images")
    selected = np.random.default_rng(seed).choice(len(files), samples, replace=False)
    rows = []
    for index in selected:
        path = files[index]
        name = path.relative_to(dataset).as_posix()
        original = preprocess_image(path)
        # Include relative path to avoid collisions in nested datasets.
        stem = Path(output) / "comparisons" / Path(name).with_suffix("")
        fig, axes = plt.subplots(1, 6, figsize=(12, 3))
        axes[0].imshow(original.reshape(80, 64), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original")
        for ax, size in zip(axes[1:], PAYLOAD_SIZES):
            payload = encode_array(original, bundle, size)
            binary_path = output_path(stem.parent / f"{stem.name}_{size}.bin")
            binary_path.write_bytes(payload)
            assert binary_path.stat().st_size == size
            # Exercise the saved binary independently of the original image.
            reconstruction = decode_payload(binary_path.read_bytes(), bundle)
            array_to_image(reconstruction).save(output_path(stem.parent / f"{stem.name}_{size}.png"))
            rows.append({"filename": name, "payload_bytes": size, **quality_metrics(original, reconstruction)})
            ax.imshow(reconstruction.reshape(80, 64), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{size} bytes")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(output_path(stem.with_suffix(".png")), dpi=150)
        plt.close(fig)
        print(f"Compared {name}", flush=True)
    fields = ["filename", "payload_bytes", "mse", "psnr", "ssim"]
    with output_path(Path(output) / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    averages = []
    for size in PAYLOAD_SIZES:
        group = [r for r in rows if r["payload_bytes"] == size]
        averages.append({"payload_bytes": size, **{key: float(np.mean([r[key] for r in group])) for key in ("mse", "psnr", "ssim")}})
    with output_path(Path(output) / "metrics_average.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields[1:])
        writer.writeheader()
        writer.writerows(averages)
    visualize_model(bundle, output)
    print(f"Saved comparisons and metrics in {output}")
    return rows, averages
