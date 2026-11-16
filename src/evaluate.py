"""Evaluate the existing shared model without training or changing calibration.

Run: python -m src.evaluate --samples 20 --seed 43
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from PIL import Image, ImageDraw

from .compare import quality_metrics
from .decode import decode_payload
from .encode import encode_array
from .model import DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_OUTPUT, PAYLOAD_SIZES, ROOT, find_images, load_model, validate_size
from .preprocess import array_to_image, preprocess_image


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_images(dataset, metadata, samples, seed):
    files = find_images(dataset)
    training = set(metadata.get("training_files", []))
    training_names = {Path(name).name for name in training}
    partitions = sorted(dataset.rglob("list_eval_partition.txt"))
    if partitions:
        partition = partitions[0]
        labels = dict(line.split() for line in partition.read_text().splitlines() if line.strip())
        candidates = [p for p in files if labels.get(p.name) == "2"]
        method = f"CelebA test/evaluation partition (label 2), {partition}"
    elif training:
        candidates = files
        method = "Saved training-selection metadata; no list_eval_partition.txt available"
    else:
        raise ValueError("No partition file or training-selection metadata; unseen status cannot be established")
    # Also exclude basenames, conservatively covering a moved dataset root.
    candidates = [p for p in candidates if p.relative_to(dataset).as_posix() not in training and p.name not in training_names]
    if len(candidates) < samples:
        raise ValueError(f"Only {len(candidates)} eligible images for {samples} requested")
    selected = [candidates[i] for i in np.random.default_rng(seed).choice(len(candidates), samples, replace=False)]
    return selected, {"method": method, "candidate_count": len(candidates), "dataset_images": len(files),
                      "seed": seed, "partition_file": str(partitions[0]) if partitions else None,
                      "training_path_overlap": 0, "training_basename_overlap": 0}


def sheet(path, panels, labels, columns, title):
    # Identical integer enlargement of every native 64x80 panel; fixed [0,1]
    # conversion, no per-panel contrast normalization or interpolation smoothing.
    scale, width, height, header = 3, 64 * 3, 80 * 3, 28
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, header + rows * (height + header)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="black")
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (i % columns) * width, header + (i // columns) * (height + header)
        draw.text((x + 5, y + 6), label, fill="black")
        image = array_to_image(panel).resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y + header))
    canvas.save(path)


def isolated_decode(model_path, payload_paths, expected, output, dataset):
    """Run existing decoder in another process; audit denies every image read
    and every dataset/output read. Only copied model and payloads are supplied.
    The audit hook is a dependency check, not an OS security sandbox.
    """
    with tempfile.TemporaryDirectory(prefix="eigenfaces_evaluation_") as directory:
        folder = Path(directory).resolve()
        shutil.copy2(model_path, folder / "model.joblib")
        for i, path in enumerate(payload_paths):
            shutil.copy2(path, folder / f"{i}.bin")
        code = '''
import json, sys
from pathlib import Path
import numpy as np
from src.model import load_model
from src.decode import decode_file
folder = Path(sys.argv[1]).resolve()
forbidden = [Path(p).resolve() for p in json.loads(sys.argv[2])]
opened = []
def audit(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = Path(args[0]).resolve()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"} or any(path.is_relative_to(p) for p in forbidden):
        raise RuntimeError(f"Decoder attempted forbidden source access: {path}")
    opened.append(str(path))
sys.addaudithook(audit)
for i in range(int(sys.argv[3])):
    values = decode_file(folder / f"{i}.bin", folder / "model.joblib")
    np.save(folder / f"{i}.npy", values)
(folder / "opened.json").write_text(json.dumps(opened))
'''
        result = subprocess.run([sys.executable, "-c", code, str(folder),
                                 json.dumps([str(dataset.resolve()), str(output.resolve())]),
                                 str(len(payload_paths))], cwd=ROOT, capture_output=True, text=True, check=True)
        maximum_difference = max(float(np.max(np.abs(np.load(folder / f"{i}.npy") - values))) for i, values in enumerate(expected))
        assert maximum_difference == 0, maximum_difference
        opened = json.loads((folder / "opened.json").read_text())
        return {"payloads_checked": len(expected), "max_absolute_difference": maximum_difference,
                "source_image_reads_forbidden": True, "dataset_and_evaluation_output_reads_forbidden": True,
                "method": "Separate Python process; copies of model and binaries only; Python open audit hook",
                "opened_paths": sorted(set(p for p in opened if not p.startswith(str(folder)))),
                "subprocess_stderr": result.stderr}


def evaluate(dataset=DEFAULT_DATASET, model_path=DEFAULT_MODEL, output=DEFAULT_OUTPUT / "evaluation", samples=20, seed=43):
    dataset, model_path, output = Path(dataset), Path(model_path), Path(output)
    # Missing model stops here; this module has no training entry point.
    before_hash = digest(model_path) if model_path.is_file() else None
    bundle = load_model(model_path)
    metadata, pca = bundle["metadata"], bundle["pca"]
    for size in PAYLOAD_SIZES:
        validate_size(size, bundle)
    if samples < 2:
        raise ValueError("At least two images are required for sample standard deviations")
    assert metadata["samples"] == pca.n_samples_ == len(metadata["training_files"])
    selected, selection = select_images(dataset, metadata, samples, seed)
    for name in ("comparisons", "contact_sheets", "payloads", "reconstructions", "graphs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    low = np.asarray(metadata["quantization"]["minimum"], dtype=np.float32)
    high = np.asarray(metadata["quantization"]["maximum"], dtype=np.float32)
    calibration_before = json.dumps(metadata["quantization"], sort_keys=True)
    originals, reconstructed, floating, coefficients, codes = [], {n: [] for n in PAYLOAD_SIZES}, {n: [] for n in PAYLOAD_SIZES}, [], []
    rows, manifest, payload_paths, expected = [], [], [], []
    transform_difference = 0.0
    for path in selected:
        name = path.relative_to(dataset).as_posix()
        original = preprocess_image(path)
        coefficient = pca.transform(original.reshape(1, -1))[0]
        direct = (original - pca.mean_) @ pca.components_.T
        transform_difference = max(transform_difference, float(np.max(np.abs(coefficient - direct))))
        coefficients.append(coefficient)
        originals.append(original)
        tag = f"{path.stem}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"
        manifest.append({"filename": name, "source_sha256": digest(path), "artifact_tag": tag})
        array_to_image(original).save(output / "reconstructions" / f"{tag}_original.png")
        for size in PAYLOAD_SIZES:
            payload = encode_array(original, bundle, size)
            target = output / "payloads" / f"{tag}_{size}.bin"
            target.write_bytes(payload)
            assert len(payload) == target.stat().st_size == size
            assert encode_array(original, bundle, size) == payload
            # A larger payload must reuse the same shared leading calibration.
            assert encode_array(original, bundle, 96)[:size] == payload
            reconstruction = decode_payload(target.read_bytes(), bundle)
            float_reconstruction = np.clip(coefficient[:size] @ pca.components_[:size] + pca.mean_, 0, 1).astype(np.float32)
            qmetrics, fmetrics = quality_metrics(original, reconstruction), quality_metrics(original, float_reconstruction)
            rows.append({"filename": name, "payload_bytes": size, **qmetrics,
                         **{f"float_{k}": v for k, v in fmetrics.items()},
                         "quantization_mse_delta": qmetrics["mse"] - fmetrics["mse"],
                         "quantized_vs_float_mse": float(np.mean((reconstruction - float_reconstruction)**2))})
            reconstructed[size].append(reconstruction)
            floating[size].append(float_reconstruction)
            payload_paths.append(target)
            expected.append(reconstruction)
            array_to_image(reconstruction).save(output / "reconstructions" / f"{tag}_{size}.png")
            if size == 96:
                codes.append(np.frombuffer(payload, dtype=np.uint8))
        sheet(output / "comparisons" / f"{tag}.png", [original] + [reconstructed[n][-1] for n in PAYLOAD_SIZES],
              ["Original"] + [f"{n} bytes" for n in PAYLOAD_SIZES], 6, name)
        print(f"Evaluated {name}", flush=True)
    coefficient_matrix, code_matrix = np.asarray(coefficients)[:, :96], np.asarray(codes)
    range_rows = [{"component": j + 1, "minimum": float(low[j]), "maximum": float(high[j]),
                   "step": float((high[j] - low[j]) / 255)} for j in range(metadata["components"])]
    write_csv(output / "coefficient_ranges.csv", range_rows)
    saturation = []
    for j in range(96):
        saturation.append({"component": j + 1, "images": samples,
                           "code_0_count": int(np.sum(code_matrix[:, j] == 0)),
                           "code_255_count": int(np.sum(code_matrix[:, j] == 255)),
                           "endpoint_rate": float(np.mean((code_matrix[:, j] == 0) | (code_matrix[:, j] == 255))),
                           "below_range_count": int(np.sum(coefficient_matrix[:, j] < low[j])),
                           "above_range_count": int(np.sum(coefficient_matrix[:, j] > high[j])),
                           "clipping_rate": float(np.mean((coefficient_matrix[:, j] < low[j]) | (coefficient_matrix[:, j] > high[j])))})
    write_csv(output / "saturation_by_coefficient.csv", saturation)
    groups = []
    for start, end in ((0, 32), (32, 48), (48, 64), (64, 80), (80, 96)):
        groups.append({"components": f"{start + 1}-{end}", "observations": samples * (end - start),
                       "endpoint_count": sum(r["code_0_count"] + r["code_255_count"] for r in saturation[start:end]),
                       "endpoint_rate": float(np.mean([r["endpoint_rate"] for r in saturation[start:end]])),
                       "clipping_rate": float(np.mean([r["clipping_rate"] for r in saturation[start:end]]))})
    write_csv(output / "saturation_by_group.csv", groups)
    write_csv(output / "metrics.csv", rows)
    metric_names = ("mse", "psnr", "ssim", "float_mse", "float_psnr", "float_ssim", "quantization_mse_delta", "quantized_vs_float_mse")
    summary = []
    for size in PAYLOAD_SIZES:
        group = [r for r in rows if r["payload_bytes"] == size]
        summary.append({"payload_bytes": size, "images": samples,
                        **{f"{key}_{stat}": float(function([r[key] for r in group]))
                           for key in metric_names for stat, function in (("mean", np.mean), ("std", lambda a: np.std(a, ddof=1)))}})
        sheet(output / "contact_sheets" / f"faces_{size}_bytes.png", reconstructed[size],
              [p.name for p in selected], 5, f"{size} bytes: same scale for every face")
    sheet(output / "contact_sheets" / "originals.png", originals, [p.name for p in selected], 5, "Preprocessed originals")
    # Direct same-component diagnostic for the best, median, and worst SSIM at 64 bytes.
    ranked = sorted(range(samples), key=lambda i: rows[i * 5 + 2]["ssim"])
    diagnostic_indices = [ranked[-1], ranked[len(ranked)//2], ranked[0]]
    panels, labels = [], []
    for i in diagnostic_indices:
        panels.extend([originals[i], reconstructed[64][i], floating[64][i]])
        labels.extend([f"{selected[i].name} original", "64 B quantized", "64 PCs float"])
    sheet(output / "comparisons" / "quantization_diagnostic.png", panels, labels, 3, "Best / median / worst SSIM at 64 bytes")
    write_csv(output / "summary.csv", summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("mse", "psnr", "ssim")):
        ax.errorbar(PAYLOAD_SIZES, [r[f"{metric}_mean"] for r in summary],
                    yerr=[r[f"{metric}_std"] for r in summary], marker="o", capsize=3, label="Quantized mean ± sample SD")
        ax.plot(PAYLOAD_SIZES, [r[f"float_{metric}_mean"] for r in summary], "x--", label="Floating PCA mean")
        ax.set(xlabel="Payload bytes / leading components", ylabel=metric.upper(), xticks=PAYLOAD_SIZES)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "graphs" / "quality_vs_payload.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("mse", "psnr", "ssim")):
        for i in range(samples):
            ax.plot(PAYLOAD_SIZES, [rows[i * 5 + j][metric] for j in range(5)], alpha=.4, linewidth=1)
        ax.set(xlabel="Payload bytes", ylabel=metric.upper(), xticks=PAYLOAD_SIZES)
        ax.grid(alpha=.25)
    fig.suptitle("Individual held-out image trajectories")
    fig.tight_layout()
    fig.savefig(output / "graphs" / "quality_by_image.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(PAYLOAD_SIZES, [100*r["quantization_mse_delta_mean"]/r["float_mse_mean"] for r in summary], "o-")
    ax.set(xlabel="Payload bytes", ylabel="Mean MSE increase (%)", xticks=PAYLOAD_SIZES,
           title="Quantized vs. same-component floating PCA")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "graphs" / "quantization_penalty.png", dpi=180)
    plt.close(fig)
    print("Checking all payloads in an isolated decoder process...", flush=True)
    isolation = isolated_decode(model_path, payload_paths, expected, output, dataset)
    # Also verify sklearn's transform against the basic stored PCA projection.
    assert transform_difference < 1e-4, transform_difference
    assert calibration_before == json.dumps(metadata["quantization"], sort_keys=True)
    assert digest(model_path) == before_hash
    checks = {"model_sha256_before": before_hash, "model_sha256_after": digest(model_path),
              "model_unchanged": True, "model_samples": int(pca.n_samples_),
              "components": metadata["components"], "image_size": metadata["image_size"],
              "total_explained_variance": float(pca.explained_variance_ratio_.sum()),
              "trained_sklearn": metadata["software_versions"]["scikit_learn"], "evaluation_sklearn": sklearn.__version__,
              "max_transform_difference_from_direct_projection": transform_difference,
              "payload_count": len(payload_paths), "all_payload_lengths_correct": True, "shared_quantization_unchanged": True,
              "deterministic_encoding_and_payload_prefixes": True, "isolated_decoder": isolation,
              "selection": selection, "selected_images": manifest,
              "explained_variance_by_payload": {str(n): float(pca.explained_variance_ratio_[:n].sum()) for n in PAYLOAD_SIZES}}
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation in {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", dest="model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "evaluation")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=43)
    evaluate(**vars(parser.parse_args()))
