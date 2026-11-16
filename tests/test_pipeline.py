import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.compare import compare_images
from src.decode import decode_file, decode_payload
from src.encode import encode_array, encode_image
from src.model import PAYLOAD_SIZES, ROOT, load_model, output_path
from src.preprocess import array_to_image, preprocess_image
from src.train import train_model


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment")
    dataset = root / "dataset"
    dataset.mkdir()
    rng = np.random.default_rng(12)
    y, x = np.mgrid[:80, :64]
    basis = np.array([np.sin(x / 10), np.cos(y / 12), np.sin((x + y) / 15),
                      np.cos((x - y) / 18)], dtype=np.float32)
    for i in range(112):
        values = np.clip(0.5 + np.einsum("i,ihw->hw", rng.uniform(-0.12, 0.12, 4), basis), 0, 1)
        array_to_image(values).save(dataset / f"{i:04}.jpg")
    model = root / "model.joblib"
    bundle = train_model(dataset, model, samples=100, components=96, seed=9)
    return root, dataset, model, bundle


def test_preprocessing(tmp_path):
    path = tmp_path / "rgb.jpg"
    Image.new("RGB", (178, 218), (80, 150, 220)).save(path)
    values = preprocess_image(path)
    assert values.shape == (5120,)
    assert values.dtype == np.float32
    assert 0 <= values.min() <= values.max() <= 1
    image = array_to_image(values)
    assert image.mode == "L" and image.size == (64, 80)
    assert np.array(array_to_image(np.full(5120, 2))).min() == 255


@pytest.mark.parametrize("size", PAYLOAD_SIZES)
def test_round_trip(experiment, tmp_path, size):
    _, dataset, model, bundle = experiment
    image = dataset / "0000.jpg"
    binary = tmp_path / "face.bin"
    payload = encode_image(image, model, size, binary)
    assert len(payload) == binary.stat().st_size == size
    assert payload == encode_image(image, model, size)
    output = tmp_path / "face.png"
    result = decode_file(binary, model, output)
    assert result.shape == (5120,) and result.dtype == np.float32
    assert np.all(np.isfinite(result)) and result.min() >= 0 and result.max() <= 1
    assert Image.open(output).mode == "L"
    assert np.mean((result - preprocess_image(image)) ** 2) < 0.001
    np.testing.assert_allclose(result, decode_payload(payload, bundle), atol=3e-7, rtol=1e-6)


def test_shared_quantization(experiment):
    _, dataset, _, bundle = experiment
    original = preprocess_image(dataset / "0000.jpg")
    coefficients = bundle["pca"].transform(original[None])[0, :64]
    limits = bundle["metadata"]["quantization"]
    low = np.array(limits["minimum"][:64], dtype=np.float32)
    high = np.array(limits["maximum"][:64], dtype=np.float32)
    expected = np.rint(np.clip((coefficients - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    payload = encode_array(original, bundle, 64)
    assert payload == expected.tobytes()
    changed = copy.deepcopy(bundle)
    changed["metadata"]["quantization"]["minimum"] = [-1000] * 96
    changed["metadata"]["quantization"]["maximum"] = [1000] * 96
    assert payload != encode_array(original, changed, 64)
    # Calibration is exactly the transform of the selected training data.
    matrix = np.stack([preprocess_image(dataset / name) for name in bundle["metadata"]["training_files"]])
    training_coefficients = bundle["pca"].transform(matrix)
    np.testing.assert_allclose(training_coefficients.min(axis=0), limits["minimum"], atol=1e-6)
    np.testing.assert_allclose(training_coefficients.max(axis=0), limits["maximum"], atol=1e-6)


def test_clamping_and_constant_ranges(experiment):
    _, dataset, _, bundle = experiment
    changed = copy.deepcopy(bundle)
    coefficients = bundle["pca"].transform(preprocess_image(dataset / "0000.jpg")[None])[0]
    changed["metadata"]["quantization"]["minimum"] = (coefficients + 10).tolist()
    changed["metadata"]["quantization"]["maximum"] = (coefficients + 11).tolist()
    assert encode_array(preprocess_image(dataset / "0000.jpg"), changed, 32) == bytes(32)
    changed["metadata"]["quantization"]["minimum"] = (coefficients - 11).tolist()
    changed["metadata"]["quantization"]["maximum"] = (coefficients - 10).tolist()
    assert encode_array(preprocess_image(dataset / "0000.jpg"), changed, 32) == bytes([255]) * 32
    changed["metadata"]["quantization"]["maximum"] = changed["metadata"]["quantization"]["minimum"]
    assert encode_array(preprocess_image(dataset / "0000.jpg"), changed, 32) == bytes(32)
    assert np.isfinite(decode_payload(bytes(32), changed)).all()


@pytest.mark.parametrize("size", [0, 1, 31, 33, 97, 128, 256])
def test_invalid_sizes(experiment, size):
    bundle = experiment[3]
    with pytest.raises(ValueError, match="Payload size"):
        encode_array(np.zeros(5120), bundle, size)
    with pytest.raises(ValueError, match="Payload size"):
        decode_payload(bytes(size), bundle)


def test_model_capacity(experiment):
    bundle = copy.deepcopy(experiment[3])
    bundle["metadata"]["components"] = 32
    with pytest.raises(ValueError, match="model has 32"):
        decode_payload(bytes(64), bundle)


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'python main.py train'"):
        load_model(tmp_path / "missing.joblib")


def test_decode_without_dataset(experiment, tmp_path):
    _, dataset, model, _ = experiment
    payload = encode_image(dataset / "0000.jpg", model, 64)
    # New process receives only the model and binary file; training dataset is hidden.
    binary = tmp_path / "face.bin"
    binary.write_bytes(payload)
    hidden = dataset.with_name("hidden_dataset")
    dataset.rename(hidden)
    try:
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), "decode", str(binary),
                                 "--model", str(model), "--output", str(tmp_path / "decoded.png")],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    finally:
        hidden.rename(dataset)


def test_comparison_and_metadata(experiment, tmp_path):
    _, dataset, model, bundle = experiment
    rows, averages = compare_images(dataset, model, tmp_path, samples=2)
    assert len(rows) == 10 and len(averages) == 5
    assert not set(r["filename"] for r in rows) & set(bundle["metadata"]["training_files"])
    with (tmp_path / "metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 10
    assert (tmp_path / "average_face.png").is_file()
    assert (tmp_path / "eigenfaces.png").is_file()
    metadata = json.loads(model.with_suffix(".json").read_text())
    assert metadata["seed"] == 9 and metadata["samples"] == 100


def test_dataset_write_protection():
    with pytest.raises(ValueError, match="outside"):
        output_path(ROOT / "dataset" / "overwrite.png")


def test_invalid_training(experiment, tmp_path):
    dataset = experiment[1]
    with pytest.raises(ValueError, match="Training samples"):
        train_model(dataset, tmp_path / "model", samples=1000)
    with pytest.raises(ValueError, match="Components"):
        train_model(dataset, tmp_path / "model", samples=5, components=5)
