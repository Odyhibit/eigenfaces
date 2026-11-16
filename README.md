# Eigenfaces — Compact Facial Image Encoding

An educational, noncommercial Python experiment: represent a grayscale face in
32–96 bytes using principal component analysis (PCA) and a shared model. QR code
generation is outside this phase. These tiny payloads measure a tradeoff between
size and reconstruction quality; they are not identity verification or encryption.

## Installation

Use Python 3.10–3.12 with the provided dependency ranges. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

NumPy, Pillow, scikit-learn, matplotlib, joblib, and scikit-image provide the CPU
pipeline; pytest runs the synthetic tests. No GPU or deep learning framework is used.

## Dataset

Obtain the aligned and cropped CelebA JPEGs separately from the
[official CelebA project](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Place them in `dataset/` or `dataset/img_align_celeba/`. Discovery is recursive;
original images are read only. Dataset, models, and generated outputs are ignored
by Git. Do not commit or redistribute the dataset or generated face artifacts.

The default paths are anchored to this project's directory, including when invoking
`main.py` from another working directory. Explicit relative paths are resolved from
the current working directory. `--dataset`, `--model`, and `--output` override paths.

## Training

```bash
python main.py train --samples 5000 --components 128 --seed 42
```

Training samples are selected without replacement from sorted JPEG paths using
the seed. Every image is converted to grayscale, resized to 64×80 (width×height)
with LANCZOS, flattened in row order, and normalized to float32 values in [0, 1].
The encoder uses exactly this same preprocessing function.

Randomized PCA learns a mean face and basis. The default image matrix occupies
about 98 MiB (5000×5120×4 bytes); PCA requires additional working memory. Training
prints progress and cumulative explained variance. The number of components must
be at most `min(samples - 1, 5120)`; comparisons at every payload size require at
least 96 components. Seeds make sampling and randomized PCA repeatable within the
same numerical software environment; cross-version bit identity is not promised.

`models/eigenfaces.joblib` contains both PCA and metadata. The adjacent JSON file
is a human-readable copy, including dimensions, normalization, training file list,
seed, solver, explained variance, and per-component quantization ranges. The
joblib bundle is authoritative; it is sufficient for encoding and decoding.
Only load trusted joblib files, since deserialization can execute Python code.

## Encoding and decoding

```bash
python main.py encode dataset/img_align_celeba/000001.jpg --bytes 64 --output output/face.bin
python main.py decode output/face.bin --output output/reconstructed.png
```

Supported payload sizes are exactly 32, 48, 64, 80, and 96 bytes, provided the model
has enough components. A payload contains one unsigned byte per leading PCA
coefficient, with no header. Its length determines the coefficient count.

For component j, training coefficients define fixed minimum `lo[j]` and maximum
`hi[j]`. Encoding uses `round(255 * clip((c[j]-lo[j])/(hi[j]-lo[j]), 0, 1))`.
Decoding uses `lo[j] + q[j]/255 * (hi[j]-lo[j])`. A zero-width range encodes zero
and decodes to its constant minimum. Rounding uses NumPy's nearest-even rule.
All ranges are calibrated once on training data and stored in the shared model;
no scale or range is estimated from an input photograph. Outliers are clamped.

The decoder sums the selected components weighted by their decoded coefficients,
adds the mean face, clips to [0, 1], and saves a grayscale PNG. Omitted coefficients
are zero. It requires neither the original photo nor the training images.

The receiver must have the **exact same model and format version (currently 1)**,
supplied externally. Payloads contain no model ID, checksum, or error correction;
using a different model can silently reconstruct the wrong face. The shared model
is much larger than a payload and is excluded from the quoted byte counts.

Reusable Python entry points are `preprocess_image`, `train_model`, `encode_image`,
`encode_array`, `decode_file`, `decode_payload`, and `compare_images` in `src/`.
The array functions accept an already-loaded model bundle to avoid repeated I/O.

## Comparisons

```bash
python main.py compare --samples 10 --seed 43
```

Comparison excludes saved training relative paths, then selects held-out images
reproducibly. Use the same dataset tree used for training; moving its root is fine,
but renaming files or supplying duplicate images under new names defeats this
path-based exclusion. This is an image holdout, not a person-identity holdout.

Outputs in `output/`:

- `comparisons/`: one labeled six-panel sheet per image (original and all five
  payload sizes), exact-size `.bin` files, and reconstructed grayscale PNGs.
- `average_face.png` and `eigenfaces.png`: the mean and first 16 components.
  Each eigenface uses its own symmetric contrast scale for visibility.
- `metrics.csv`: filename, payload bytes, MSE, PSNR, and SSIM per reconstruction.
- `metrics_average.csv`: arithmetic mean of each metric across the selected
  images for each payload size, including mean per-image PSNR in decibels.

Metrics compare floating-point reconstructions before PNG rounding with the
preprocessed 64×80 original, using a pixel range of 1. Lower MSE and higher PSNR
and SSIM indicate closer reconstructions. An exact match has infinite PSNR.
These metrics measure pixel/structure similarity, not definitive recognizability.
Increasing payload size usually improves quality, but quantization and clipping
mean individual metrics need not improve monotonically.

## Testing

For the extended model evaluation (20 held-out images, sample standard deviations,
contact sheets, quality graphs, saturation diagnostics, same-component floating
PCA baselines, and decoding in a separate process with image reads blocked):

```bash
python -m src.evaluate --samples 20 --seed 43
```

Outputs are saved under `output/evaluation/`. The utility prefers CelebA's test
partition when `dataset/**/list_eval_partition.txt` exists and otherwise excludes
saved training filenames. `verification.json` records the selection, source and
model hashes, decoder checks, and software versions. It never trains a model.
The accompanying `report.md` documents the reviewed evaluation run.

```bash
python -m pytest -q
```

Tests build a small synthetic dataset without CelebA. They check dimensions,
payload lengths, deterministic encoding, reconstruction accuracy and ranges,
shared calibration, saturation and constant ranges, invalid inputs, missing
models, training exclusion, reports, and decoding in a separate process while
the synthetic training dataset is unavailable.

## Limitations

PCA captures linear variation across similarly aligned faces. It loses fine
detail, color, pose, expression, and high-frequency texture. Alignment or lighting
changes and faces outside the training distribution can cause poor reconstruction.
Min/max quantization is simple and reproducible, but extreme training coefficients
can waste precision; unseen extremes saturate. More coefficients also consume more
bytes. The resizing discards information before PCA compression begins.

This project does not perform face detection or alignment: new inputs should be
aligned and cropped comparably to the training images. It provides no recognition
accuracy claims, privacy protection, or guarantee that a reconstruction preserves
identity. Dataset composition affects the learned basis and resulting quality.

## CelebA attribution and usage

CelebA was created by Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang at the
Multimedia Laboratory, The Chinese University of Hong Kong. Cite:

> Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. *Deep Learning Face
> Attributes in the Wild*. Proceedings of ICCV, 2015.

The [official dataset agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
limits CelebA to noncommercial research, restricts redistribution, and restricts
commercial exploitation of images and derived data. The dataset is not included
in this source project. Source code is MIT licensed (see `LICENSE`); that license
does not grant rights to CelebA, trained models, or generated face artifacts.
Review those terms before sharing trained models or generated comparisons. This
is an educational, noncommercial portfolio experiment.
