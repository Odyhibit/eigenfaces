# Eigenfaces — Compact Facial Image Encoding

An educational, noncommercial Python experiment: represent a grayscale face in
32–96 bytes using principal component analysis (PCA) and a shared model. These tiny payloads measure a tradeoff between
size and reconstruction quality; they are not identity verification or encryption.

## Installation

Use Python 3.10–3.12 with the provided dependency ranges. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

NumPy, Pillow, scikit-learn, matplotlib, joblib, and scikit-image provide the CPU
PCA pipeline; pytest runs the synthetic tests. The optional neural commands use
PyTorch and tqdm. PCA remains a separate codec with unchanged model weights.

## Dataset

Obtain the aligned and cropped CelebA JPEGs separately from the
[official CelebA project](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 
original images are read only. Dataset, models, and generated outputs are ignored
by Git. Do not commit or redistribute the dataset or generated face artifacts.

Explicit relative paths are resolved from
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

## Phase 2: convolutional neural codec

The neural codec coexists with PCA. It does not train, recalibrate, or overwrite
the PCA model. It reuses `src/preprocess.py`: grayscale, LANCZOS resize to **64x80
(width x height)**, float32 pixels divided by 255. No image augmentation or
pretrained recognition/identity model is used. Both codecs require aligned faces.

### CUDA setup and verification (Windows PowerShell)

Activate the existing environment before using `python`; on this machine the
unqualified global interpreter is different from `.venv` and has no PyTorch.
For a new environment, Python 3.11 or 3.12 fits the existing dependency ranges.
The existing Python 3.13 environment was also tested for this phase.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print(torch.cuda.mem_get_info() if torch.cuda.is_available() else '')"
```

Use the [official PyTorch installer](https://pytorch.org/get-started/locally/)
or [version-specific commands](https://pytorch.org/get-started/previous-versions/)
for other CUDA/CPU builds. Install the CUDA build explicitly before the generic
requirements file. A CPU build satisfies the `torch` requirement but cannot train
on the GPU. TorchVision and TorchAudio are unnecessary.

Verified hardware: RTX 4080, PyTorch 2.11.0+cu128, CUDA available, 15.99 GiB total
and about 14.72 GiB free VRAM at inspection. Free memory varies with other apps.
`--device auto` chooses CUDA when available and falls back to CPU. Explicit
`--device cuda` fails clearly if unavailable. Inference defaults to CPU.

### Architecture and quantization

Tensor layout is NCHW; spatial shapes below are height x width.

| Stage | Output per image |
|---|---|
| Input | 1 x 80 x 64 |
| Conv 4x4, stride 2, padding 1 + ReLU | 16 x 40 x 32 |
| Conv 4x4, stride 2, padding 1 + ReLU | 32 x 20 x 16 |
| Conv 4x4, stride 2, padding 1 + ReLU | 64 x 10 x 8 |
| Flatten, linear, sigmoid | L bounded values |
| Round to uint8, dequantize | L values from the 256-level grid |
| Linear + ReLU, reshape | 64 x 10 x 8 |
| Transposed conv 4x4, stride 2, padding 1 + ReLU | 32 x 20 x 16 |
| Transposed conv 4x4, stride 2, padding 1 + ReLU | 16 x 40 x 32 |
| Transposed conv 4x4, stride 2, padding 1 + sigmoid | 1 x 80 x 64 |

There are no skip connections, image-specific scales, or other side channels.
The decoder's only image-specific input is the quantized latent. Architecture
`cae-v1` has `87,713 + 10,241 * L` trainable parameters: **743,137 at L=64**
(369,072 encoder, 374,065 decoder). Supported L values are 32, 48, 64, 80, 96.
Each dimension requires its own trained model; truncation is invalid.

Encoding computes `q = round(255 * clamp(sigmoid_output, 0, 1))` in float32,
using nearest-even rounding, and writes exactly L unsigned bytes with **no
header**. Decoding uses `q.float() / 255`. During training, a straight-through
estimator feeds these exact dequantized byte values to the decoder in the
forward pass while using an identity gradient for the quantizer in backward.
Thus the codec's quantization grid matches training. Quantization stays float32
even when optional CUDA autocast runs convolutions in mixed precision.

The receiver needs the exact decoder checkpoint externally. There is no model
identifier, checksum, error correction, or QR wrapper in the payload. Shared
weights are excluded from byte counts. Evaluation is deterministic for the same
model, device and software environment; bit identity across CPU/GPU, AMP, or
library versions is not guaranteed. There is no stochastic inference layer.

### First training run

No full CelebA neural training run has been started as part of implementation.
On this machine, the existing PCA default points to the dataset on D:. To use a
local `dataset/` tree instead, change only the `--dataset` argument.

```powershell
python main.py neural-train --dataset "D:\eigenfaces\img_align_celeba" --output models/neural64 --samples 20000 --validation-samples 2000 --latent 64 --epochs 50 --batch-size 128 --learning-rate 0.001 --seed 42 --device cuda --workers 4 --amp
```

Training uses AdamW, weighted L1 (default 1.0) plus differentiable local SSIM loss
(default 0.2). Adjust with `--l1-weight` and `--ssim-weight`; zero disables a term.
Training SSIM uses uniform valid 7x7 windows and population covariance; evaluation
uses the existing skimage SSIM function so metrics match PCA. Validation loss
selects the best model; `--patience 10` enables early stopping after ten epochs
without improvement. `--epochs` is the maximum total epoch count.

Gradient clipping is enabled by default: `--max-grad-norm 1.0` limits the global
L2 gradient norm before each optimizer step, after AMP gradients are unscaled.
Use `--max-grad-norm 0` to disable it. The reported gradient norms are measured
before clipping; clipping frequency is the fraction of training batches clipped.
This limits large gradients but does **not** guarantee sigmoid saturation cannot
occur, especially with an adaptive optimizer.

Every epoch prints validation latent min/max, the fraction near sigmoid boundaries
(`z <= 0.01` or `z >= 0.99`), the fraction quantized to byte 0 or 255, the mean
per-coordinate standard deviation between images, and the number of coordinates
whose quantized value is constant across validation images. These statistics are
computed from actual forward-pass latents with no extra encoder pass. Both train
and validation statistics are saved in `history.csv`; training statistics span
an evolving model, while validation measures one fixed model. CSV rates are in
[0,1], and the standard deviation uses population variance across images.

A console warning appears when at least 95% of validation latent values are near
the boundaries, or all validation payloads are identical (with at least two
validation images). It reports a possible collapse and leaves stopping decisions
to the user and ordinary validation early stopping. Boundary occupancy alone is
not proof of collapse; inspect variation and reconstructions too. An initial run
was found to emit all-255 latents and identical reconstructions despite valid
payloads, motivating these diagnostics. Clipping cannot revive an encoder whose
sigmoid gradients have already vanished.

For the next experiment, use fresh weights, a 10x lower learning rate, no AMP,
and the existing split. Replace `YOUR_SSD_DATASET_PATH` with the new dataset root,
keeping relative image paths unchanged after the SSD transfer:

```powershell
.\.venv\Scripts\python.exe main.py neural-train --dataset "YOUR_SSD_DATASET_PATH" --output models/neural64_retry --split-manifest models/neural64/split.json --latent 64 --epochs 50 --batch-size 128 --learning-rate 0.0001 --max-grad-norm 1.0 --seed 42 --device cuda --workers 4
```

The trainer discovers `list_eval_partition.txt` recursively under the supplied
dataset, or accepts `--partition path/to/list_eval_partition.txt`. It uses labels
0/1/2 for train/validation/test. Without this file it saves a deterministic split:
10% test (excluding PCA training filenames), the requested validation count, and
the requested training subset. All splits are disjoint by filename. **This does
not guarantee identity-level separation or detect duplicate image content.**
PCA training images may be used for neural training, but never for shared test.
An existing `output/evaluation/verification.json` reserves prior PCA evaluation
images before splitting; `--evaluation-manifest` overrides its location. An
incompatible prior evaluation causes a clear error rather than test leakage.

For other budgets, reuse `--split-manifest models/neural64/split.json`, set
`--latent 32` (or 48, 80, 96), and choose a separate output directory such as
`models/neural32`. A supplied manifest determines image counts, overriding
`--samples` and `--validation-samples`. This keeps later experiments comparable.

DataLoaders default to up to four workers, pin memory on CUDA, and keep workers
persistent. All entry points have Windows multiprocessing guards. Set
`--workers 0` when diagnosing worker issues. CPU training is supported with
`--device cpu`; omit `--amp` there. Training on the 4080 was smoke-tested with
two workers and AMP, without using CelebA.

Outputs under the selected run directory:

- `best.pt`: best encoder+decoder weights, versioned configuration and split provenance.
- `decoder.pt`: matching best decoder only; no encoder, optimizer, or filenames.
- `last.pt`: latest weights, optimizer/scaler state, epoch, best weights, history,
  split provenance, and random generator states for epoch-boundary resume.
- `config.json`, `split.json`: model/training configuration and exact split filenames.
- `history.csv`, `training_loss.png`: sample-weighted train/validation total,
  L1 and SSIM losses, latent statistics and gradient clipping statistics by epoch,
  and total loss curves. Gradient statistics are averaged per optimizer attempt.

Resume from **last.pt**, not best.pt. Repeat the latent, batch size, learning
rate, seed, loss weights, and AMP setting from the original run; these are
validated. Saved splits are reused, and `--epochs` specifies the total target.
Resume can use a different output directory and preserves the previous best.
Fresh training refuses to overwrite an existing run with checkpoints.
Clipping can be changed on resume. Older checkpoints without clipping settings or
monitoring columns remain loadable; monitoring cells for those older epochs stay
empty in the CSV because the measurements were not collected at that time.

```powershell
python main.py neural-train --dataset "D:\eigenfaces\img_align_celeba" --output models/neural64 --latent 64 --epochs 50 --batch-size 128 --seed 42 --device cuda --workers 4 --amp --resume models/neural64/last.pt
```

### Encode and independently decode

```powershell
python main.py neural-encode "D:\eigenfaces\img_align_celeba\img_align_celeba\000001.jpg" --model models/neural64/best.pt --output output/neural_face.bin --device cpu
python main.py neural-decode output/neural_face.bin --model models/neural64/decoder.pt --output output/neural_face.png --device cpu
python main.py neural-export-decoder --model models/neural64/best.pt --output output/decoder.pt
```

The first example uses the nested image directory found on this machine; use the
actual source path if your extraction differs. A 64-dimensional model writes
exactly **64 bytes**. Decoder payload length and checkpoint configuration are
validated. Decoding requires only the decoder weights, payload, installed code
and dependencies; no original image, encoder network, split file, or dataset.
The output PNG is grayscale 64x80. Only load trusted checkpoints; neural loading
uses `torch.load(..., weights_only=True)` and strict weight-shape validation.

### Compare against the existing PCA model

```powershell
python main.py neural-compare --dataset "D:\eigenfaces\img_align_celeba" --models models/neural64/best.pt --samples 20 --device cpu
python main.py neural-compare --dataset "D:\eigenfaces\img_align_celeba" --models models/neural32/best.pt models/neural48/best.pt models/neural64/best.pt models/neural96/best.pt --samples 20
```

Only supply models that have actually been trained. The evaluator checks shared
test filenames against PCA training and every supplied neural train/validation
split. It reuses saved PCA evaluation images when available; otherwise it draws
deterministically from the shared neural test set and evaluates both codecs on
those identical images. It never fits PCA. The original PCA model hash is checked
before/after. Defaults can be changed with `--pca-model`, `--evaluation-manifest`,
`--output`, and `--seed`.

`output/neural_evaluation/` contains per-image three-column Original/PCA/Neural
sheets for each available byte budget, `.bin` and `.png` reconstructions,
`metrics.csv`, aggregate `summary.csv` (mean and sample standard deviation),
`quality.png`, `verification.json` (models, hashes and selected filenames), and
`report.md`. MSE, PSNR and SSIM are the same existing PCA metrics, computed on
float reconstructions before PNG rounding. Training loss graphs remain in each
model directory. No 96-byte result is emitted until a 96-dimensional model exists.

Higher SSIM need not mean a more recognizable face. A short synthetic test proves
the pipeline works; it provides no evidence that the neural codec outperforms PCA
on CelebA. Assess held-out comparisons after full training. Small latents may
collapse toward average faces; L1/SSIM can blur fine features. Future work may
explore architecture/loss tuning, identity-disjoint splits and human assessment,
learned rate allocation, or a separately specified QR transport format.

### Neural tests and synthetic smoke run

```powershell
python -m pytest -q
python scripts/neural_smoke.py --device cuda --workers 2
```

Tests use synthetic images and isolated temporary models. They cover all five
latent sizes, exact byte lengths, STE forward values and gradients, deterministic
inference, strict checkpoint compatibility, training/resume, saved splits,
PCA comparison artifacts, CPU fallback, and a CUDA AMP optimizer step when CUDA
is available. They also verify clipping after AMP unscaling, disabled clipping,
streaming latent statistics, warnings for forced collapse, and legacy resume
with missing monitoring columns. A separate decoder process is tested with original images hidden,
the full checkpoint deleted, dataset reads blocked and encoder construction
disabled. The smoke script trains one epoch on 16 synthetic images, deletes
their temporary dataset, and decodes on CPU in a new process. Its artifacts are
saved under `output/neural_smoke/run_*/`. It never accesses the real PCA model.

Dataset images, generated artifacts and weights remain ignored by Git. CelebA's
noncommercial research restrictions also apply to this educational neural
experiment; the MIT code license does not override the dataset agreement above.

## Phase 5: spatial VQ-VAE codec

This is a separate codec in `src/vq_*.py`. Existing PCA, scalar-latent neural
codecs, their checkpoints and earlier reports remain intact. No full CelebA
VQ-VAE training has been started during implementation. Improved visual quality
is an experimental goal, not an established result.

### Architecture and 64-byte format

`vqface-v1` has **1,030,625 trainable parameters**: encoder 517,408; codebook
16,384; decoder 496,833. The codebook has 256 entries of 64 float values each.
Residual blocks contain ReLU, a padded 3x3 convolution, ReLU, a 1x1 convolution,
and an internal residual addition. These additions never bypass the bottleneck.

| Stage | Channels x height x width |
|---|---|
| Grayscale input, unchanged shared preprocessing | 1 x 80 x 64 |
| Conv 4x4, stride 2, padding 1 + ReLU | 32 x 40 x 32 |
| Conv 4x4, stride 2, padding 1 + ReLU | 64 x 20 x 16 |
| Conv 4x4, stride 2, padding 1 | 128 x 10 x 8 |
| Two residual blocks + ReLU | 128 x 10 x 8 |
| Valid learned 3x1 convolution | 64 x 8 x 8 |
| Nearest codebook entry at each spatial position | 8 x 8 indices |
| Retrieve embeddings; transposed 3x1 convolution | 128 x 10 x 8 |
| Two residual blocks + ReLU | 128 x 10 x 8 |
| Bilinear 2x upsample, conv 3x3, ReLU, residual block | 64 x 20 x 16 |
| Bilinear 2x upsample, conv 3x3, ReLU, residual block | 32 x 40 x 32 |
| Bilinear 2x upsample, conv 3x3, sigmoid | 1 x 80 x 64 |

The 3x1 projection handles the rectangular input explicitly. Its eight outputs
cover all ten incoming feature rows. The decoder's transposed 3x1 layer restores
ten rows before upsampling. No source-image crop, square resize, extra image
padding, or encoder-to-decoder skip is used. Ordinary convolution padding keeps
the documented feature sizes.

For each spatial position the encoder supplies a 64-value vector. Squared
Euclidean nearest-neighbor search selects one of 256 embeddings, computed in
float32 even with AMP. Exact distance ties select the lowest entry index. The
decoder receives the selected embeddings in forward computation; the
straight-through estimator copies reconstruction gradients to encoder features.

This implementation uses the **standard embedding and commitment losses**, not
EMA updates. The embedding loss moves selected codebook entries toward detached
encoder vectors; the commitment loss moves encoder vectors toward detached
entries. This keeps updates explicit and covered by ordinary AdamW checkpoint
state. It follows the basic learning approach in the
[original VQ-VAE paper](https://arxiv.org/html/1711.00937v2#S3.SS2).
There is no discriminator, pretrained face-recognition loss, generative prior,
or entropy coding in this experiment.

The payload is exactly 64 unsigned bytes: grid row 0 left-to-right, then row 1,
through row 7. Byte offset is `8*y + x`. Every byte is an index in [0,255]. There
is no header, scale, image size, embedding vector, checksum, or model ID. The
receiver needs the matching codebook and decoder checkpoint externally. Shared
weights are excluded from byte budgets. Different models may assign entirely
different meanings to the same indices. Same-model evaluation is deterministic
on a fixed device/software environment; CPU/GPU/AMP bit identity is not promised.

### CUDA and training

The existing dependencies already include everything required; no new dependency
setup replaces the current one. CUDA setup is described above. Verify using the
project interpreter:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The implementation was tested with PyTorch 2.11.0+cu128 and the RTX 4080. A
batch-128, two-step synthetic probe including initialized AdamW moments measured
approximately **0.42 GiB peak allocated / 0.49 GiB reserved** without AMP, and
**0.28 GiB allocated / 0.43 GiB reserved** with AMP. These are PyTorch allocator
measurements, excluding display usage, CUDA context and other driver allocations;
they are approximate, not a full-run memory guarantee.

Full training is a separate, user-initiated action. On this machine:

```powershell
.\.venv\Scripts\python.exe main.py vq-train --dataset "E:\eigenfaces\img_align_celeba" --output models/vq64 --samples 50000 --epochs 100 --batch-size 128 --learning-rate 0.0002 --seed 42 --device cuda --workers 4
```

The default is full precision. Optional `--amp` enables CUDA autocast and a saved
gradient scaler; float32 nearest-neighbor search remains unchanged. Both precision
modes passed short synthetic CUDA training tests. `--device auto` uses CUDA when
available and otherwise CPU; explicit `--device cpu` supports CPU training.

By default, `models/neural64_retry/split.json` supplies the existing validation
and test holdouts. The trainer keeps both unchanged, includes the original
20,000 training files, and deterministically adds 30,000 other files outside
those holdouts. Preflight verified **50,000 train / 2,000 validation / 20,259 test**
images on this machine. The dataset remains read-only. This is filename-level
separation, not identity-level separation or duplicate-content detection.

`--base-split` selects another prior split. `--split-manifest` uses an exact saved
VQ split and takes precedence over requested sample counts. `--new-split`
explicitly creates a fresh deterministic split, primarily for independent data
and synthetic tests; use the default extension for comparisons with prior models.
The extension retains the base validation count regardless of
`--validation-samples`; that option controls newly created splits.

AdamW uses a constant logged learning rate, no scheduler, and gradient-norm
clipping at 1.0 by default. The objective is:

```text
reconstruction = 1.0*L1 + 0.2*(1-SSIM) + 0.2*edge
total = reconstruction + 1.0*codebook_MSE + 0.25*commitment_MSE
```

All five weights are configurable with `--l1-weight`, `--ssim-weight`,
`--edge-weight`, `--codebook-weight`, and `--commitment-weight`. Codebook and
commitment weights must be positive. Edge loss compares signed neighboring-pixel
differences with L1, giving equal weight to horizontal and vertical directions.
It encourages visible boundaries across the whole image and is not a detector
of eyes or mouths. SSIM training loss reuses the earlier differentiable local
implementation; reported evaluation SSIM uses the existing skimage metric.

Best-model selection and early stopping use the **validation reconstruction
objective**, excluding the two codebook-fitting terms. Total and reconstruction
losses, every individual component, MSE, SSIM, active entries, perplexity,
unused-entry percentage, actual learning rate and epoch compute duration are
logged separately. `--patience 15` controls reconstruction-loss early stopping.

Before the first optimizer update, 256 codebook vectors are sampled from encoder
features of at most 256 **training** images. This one-time initialization, file
count and seed are recorded. No validation/test image initializes the codebook,
and dead entries are not silently reseeded during training.

### Collapse monitoring and saved artifacts

The default severe-collapse stop requires three consecutive validation checks
after a five-epoch grace period with fewer than eight active entries, perplexity
below four, or identical payloads for every validation image. Configure using
`--collapse-grace`, `--collapse-patience`, `--collapse-min-active`, and
`--collapse-min-perplexity`. A severe stop is printed and checkpointed; resuming
that run is refused until a new experiment is deliberately chosen. These are
operational thresholds, not proof that every run above them has good utilization.

Outputs under `models/vq64/`:

- `best.pt`: selected full model, codebook, configuration and split provenance;
  it is an inference/selection checkpoint, not a resume checkpoint.
- `decoder.pt`: matching selected codebook and decoder only; no encoder,
  optimizer, original images, or filename manifest.
- `last.pt`: resumable model, optimizer, AMP scaler, Python/NumPy/Torch/CUDA/loader
  RNG states, current history and retained best-model weights.
- `config.json`, `split.json`, `history.csv`, `training.png`.
- `codebook_usage.csv`: frequency of every entry, for both train and validation
  at every epoch. Training frequencies span an evolving model; validation uses
  one fixed model per epoch. Perplexity is `exp(-sum(p*log(p)))` over assignments.
- `positions_NNN.npz`: per-position entry counts, shape 64x256; the position is
  `8*y+x`. `codebook_positions.png` plots latest validation frequencies.
- `validation_NNN.png`: the same seeded validation faces at baseline display
  scale, every five epochs and on the final/stopping epoch. No test images are
  evaluated during training. Constant positions and unique validation payloads
  are also tracked to expose spatially repetitive templates.

Fresh training refuses to overwrite an existing run. To continue the same run,
repeat its settings and specify a larger total epoch target:

```powershell
.\.venv\Scripts\python.exe main.py vq-train --dataset "E:\eigenfaces\img_align_celeba" --output models/vq64 --epochs 150 --batch-size 128 --learning-rate 0.0002 --device cuda --workers 4 --resume models/vq64/last.pt
```

Only `last.pt` restores the corresponding optimizer/RNG state. Recreated worker
processes can prevent bit identity with an uninterrupted multi-worker run;
preprocessing itself has no random augmentation. CPU synthetic resume was tested
to match uninterrupted training weights and numeric histories.

### Encoding, independent CPU decoding and comparison

```powershell
.\.venv\Scripts\python.exe main.py vq-encode "E:\eigenfaces\img_align_celeba\img_align_celeba\000001.jpg" --model models/vq64/best.pt --output output/vq_face.bin
.\.venv\Scripts\python.exe main.py vq-decode output/vq_face.bin --model models/vq64/decoder.pt --output output/vq_face.png --device cpu
.\.venv\Scripts\python.exe main.py vq-export-decoder --model models/vq64/best.pt --output output/vq_decoder.pt
.\.venv\Scripts\python.exe main.py vq-compare --dataset "E:\eigenfaces\img_align_celeba" --samples 25 --device cpu
```

The comparator defaults to the preserved PCA model and the best previous neural
model, `models/neural64_continue_A/best.pt`. Options select other checkpoints.
It uses their common held-out filenames, excludes each model's training and
validation files, and produces four-column Original/PCA/Neural/VQ sheets, 25+
face contact sheets, MSE/PSNR/SSIM/edge-error tables, utilization counts, a graph,
model hashes, selection metadata, and `report.md` under `output/vq_evaluation/`.
All metrics score float reconstructions before PNG rounding. Edge error uses the
same signed-gradient definition as training, with lower values better.

For deliberate expression/accessory/hairstyle coverage, pass
`--attributes path/to/list_attr_celeba.txt`. Selection covers available CelebA
annotations for smiles, glasses, bangs, baldness, facial hair, hats and hair
texture. No such attribute file was found in this machine's current dataset.
Without annotations, the default uses varied spatial brightness and edge
strength as a proxy; it **cannot guarantee semantic coverage**. It creates an
original-image candidate sheet for manual review. Supply a JSON list of at least
25 common held-out relative filenames using `--selection curated_faces.json`
to curate varied expressions, glasses, facial hair and styles explicitly.

The evaluation report provides a qualitative-review checklist for eyes, nose
shape, mouth/expression, glasses, facial hair, hairline, proportions, ghosting,
generic structures and invented details. A trained CelebA checkpoint and human
review are required before filling in conclusions: synthetic comparisons prove
only that the pipeline runs. SSIM and edge error do not establish identity
preservation. More active codes do not alone imply recognizable faces. This
codec may still blur, memorize templates, or fail to recover distinctive traits.

### Tests and short synthetic smoke runs

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/vq_smoke.py --device cuda --workers 2 --probe-batch-size 128
# Optional AMP smoke test:
.\.venv\Scripts\python.exe scripts/vq_smoke.py --device cuda --workers 2 --probe-batch-size 128 --amp
```

Tests cover exact geometry, nearest-neighbor and tie behavior, gradient routing,
row-major 64-byte serialization, deterministic inference, configuration/weight
compatibility, CPU decoding in a separate process with originals hidden and
encoder construction blocked, split extension, utilization/spatial statistics,
collapse stopping, training/resume, CUDA AMP, and four-way evaluation artifacts.
Smoke runs train two epochs on synthetic images, delete their temporary dataset,
decode in a separate CPU process, and probe GPU memory. Artifacts live only under
`output/vq_smoke/run_*/`. They never train on CelebA or read prior checkpoints.

All dataset images, weights and generated outputs remain excluded from Git.
CelebA is for noncommercial research; this educational project and its MIT code
license do not grant redistribution or commercial rights to dataset-derived
images or trained models. See the dataset agreement and attribution above.

### Phase 7: frontal specialization

The separate scratch-training experiment uses 50,000 strict-frontal images and
preserves all historical validation/test filenames. It retains the Phase 5
architecture, optimizer, losses, preprocessing and 64-byte format. Preparation
freezes source hashes, training/validation/test manifests, pose-review panels,
and paired frontal/nonfrontal evaluation cohorts before training. Five-point
geometry does not reliably detect head pitch; unapproved manual exclusions
remain unapplied. Identity separation cannot be established without annotations.

```powershell
.venv\Scripts\python.exe -m src.phase7 prepare  # new experiment directory only
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m src.phase7 train
.venv\Scripts\python.exe -m src.phase7 evaluate
```

Checkpoints are isolated under `models/vq64_frontal/`; all manifests, audit
images, monitored validation comparisons, paired bootstrap results and the
report are under `output/phase7_frontal_vq/`. Training refuses an occupied model
directory and stops for numerical failure or severe codebook collapse. The
existing VQ resume command remains available for an explicitly resumed run.

An exact 50,000-image control manifest/configuration is prepared independently.
The command below requires a separate decision; Phase 7 never launches it:

```powershell
.venv\Scripts\python.exe -m src.phase7 control
```

### Phase 8 Stage A: background-removal prototype

Stage A audits 200 frozen Phase 7 **training** images using the author's pinned
BiRefNet-portrait matting model. It does not preprocess the full dataset or
train another codec. Review [PHASE8_SEGMENTATION.md](PHASE8_SEGMENTATION.md) for
the model source, MIT license metadata, dependency choices and external weight
location before installing optional `requirements-matting.txt` dependencies.
Keep the existing PyTorch CUDA build and install its matching torchvision.

```powershell
.venv\Scripts\python.exe scripts/phase8_fetch_model.py
.venv\Scripts\python.exe -m src.foreground_audit --device cuda --samples 200 --seed 42
.venv\Scripts\python.exe -m pytest -q
```

The fetcher verifies the upstream SHA-256 of Safetensors and reviewed model
source. Inference loads those local files without additional downloads.
Native-resolution masks, gray 64x80 targets, cache integrity records, failure
logs and identical-alpha gray/white/black previews are saved under
`output/phase8_nobg/`. Re-running verifies and reuses successful cache entries.
Source JPEGs and every Phase 7 split remain unchanged. Diagnostic mask flags
request review; they do not automatically exclude images. The audit is capped
at 200 images. Approval after visual review is required before full preprocessing
under `E:\eigenfaces\foreground_cache` or VQ training.

The VQ network and its 64-byte format are unchanged. A constant-gray target is
not a transparent decoder output; no alpha-aware decoder is implemented here.

Stage B full-cache preprocessing is documented in
[PHASE8_STAGE_B.md](PHASE8_STAGE_B.md):

```powershell
.venv\Scripts\python.exe -m src.foreground_full run
```

The job resumes verified cache entries under `E:\eigenfaces\foreground_cache`
and uses the exact 61,116 Phase 7 filenames. Quality flags generate reviews;
they do not exclude images. Final auditing checks every hash, image shape and
exact target reproduction. Outputs/report are separate from Stage A, under
`output/phase8_nobg/stage_b/` and `stage_b_report.md`. The utility stops before
training. Stage C requires a separate approval after reviewing this cache.

### Phase 10: ID-style portrait prototype

[PHASE10.md](PHASE10.md) documents the capped 500-image curation prototype,
detector provenance, similarity-aligned conservative/balanced/tight previews,
and auditable manual-review records. Results are under
`output/phase10_id_portraits/prototype_report.md`. No VQ training or full-dataset
processing occurs; filtering rules and a final crop require visual approval.

```powershell
.venv\Scripts\python.exe -m src.id_portrait_prototype report
```
