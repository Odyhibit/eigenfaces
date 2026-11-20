# Phase 19 Face24 QR demo

A local Gradio proof of concept for the frozen Phase 18 codec. The demo does not train a model, load ArcFace, change the dataset, or upload images to a service.

## Requirements and launch

Use the repository virtual environment on Windows. Install only the small UI/QR dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe demo\qr_face_app.py
```

Open `http://127.0.0.1:7860`. The server binds to loopback and sharing is disabled. The Phase 18 selected checkpoint and matching decoder export must exist at `output/phase18_identity_loss/checkpoints/best.pt` and `decoder.pt`; their SHA-256 hashes are checked against `training_completion.json` at startup. The existing local MediaPipe FaceLandmarker model is hash-verified against its project provenance record.

The encoder uses CUDA when PyTorch reports it available and otherwise uses CPU. The decoder-only export is loaded on CPU. Downloadable QR and reconstruction PNGs are written to `output/phase19_qr_demo/`; uploaded source photos are not saved by the app.

## Alignment and payload

The frozen dataset uses CelebA five-point annotations, unavailable for arbitrary uploads. This demo obtains a five-point estimate from the project's existing MediaPipe FaceLandmarker: eye centers from the paired eye corners, nose tip, and mouth corners. It applies the same Phase 10C `24`-pixel eye-separation similarity crop, midpoint `(31.5, 23)`, uniform scale, eye-line rotation, gray out-of-source padding, and established 4× cubic / Lanczos crop renderer. It does not use matting, enhancement, ArcFace, or identity matching. Missing, multiple, implausible, or critically clipped faces are rejected with an explanatory message. This inferred upload geometry will not be pixel-identical to CelebA's annotated geometry.

The QR stores exactly 64 raw bytes in QR byte mode, with automatic version selection, medium error correction, and a four-module quiet zone. There is no header, checksum, text encoding, Base64, or hex conversion. `zxing-cpp` must return the raw byte sequence exactly; the app checks the generated QR by scanning its rendered image before showing it. The Decode tab accepts only one QR containing exactly 64 bytes and reconstructs from those bytes alone.

## Tests

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_phase19_qr_demo.py -q
```

Tests include arbitrary binary QR round trips (including null and high-bit bytes), byte ordering, invalid payload rejection, actual checkpoint/export agreement, and a complete upload-to-QR-scan-to-reconstruction path using a frozen source image when the local CelebA source is available.
