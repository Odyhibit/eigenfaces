"""Inference-only Face24 / Phase 18 / binary QR helpers for the local demo."""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.face24_large_model import (
    decode_bytes,
    load_decoder,
    load_full,
    parameter_counts,
)
from src.face_crop_geometry import similarity
from src.id_portrait_geometry import apply_transform, render_crop
from src.vq_codec import encode_array

ROOT = Path(__file__).resolve().parents[1]
PHASE18 = ROOT / "output/phase18_identity_loss"
CHECKPOINT = PHASE18 / "checkpoints/best.pt"
DECODER_EXPORT = PHASE18 / "checkpoints/decoder.pt"
COMPLETION = PHASE18 / "training_completion.json"
DETECTOR_MODEL = ROOT / "models/id_portrait_detectors/face_landmarker.task"
DETECTOR_PROVENANCE = ROOT / "models/id_portrait_detectors/provenance.json"
DEMO_OUTPUT = ROOT / "output/phase19_qr_demo"
EXPECTED_ARC = 24.0
EXPECTED_EYE = (31.5, 23.0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def five_points_from_mesh(mesh, width: int, height: int) -> np.ndarray:
    """Return screen-left/right eye centers, nose tip, and screen-left/right mouth corners."""
    try:
        p = np.asarray(mesh, dtype=np.float64)
    except (TypeError, ValueError):
        try:
            p = np.asarray([[point.x, point.y] for point in mesh], dtype=np.float64)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Face landmarks are missing or invalid; try a clearer, front-facing photo.") from exc
    if p.ndim != 2 or p.shape[0] < 468 or p.shape[1] < 2 or not np.isfinite(p[:, :2]).all():
        raise ValueError("Face landmarks are missing or invalid; try a clearer, front-facing photo.")
    xy = p[:, :2] * np.asarray([width, height], dtype=np.float64)
    # MediaPipe Face Mesh's eye corner pairs; centers match the Phase 10C
    # five-point eye-center convention. Sort by image x for stable orientation.
    eyes = [xy[[33, 133]].mean(axis=0), xy[[362, 263]].mean(axis=0)]
    eyes.sort(key=lambda q: q[0])
    mouth = [xy[61], xy[291]]
    mouth.sort(key=lambda q: q[0])
    points = np.asarray([eyes[0], eyes[1], xy[1], mouth[0], mouth[1]], dtype=np.float64)
    iod = float(np.linalg.norm(points[1] - points[0]))
    if points[1, 0] <= points[0, 0] or not 2.0 <= iod <= min(width, height) * 0.8:
        raise ValueError("Detected eye landmarks have implausible geometry; try a clearer, front-facing photo.")
    return points


def _verify_detector_asset() -> None:
    doc = json.loads(DETECTOR_PROVENANCE.read_text(encoding="utf-8"))
    entry = next((x for x in doc["files"] if Path(x["path"]).name == DETECTOR_MODEL.name), None)
    if entry is None or _sha256(DETECTOR_MODEL) != entry["sha256"]:
        raise RuntimeError("The local MediaPipe face-landmark model is missing or failed its recorded hash check.")


def align_from_landmarks(image: Image.Image, five_points, dense_points) -> Image.Image:
    """Apply the frozen Face24 B geometry to pixel-space five/dense landmarks."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    five = np.asarray(five_points, dtype=np.float64)
    dense = np.asarray(dense_points, dtype=np.float64)
    if five.shape != (5, 2) or dense.ndim != 2 or dense.shape[0] < 468 or dense.shape[1] != 2:
        raise ValueError("Face landmarks are missing or invalid; try a clearer, front-facing photo.")
    if not np.isfinite(five).all() or not np.isfinite(dense).all():
        raise ValueError("Face landmarks contain invalid coordinates.")
    transform = similarity(five, EXPECTED_ARC, EXPECTED_EYE[1])
    matrix = np.asarray(transform["matrix_source_to_output"], dtype=np.float64)
    important = np.vstack([five, dense[[70, 105, 336, 300, 152]]])
    fitted = apply_transform(important, matrix)
    if not np.isfinite(fitted).all() or np.any(fitted[:, 0] < -0.5) or np.any(fitted[:, 0] > 63.5) or np.any(fitted[:, 1] < -0.5) or np.any(fitted[:, 1] > 79.5):
        raise ValueError("This face's critical landmarks do not fit the Face24 crop. Try a more frontal or less tightly framed photo.")
    gray, _, _ = render_crop(image, np.ones((h, w), dtype=np.float32), transform)
    if gray.mode != "L" or gray.size != (64, 80):
        raise RuntimeError("Face24 renderer returned an invalid image format.")
    return gray


class Face24Aligner:
    """One persistent MediaPipe detector and the published Face24 renderer."""

    def __init__(self):
        _verify_detector_asset()
        import mediapipe as mp

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(DETECTOR_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=8,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._mp = mp
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._detector.close()

    def align(self, image: Image.Image) -> Image.Image:
        if image is None:
            raise ValueError("Upload a JPG or PNG face photograph first.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        frame = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._detector.detect(frame)
        count = len(result.face_landmarks)
        if count == 0:
            raise ValueError("No face was detected. Try a clear, well-lit image with the face visible.")
        if count != 1:
            raise ValueError(f"Detected {count} faces. Please provide an image containing exactly one face.")
        h, w = rgb.shape[:2]
        five = five_points_from_mesh(result.face_landmarks[0], w, h)
        mesh = np.asarray([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float64)
        return align_from_landmarks(image, five, mesh)


class Phase18Runtime:
    """Loads and verifies the matching encoder and CPU decoder exactly once."""

    def __init__(self, device: str | None = None):
        if not COMPLETION.is_file():
            raise RuntimeError("Phase 18 training completion record is missing.")
        completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
        if completion.get("selected_epoch") != 147 or completion.get("optimizer_steps") != 58000:
            raise RuntimeError("Phase 18 selected-checkpoint provenance is invalid.")
        if _sha256(CHECKPOINT) != completion["best_checkpoint_sha256"]:
            raise RuntimeError("Phase 18 selected checkpoint hash mismatch.")
        if _sha256(DECODER_EXPORT) != completion["decoder_sha256"]:
            raise RuntimeError("Phase 18 decoder export hash mismatch.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder, bundle = load_full(CHECKPOINT, self.device)
        self.decoder, decoder_bundle = load_decoder(DECODER_EXPORT, "cpu")
        if (bundle.get("codec") != "vqface-face24-large-v1" or bundle.get("kind") != "vqvae"
                or bundle.get("checkpoint_role") != "selection" or bundle.get("epoch") != completion["selected_epoch"]
                or bundle.get("experiment_sha256") != completion["experiment_sha256"]):
            raise RuntimeError("Loaded checkpoint is not the selected Phase 18 model.")
        if parameter_counts(self.encoder) != {
            "encoder": 517408, "codebook": 16384, "decoder": 3411585, "full": 3945377
        }:
            raise RuntimeError("Phase 18 architecture does not match the frozen codec specification.")
        full_state = bundle["state_dict"]
        exported_state = decoder_bundle["state_dict"]
        if not torch.equal(full_state["quantizer.embedding.weight"].cpu(), exported_state["codebook.weight"].cpu()):
            raise RuntimeError("Phase 18 decoder export uses a different codebook.")
        for key, value in self.encoder.decoder.state_dict().items():
            if not torch.equal(value.cpu(), exported_state["decoder." + key].cpu()):
                raise RuntimeError("Phase 18 decoder export does not match the selected checkpoint.")
        self._lock = threading.Lock()

    def encode(self, face24: Image.Image) -> bytes:
        gray = np.asarray(face24.convert("L"), dtype=np.uint8)
        if gray.shape != (80, 64):
            raise ValueError("Expected a 64×80 Face24 image.")
        with self._lock:
            payload = encode_array(gray.astype(np.float32) / 255.0, self.encoder)
        if len(payload) != 64:
            raise RuntimeError(f"Encoder returned {len(payload)} bytes; expected exactly 64.")
        return payload

    def reconstruct(self, payload: bytes) -> Image.Image:
        if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != 64:
            raise ValueError("Invalid Phase 18 payload: expected exactly 64 bytes.")
        with self._lock:
            pixels = decode_bytes(payload, self.decoder)
        values = np.asarray(pixels, dtype=np.float32)
        if values.shape != (80, 64) or not np.isfinite(values).all():
            raise ValueError("Phase 18 decoder returned invalid pixels.")
        return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))


def make_binary_qr(payload: bytes) -> Image.Image:
    """Create a standard QR with raw byte-mode data, never text/base64/hex."""
    if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != 64:
        raise ValueError("QR payload must contain exactly 64 raw bytes.")
    import qrcode
    from qrcode.constants import ERROR_CORRECT_M
    from qrcode.util import MODE_8BIT_BYTE, QRData

    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M, box_size=8, border=4)
    qr.add_data(QRData(bytes(payload), mode=MODE_8BIT_BYTE), optimize=0)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def read_binary_qr(image: Image.Image) -> bytes:
    """Decode exactly one QR barcode and return its raw bytes property."""
    import zxingcpp

    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    results = zxingcpp.read_barcodes(bgr)
    if len(results) != 1:
        raise ValueError("No QR code was decoded (or more than one code was found). Upload one clear QR image.")
    result = results[0]
    if str(result.format).lower().split(".")[-1] != "qrcode":
        raise ValueError("The uploaded image is not a QR code.")
    raw = getattr(result, "bytes", None)
    if raw is None:
        raise RuntimeError("The installed QR scanner does not expose raw binary payload bytes.")
    payload = bytes(raw)
    if len(payload) != 64:
        raise ValueError(f"Decoded QR payload has {len(payload)} bytes; Phase 18 requires exactly 64.")
    return payload


def save_demo_image(image: Image.Image, suffix: str) -> str:
    DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = DEMO_OUTPUT / f"{secrets.token_hex(12)}{suffix}"
    image.save(path)
    return str(path)


def encode_upload(image: Image.Image, aligner: Face24Aligner, runtime: Phase18Runtime):
    aligned = aligner.align(image)
    payload = runtime.encode(aligned)
    qr = make_binary_qr(payload)
    # Test the actual generated QR pixels before exposing the result.
    recovered = read_binary_qr(qr)
    if recovered != payload:
        raise RuntimeError("QR scanner did not recover the exact 64 encoded bytes.")
    reconstructed = runtime.reconstruct(recovered)
    download = save_demo_image(qr, ".png")
    return ImageOps.exif_transpose(image).convert("RGB"), aligned, reconstructed, qr, download, f"Ready · payload {len(payload)} bytes · encoder device {runtime.device} · QR binary round-trip verified."


def decode_upload(image: Image.Image, runtime: Phase18Runtime):
    if image is None:
        raise ValueError("Upload a QR code image first.")
    payload = read_binary_qr(image)
    reconstructed = runtime.reconstruct(payload)
    download = save_demo_image(reconstructed, ".png")
    return reconstructed, download, f"Valid Phase 18 QR | {len(payload)} raw bytes | reconstructed on CPU | no source image used."


def create_app():
    import gradio as gr

    runtime = Phase18Runtime()
    aligner = Face24Aligner()

    def encode_callback(path):
        try:
            if path is None:
                raise ValueError("Upload a JPG or PNG face photograph first.")
            path = Path(path)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Only JPG and PNG photographs are supported.")
            with Image.open(path) as image:
                image.load()
                values = encode_upload(image, aligner, runtime)
            return values
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def decode_callback(path):
        try:
            if path is None:
                raise ValueError("Upload a QR code PNG or JPG first.")
            with Image.open(path) as image:
                image.load()
                values = decode_upload(image, runtime)
            return values
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Face24 QR codec demo") as demo:
        gr.Markdown("# Face24 QR codec demo\nLocal demonstration. Images are processed on this machine; no account, cloud service, or identity recognition is used.")
        with gr.Tab("Encode Image"):
            upload = gr.File(label="Face photograph (JPG or PNG)", file_types=[".jpg", ".jpeg", ".png"], type="filepath")
            encode_button = gr.Button("Encode to QR", variant="primary")
            with gr.Row():
                original_preview = gr.Image(label="Uploaded image", type="pil")
                face_preview = gr.Image(label="Face24 B grayscale 64x80", type="pil")
                reconstruction = gr.Image(label="Phase 18 reconstruction", type="pil")
                qr_preview = gr.Image(label="Binary QR code", type="pil")
            qr_download = gr.DownloadButton(label="Download QR PNG")
            encode_status = gr.Textbox(label="Status", interactive=False)
            encode_button.click(encode_callback, inputs=upload, outputs=[original_preview, face_preview, reconstruction, qr_preview, qr_download, encode_status])
        with gr.Tab("Decode QR"):
            qr_upload = gr.File(label="QR code image (PNG or JPG)", file_types=[".png", ".jpg", ".jpeg"], type="filepath")
            decode_button = gr.Button("Decode QR", variant="primary")
            with gr.Row():
                decoded_face = gr.Image(label="Reconstructed grayscale face", type="pil")
                decoded_download = gr.DownloadButton(label="Download reconstructed PNG")
            decode_status = gr.Textbox(label="Payload validation", interactive=False)
            decode_button.click(decode_callback, inputs=qr_upload, outputs=[decoded_face, decoded_download, decode_status])
    return demo


def main():
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
