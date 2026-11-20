"""64 unsigned codebook indices: row-major grid[y,x], no per-image metadata."""
from pathlib import Path

import numpy as np
import torch

from .preprocess import array_to_image, preprocess_image
from .vq_model import DEFAULT_VQ, DEFAULT_VQ_DECODER, load_vq, save_vq, vq_output


@torch.inference_mode()
def encode_array(values, model):
    values = np.asarray(values, dtype=np.float32)
    if values.size != 5120 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Expected 5120 finite pixels in [0,1]")
    model.eval()
    x = torch.from_numpy(values.copy().reshape(1, 1, 80, 64)).to(next(model.parameters()).device)
    indices = model.quantizer.indices(model.encoder(x))
    return indices[0].to(torch.uint8).cpu().numpy().tobytes(order="C")


@torch.inference_mode()
def decode_payload(payload, decoder):
    if len(payload) != 64:
        raise ValueError(f"Expected exactly 64 VQ bytes; got {len(payload)}")
    decoder.eval()
    indices = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy().reshape(1, 8, 8)).long()
    result = decoder(indices.to(next(decoder.parameters()).device))
    return result.cpu().numpy().reshape(-1)


def encode_image(image_path, model_path=DEFAULT_VQ, output=None, device="cpu"):
    model, _ = load_vq(model_path, device)
    payload = encode_array(preprocess_image(image_path), model)
    if output is not None:
        vq_output(output, Path(image_path).resolve().parent).write_bytes(payload)
    return payload


def decode_file(payload_path, model_path=DEFAULT_VQ_DECODER, output=None, device="cpu"):
    if Path(payload_path).stat().st_size != 64:
        raise ValueError("Expected exactly 64 VQ bytes")
    decoder, _ = load_vq(model_path, device, decoder_only=True)
    result = decode_payload(Path(payload_path).read_bytes(), decoder)
    if output is not None:
        array_to_image(result).save(vq_output(output), format="PNG")
    return result


def export_decoder(model_path=DEFAULT_VQ, output=DEFAULT_VQ_DECODER):
    model, _ = load_vq(model_path)
    save_vq(output, model, decoder_only=True)
