"""Phase13 Face24 model: Phase8 encoder/quantizer plus the established Phase9 decoder."""
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .decoder_capacity_model import LargeConfig, LargeDecoder, LargeDecoderOnly, validate_indices
from .vq_model import VQConfig, VQEncoder, VectorQuantizer


class Face24LargeVQVAE(nn.Module):
    def __init__(self, config=VQConfig(), decoder_config=LargeConfig()):
        super().__init__()
        self.config = config
        self.decoder_config = decoder_config
        self.encoder = VQEncoder(config)
        self.quantizer = VectorQuantizer(config)
        self.decoder = LargeDecoder(decoder_config)

    def forward(self, image):
        z = self.encoder(image)
        quantized, indices, codebook, commitment = self.quantizer(z)
        return self.decoder(quantized), indices, codebook, commitment


def parameter_counts(model):
    return {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "codebook": sum(p.numel() for p in model.quantizer.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "full": sum(p.numel() for p in model.parameters()),
    }


def _atomic_torch_save(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)


def save_full(model, path, **extra):
    bundle = {
        "format_version": 1,
        "codec": "vqface-face24-large-v1",
        "kind": "vqvae",
        "vq_config": asdict(model.config),
        "decoder_config": asdict(model.decoder_config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **extra,
    }
    _atomic_torch_save(bundle, path)


def save_decoder(model, path):
    state = {"codebook.weight": model.quantizer.embedding.weight.detach().cpu()}
    state.update({"decoder." + k: v.detach().cpu() for k, v in model.decoder.state_dict().items()})
    _atomic_torch_save({
        "format_version": 1,
        "codec": "vqface-face24-large-v1",
        "kind": "decoder",
        "vq_config": asdict(model.config),
        "decoder_config": asdict(model.decoder_config),
        "state_dict": state,
    }, path)


def load_full(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("format_version") != 1 or bundle.get("codec") != "vqface-face24-large-v1" or bundle.get("kind") != "vqvae":
        raise ValueError("Unsupported Phase13 full checkpoint")
    config = VQConfig(**bundle["vq_config"])
    decoder_config = LargeConfig(**bundle["decoder_config"])
    model = Face24LargeVQVAE(config, decoder_config)
    model.load_state_dict(bundle["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite Phase13 checkpoint")
    return model.to(device).eval(), bundle


def load_decoder(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1 or bundle.get("codec") != "vqface-face24-large-v1" or bundle.get("kind") != "decoder":
        raise ValueError("Unsupported Phase13 decoder artifact")
    vq_config = VQConfig(**bundle["vq_config"])
    decoder_config = LargeConfig(**bundle["decoder_config"])
    model = LargeDecoderOnly(decoder_config)
    model.load_state_dict(bundle["state_dict"], strict=True)
    if model.codebook.weight.shape != (vq_config.entries, vq_config.embedding_dim):
        raise ValueError("Decoder codebook disagrees with VQ configuration")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval(), bundle


def decode_bytes(payload, decoder, device=None):
    if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != 64:
        raise ValueError("Phase13 payload must contain exactly 64 bytes")
    validate_indices(torch.zeros((1, 8, 8), dtype=torch.long))
    target_device = device or next(decoder.parameters()).device
    indices = torch.tensor(list(payload), dtype=torch.long, device=target_device).reshape(1, 8, 8)
    with torch.inference_mode():
        out = decoder(indices)
    if tuple(out.shape) != (1, 1, 80, 64):
        raise ValueError("Phase13 decoder output must be N x 1 x 80 x 64")
    return out[0, 0].detach().cpu().numpy()
