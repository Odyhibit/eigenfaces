"""VQ face codec v1: 8x8 indices, 256 entries, 64-dimensional embeddings."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .neural import ROOT, resolve_device, safe_output

DEFAULT_VQ = ROOT / "models/vq64/best.pt"
DEFAULT_VQ_DECODER = ROOT / "models/vq64/decoder.pt"


@dataclass(frozen=True)
class VQConfig:
    architecture: str = "vqface-v1"
    embedding_dim: int = 64
    entries: int = 256
    grid: tuple = (8, 8)
    image_size: tuple = (64, 80)
    order: str = "row-major-yx"

    def __post_init__(self):
        if (self.architecture != "vqface-v1" or self.embedding_dim != 64 or self.entries != 256
                or tuple(self.grid) != (8, 8) or tuple(self.image_size) != (64, 80) or self.order != "row-major-yx"):
            raise ValueError("Incompatible VQ codec configuration")


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.ReLU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.layers(x)


class VQEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),  # 40 x 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 20 x 16
            nn.Conv2d(64, 128, 4, 2, 1),  # 10 x 8
            Residual(128), Residual(128), nn.ReLU(),
            # Valid learned projection uses ALL ten feature rows, without
            # cropping/padding/resizing the original rectangular image.
            nn.Conv2d(128, config.embedding_dim, (3, 1)))  # 8 x 8

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 80, 64):
            raise ValueError("Expected N x 1 x 80 x 64 images")
        return self.layers(x)


class VQDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(config.embedding_dim, 128, (3, 1)),  # 10 x 8
            Residual(128), Residual(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), Residual(64),  # 20 x 16
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), Residual(32),  # 40 x 32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())  # 80 x 64

    def forward(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (64, 8, 8):
            raise ValueError("Expected N x 64 x 8 x 8 embedding maps")
        return self.layers(z)


class VectorQuantizer(nn.Module):
    """Standard codebook + commitment losses, not EMA; ties select first index."""
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.entries, config.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / config.entries, 1 / config.entries)

    @torch.no_grad()
    def indices(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (64, 8, 8):
            raise ValueError("Expected N x 64 x 8 x 8 features")
        with torch.autocast(device_type=z.device.type, enabled=False):
            vectors = z.float().permute(0, 2, 3, 1).reshape(-1, 64)
            weights = self.embedding.weight.float()
            distance = vectors.square().sum(1, keepdim=True) + weights.square().sum(1)[None] - 2 * vectors @ weights.T
            return distance.argmin(1).reshape(-1, 8, 8)

    def forward(self, z):
        indices = self.indices(z)
        embedding = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(embedding.float(), z.detach().float())
        commitment_loss = F.mse_loss(z.float(), embedding.detach().float())
        # Exact embedding values in forward; identity gradient to encoder.
        quantized = embedding.detach() + (z.float() - z.float().detach())
        return quantized, indices, codebook_loss, commitment_loss


class VQVAE(nn.Module):
    def __init__(self, config=VQConfig()):
        super().__init__()
        self.config = config
        self.encoder, self.quantizer, self.decoder = VQEncoder(config), VectorQuantizer(config), VQDecoder(config)

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, codebook, commitment = self.quantizer(z)
        return self.decoder(quantized), indices, codebook, commitment


class DecoderOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.codebook = nn.Embedding(config.entries, config.embedding_dim)
        self.decoder = VQDecoder(config)

    def forward(self, indices):
        if (indices.ndim != 3 or tuple(indices.shape[1:]) != (8, 8)
                or indices.dtype != torch.long or indices.min() < 0 or indices.max() > 255):
            raise ValueError("Expected N x 8 x 8 integer indices in [0,255]")
        return self.decoder(self.codebook(indices).permute(0, 3, 1, 2).contiguous())


def vq_output(path, dataset=None):
    path = Path(path).resolve()
    # Existing model/report locations remain read-only for this new codec.
    for root in (ROOT / "models", ROOT / "output"):
        if path.is_relative_to(root) and not path.relative_to(root).parts[0].startswith("vq"):
            raise ValueError("VQ outputs must not overwrite a previous model or report directory")
    return safe_output(path, dataset)


def save_vq(path, model, decoder_only=False, **extra):
    if decoder_only:
        state = {"codebook.weight": model.quantizer.embedding.weight.detach().cpu()}
        state.update({"decoder." + k: v.detach().cpu() for k, v in model.decoder.state_dict().items()})
        if extra:
            raise ValueError("Decoder export cannot contain training state")
    else:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    bundle = {"format_version": 1, "codec": "vqface-v1", "kind": "decoder" if decoder_only else "vqvae",
              "config": asdict(model.config), "state_dict": state, **extra}
    path = vq_output(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)


def load_vq(path=DEFAULT_VQ, device="cpu", decoder_only=False):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("codec") != "vqface-v1" or bundle.get("format_version") != 1 or bundle.get("kind") not in {"vqvae", "decoder"}:
        raise ValueError("Unsupported VQ checkpoint")
    try:
        config = VQConfig(**bundle["config"])
    except TypeError as error:
        raise ValueError("Invalid VQ model configuration") from error
    if decoder_only:
        if bundle["kind"] != "decoder":
            raise ValueError("Independent VQ decoding requires a decoder-only checkpoint")
        model = DecoderOnly(config)
    else:
        if bundle["kind"] != "vqvae":
            raise ValueError("VQ encoding/training requires the full checkpoint")
        model = VQVAE(config)
    try:
        model.load_state_dict(bundle["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("VQ checkpoint weights do not match configuration") from error
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite VQ model weights")
    return model.to(resolve_device(device)).eval(), bundle
