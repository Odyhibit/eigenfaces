"""Versioned CAE, byte quantizer and portable checkpoints (no PCA dependency)."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .preprocess import IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEURAL = ROOT / "models" / "neural64" / "best.pt"
DEFAULT_DECODER = ROOT / "models" / "neural64" / "decoder.pt"
LATENT_SIZES = (32, 48, 64, 80, 96)


@dataclass(frozen=True)
class ModelConfig:
    latent: int = 64
    architecture: str = "cae-v1"
    image_size: tuple = IMAGE_SIZE
    quantization: str = "uint8-round-even-v1"

    def __post_init__(self):
        if (self.latent not in LATENT_SIZES or self.architecture != "cae-v1"
                or tuple(self.image_size) != IMAGE_SIZE
                or self.quantization != "uint8-round-even-v1"):
            raise ValueError("Incompatible neural model configuration")


def quantize(z):
    # Quantization always runs in float32, including inside autocast.
    return torch.round(z.float().clamp(0, 1) * 255).to(torch.uint8)


def dequantize(q):
    if q.dtype != torch.uint8:
        raise ValueError("Expected unsigned 8-bit latent values")
    return q.float() / 255


def quantize_ste(z):
    z = z.float()
    q = dequantize(quantize(z))
    # Forward is EXACTLY q (no cancellation error); backward is identity.
    return q + (z - z.detach())


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 10 * 8, config.latent), nn.Sigmoid())

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 80, 64):
            raise ValueError("Expected N x 1 x 80 x 64 images")
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            nn.Linear(config.latent, 64 * 10 * 8), nn.ReLU(),
            nn.Unflatten(1, (64, 10, 8)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.config.latent:
            raise ValueError(f"Expected N x {self.config.latent} latent values")
        return self.layers(z)


class Autoencoder(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder, self.decoder = Encoder(config), Decoder(config)

    def forward(self, x, return_latent=False):
        z = self.encoder(x)
        reconstruction = self.decoder(quantize_ste(z))
        return (reconstruction, z) if return_latent else reconstruction


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu or cuda[:index]")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return result


def safe_output(path, dataset=None):
    path = Path(path).resolve()
    protected = [ROOT / "dataset", Path(r"D:\eigenfaces\img_align_celeba")]
    if dataset is not None:
        protected.append(Path(dataset).resolve())
    if any(path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Output must be outside the original dataset directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path, model, kind="autoencoder", **extra):
    if kind not in {"autoencoder", "decoder"}:
        raise ValueError("Invalid checkpoint kind")
    state = model.decoder.state_dict() if kind == "decoder" else model.state_dict()
    bundle = {"format_version": 1, "kind": kind, "config": asdict(model.config),
              "state_dict": {k: v.detach().cpu() for k, v in state.items()}}
    if kind == "decoder" and extra:
        raise ValueError("Decoder export must contain only configuration and decoder weights")
    bundle.update(extra)
    path = safe_output(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu", decoder_only=False):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1 or bundle.get("kind") not in {"autoencoder", "decoder"}:
        raise ValueError("Unsupported neural checkpoint")
    config = ModelConfig(**bundle["config"])
    if decoder_only:
        model = Decoder(config)
        state = bundle["state_dict"]
        if bundle["kind"] == "autoencoder":
            state = {k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")}
    else:
        if bundle["kind"] != "autoencoder":
            raise ValueError("Encoding/training requires an autoencoder checkpoint")
        model, state = Autoencoder(config), bundle["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint weights do not match model configuration") from error
    return model.to(resolve_device(device)).eval(), bundle


def export_decoder(model_path, output, device="cpu"):
    model, _ = load_checkpoint(model_path, device)
    save_checkpoint(output, model, kind="decoder")
