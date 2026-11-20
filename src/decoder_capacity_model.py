"""One larger decoder; the Phase 8 encoder/codebook and wire format are frozen."""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy
import torch
from torch import nn

from .phase7 import state_hash
from .vq_model import VQConfig,VQEncoder,VectorQuantizer,Residual,vq_output
from .neural import resolve_device


@dataclass(frozen=True)
class LargeConfig:
    architecture:str="vqface-decoder-large-v1"
    encoder_architecture:str="vqface-v1"
    channels:tuple=(256,128,64)
    residual_blocks:tuple=(4,2,1)
    embedding_dim:int=64
    entries:int=256
    grid:tuple=(8,8)
    image_size:tuple=(64,80)
    order:str="row-major-yx"

    def __post_init__(self):
        expected={"architecture":"vqface-decoder-large-v1","encoder_architecture":"vqface-v1",
            "channels":(256,128,64),"residual_blocks":(4,2,1),"embedding_dim":64,"entries":256,
            "grid":(8,8),"image_size":(64,80),"order":"row-major-yx"}
        for key,value in expected.items():
            actual=getattr(self,key)
            if isinstance(value,tuple):actual=tuple(actual)
            if actual!=value:raise ValueError("Unsupported larger-decoder configuration")


class LargeDecoder(nn.Module):
    def __init__(self,config=LargeConfig()):
        super().__init__()
        self.layers=nn.Sequential(
            nn.ConvTranspose2d(64,256,(3,1)),
            *[Residual(256) for _ in range(4)],nn.ReLU(),
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(256,128,3,padding=1),nn.ReLU(),Residual(128),Residual(128),
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(128,64,3,padding=1),nn.ReLU(),Residual(64),
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(64,1,3,padding=1),nn.Sigmoid())

    def forward(self,z):
        if z.ndim!=4 or tuple(z.shape[1:])!=(64,8,8):raise ValueError("Expected N x 64 x 8 x 8 embeddings")
        return self.layers(z)


class FrozenCodec(nn.Module):
    def __init__(self,baseline=None,config=LargeConfig()):
        super().__init__();self.config=config
        self.encoder=copy.deepcopy(baseline.encoder) if baseline is not None else VQEncoder(VQConfig())
        self.quantizer=copy.deepcopy(baseline.quantizer) if baseline is not None else VectorQuantizer(VQConfig())
        self.encoder.requires_grad_(False);self.quantizer.requires_grad_(False)
        self.decoder=LargeDecoder(config)
        self.train(False)

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();self.quantizer.eval();return self

    def forward(self,indices):
        validate_indices(indices)
        with torch.no_grad():z=self.quantizer.embedding(indices).permute(0,3,1,2).contiguous()
        return self.decoder(z)


def validate_indices(indices):
    if indices.ndim!=3 or tuple(indices.shape[1:])!=(8,8) or indices.dtype!=torch.long or indices.min()<0 or indices.max()>255:
        raise ValueError("Expected N x 8 x 8 long indices in [0,255]")


class LargeDecoderOnly(nn.Module):
    def __init__(self,config=LargeConfig()):
        super().__init__();self.config=config
        self.codebook=nn.Embedding(256,64);self.codebook.requires_grad_(False)
        self.decoder=LargeDecoder(config)

    def forward(self,indices):
        validate_indices(indices)
        return self.decoder(self.codebook(indices).permute(0,3,1,2).contiguous())


def frozen_hashes(model):
    return {"encoder":state_hash(model.encoder.state_dict()),"codebook":state_hash(model.quantizer.state_dict())}


def save_large(path,model,decoder_only=False,**extra):
    if decoder_only:
        if extra:raise ValueError("Decoder-only export excludes training state")
        state={"codebook.weight":model.quantizer.embedding.weight.detach().cpu()}
        state.update({"decoder."+k:v.detach().cpu() for k,v in model.decoder.state_dict().items()})
    else:state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    bundle={"format_version":1,"codec":"vqface-decoder-large-v1","kind":"decoder" if decoder_only else "frozen-codec",
        "config":asdict(model.config),"state_dict":state,**extra}
    path=vq_output(path);temp=path.with_suffix(path.suffix+".tmp")
    torch.save(bundle,temp);temp.replace(path)


def load_large(path,device="cpu",decoder_only=False):
    bundle=torch.load(path,map_location="cpu",weights_only=True)
    if bundle.get("format_version")!=1 or bundle.get("codec")!="vqface-decoder-large-v1":raise ValueError("Unsupported decoder checkpoint")
    if bundle.get("kind")!=("decoder" if decoder_only else "frozen-codec"):raise ValueError("Wrong checkpoint kind")
    try:
        config=LargeConfig(**bundle["config"])
        model=LargeDecoderOnly(config) if decoder_only else FrozenCodec(config=config)
        model.load_state_dict(bundle["state_dict"],strict=True)
    except (RuntimeError,TypeError,KeyError) as error:raise ValueError("Incompatible decoder checkpoint") from error
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError("Nonfinite model weights")
    return model.to(resolve_device(device)).eval(),bundle
