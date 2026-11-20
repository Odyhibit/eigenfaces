"""Run the frozen frontal-specialization experiment; control is opt-in only."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .evaluate import sheet
from .model import ROOT
from .phase7_data import DATASET, RESULTS, dump, prepare, sha, validate_groups
from .preprocess import preprocess_image
from .vq_codec import decode_payload, encode_array
from .vq_model import VQVAE, load_vq
from .vq_train import train_vq

MODEL = ROOT / "models/vq64_frontal"
GENERAL = ROOT / "models/vq64/best.pt"


def state_hash(state):
    result = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        result.update(key.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def verify_sources(output=RESULTS):
    metadata = json.loads((output / "manifest_metadata.json").read_text())
    for path, expected in metadata["sources"].items():
        if sha(path) != expected:
            raise ValueError(f"Source changed after protocol freeze: {path}")
    for key, expected in metadata["manifests"].items():
        if sha(output / f"{key}.txt") != expected:
            raise ValueError("Prepared filename manifest changed")
    manifest = json.loads((output / "split.json").read_text())
    basenames = {k: [Path(n).name for n in manifest[k]] for k in ("train", "validation", "test")}
    forbidden = set((output / "historical_holdouts.txt").read_text().splitlines())
    strict = set((ROOT / "output/frontal_curation/frontal_strict.txt").read_text().splitlines())
    validate_groups(basenames, forbidden, strict)
    for key in basenames:
        if basenames[key] != (output / f"{key}.txt").read_text().splitlines():
            raise ValueError("Text and JSON manifests differ")
    from .neural_data import validate_manifest
    validate_manifest(manifest, DATASET)
    return manifest


def prepare_configuration(output=RESULTS):
    original = json.loads((ROOT / "models/vq64/config.json").read_text())
    last = torch.load(ROOT / "models/vq64/last.pt", weights_only=True, map_location="cpu")
    general, selected = load_vq(GENERAL)
    if selected["training_settings"] != original["training"] or last["training_settings"] != original["training"]:
        raise ValueError("Recorded Phase 5 training settings disagree")
    if general.config.__dict__ != VQVAE().config.__dict__:
        raise ValueError("Existing architecture differs from scratch model")
    parameters = {k:v for k,v in last["optimizer"]["param_groups"][0].items() if k != "params"}
    expected = {"lr": original["training"]["learning_rate"], "betas": (.9,.999), "eps": 1e-8,
                "weight_decay": .01, "amsgrad": False, "maximize": False}
    if any(parameters[k] != v for k,v in expected.items()):
        raise ValueError("Optimizer differs from trainer's unchanged AdamW settings")
    if last["epoch"] != 100 or last["stop_reason"] != "max_epochs":
        raise ValueError("Cannot establish the requested original 100-epoch limit")
    config = {"model": original["model"], "training": original["training"], "epochs": last["epoch"],
        "dataset": str(DATASET), "split_manifest": str(output / "split.json"), "output": str(MODEL),
        "initialization": "Fresh randomly initialized network; train-only sampled-vector codebook initialization, unchanged Phase 5 method",
        "original_selected_epoch": selected["epoch"], "original_selected_sha256": sha(GENERAL),
        "original_state_sha256": state_hash(general.state_dict()), "original_optimizer": parameters,
        "checkpoint_selection": "Minimum validation_reconstruction_loss; codebook and commitment terms excluded",
        "scheduler": "none", "workers": 4, "device": "cuda", "control_started": False,
        "manifest_hashes": {k: sha(output/f"{k}.txt") for k in ("train","validation","test")},
        "split_sha256": sha(output / "split.json"), "evaluation_protocol_sha256": sha(output / "evaluation_protocol.json"),
        "pretraining_software": {"python":platform.python_version(),"torch":str(torch.__version__),"cuda":torch.version.cuda},
        "codec_source_hashes": {str(ROOT/"src"/n): sha(ROOT/"src"/n) for n in ("vq_model.py","vq_codec.py","preprocess.py")}}
    dump(output / "experiment.json", config)
    control = {**config, "output": str(ROOT / "models/vq64_control"),
               "split_manifest": str(output / "control_split.json"), "status": "Prepared, NOT authorized to launch automatically",
               "split_sha256": sha(output / "control_split.json"),
               "training_manifest_sha256": sha(output / "control_train.txt"), "control_started": False}
    control["manifest_hashes"] = {**config["manifest_hashes"], "train": sha(output / "control_train.txt")}
    dump(output / "control_configuration.json", control)
    return config


def verify_preservation(output=RESULTS):
    before = json.loads((output / "protected_inputs.json").read_text())
    changed = [path for path, expected in before.items() if not Path(path).is_file() or sha(path) != expected]
    if changed:
        raise ValueError(f"Protected files changed: {changed[:5]}")
    config = json.loads((output / "experiment.json").read_text())
    if any(sha(path) != value for path,value in config["codec_source_hashes"].items()):
        raise ValueError("Architecture, codec or preprocessing changed")
    dump(output / "preservation_verification.json", {"protected_files":len(before),"all_unchanged":True,
         "architecture_quantization_codec_preprocessing_unchanged":True,"control_started":False})


def run(output=RESULTS, control=False):
    verify_sources(output)
    config = json.loads((output / ("control_configuration.json" if control else "experiment.json")).read_text())
    if not torch.cuda.is_available() or "4080" not in torch.cuda.get_device_name(0):
        raise ValueError("Expected CUDA RTX 4080 unavailable; no silent CPU training substitution")
    destination = Path(config["output"])
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Isolated training directory must be empty; no checkpoint overwrite")
    if sha(config["split_manifest"]) != config["split_sha256"]:
        raise ValueError("Frozen split differs")
    if sha(output / "evaluation_protocol.json") != config["evaluation_protocol_sha256"]:
        raise ValueError("Evaluation protocol changed")
    torch.set_num_threads(4)
    protocol = json.loads((output / "evaluation_protocol.json").read_text())
    reference, _ = load_vq(GENERAL, "cuda")
    decoder, _ = load_vq(ROOT / "models/vq64/decoder.pt", "cuda", decoder_only=True)
    monitor = protocol["monitor_names"]
    source_images = [preprocess_image(DATASET/name) for name in monitor]
    reference_images = [decode_payload(encode_array(x,reference),decoder) for x in source_images]
    del reference, decoder
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sheets = output / ("control_validation_sheets" if control else "validation_sheets")
    sheets.mkdir(exist_ok=True)

    def observe(event, **values):
        if event == "initialized":
            model, optimizer = values["model"], values["optimizer"]
            initialized_hash = state_hash(model.state_dict())
            if optimizer.state or initialized_hash == config["original_state_sha256"]:
                raise ValueError("Experiment did not initialize from scratch")
            if values["monitor_names"] != monitor:
                raise ValueError("Fixed monitoring images differ from protocol")
            metadata = {**config,"initial_state_sha256":initialized_hash,"optimizer_initial_state_entries":len(optimizer.state),
                "model_parameters":sum(p.numel() for p in model.parameters()),"loaded_pretrained_weights":False,
                "gpu":torch.cuda.get_device_name(0),"initialization_record":values["initialization"]}
            dump(destination / "phase7_metadata.json", metadata)
            return
        if values["epoch"] % 5 == 0 or values["stop"]:
            panels, labels = [], []
            for name, original, general in zip(monitor,source_images,reference_images):
                i = values["validation_names"].index(name)
                panels.extend([original,general,values["prediction"][i]])
                labels.extend([Path(name).name+" original","General VQ","Frontal VQ"])
            sheet(sheets/f"epoch_{values['epoch']:03}.png",panels,labels,3,
                  f"Fixed validation monitoring, epoch {values['epoch']}")

    settings = dict(config["training"])
    if settings.pop("scheduler") != "none":
        raise ValueError("Unrecognized scheduler")
    history = train_vq(dataset=DATASET,output=destination,epochs=config["epochs"],workers=config["workers"],
        device="cuda",split_manifest=Path(config["split_manifest"]),base_split=None,
        experiment_observer=observe,**settings)
    last = torch.load(destination / "last.pt",weights_only=True,map_location="cpu")
    dump(output / ("control_training_completion.json" if control else "training_completion.json"), {
        "epochs":len(history),"best_epoch":last["best_epoch"],"best_loss":last["best_loss"],"stop_reason":last["stop_reason"],
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),
        "best_sha256":sha(destination/"best.pt"),"decoder_sha256":sha(destination/"decoder.pt")})
    if last["stop_reason"] == "severe_codebook_collapse":
        raise ValueError("Severe codebook collapse: stopped; no retries or evaluation-driven changes")
    verify_preservation(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","verify","train","evaluate","control"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        prepare_configuration()
    elif args.action == "verify":
        verify_sources()
        verify_preservation()
    elif args.action in ("train","control"):
        run(control=args.action=="control")
    else:
        from .phase7_evaluate import evaluate
        evaluate()


if __name__ == "__main__":
    main()
