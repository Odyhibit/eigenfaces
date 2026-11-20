"""Standard-loss VQ-VAE training, utilization monitoring and full-state resume."""
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compare import quality_metrics
from .evaluate import digest, sheet, write_csv
from .model import DEFAULT_DATASET, ROOT
from .neural import resolve_device
from .neural_data import FaceDataset, validate_manifest
from .neural_train import seed_worker, ssim_loss
from .vq_data import DEFAULT_BASE_SPLIT, make_vq_split
from .vq_model import VQVAE, load_vq, save_vq, vq_output


def edge_loss(prediction, target):
    """L1 difference of signed first differences, equal horizontal/vertical weight."""
    x, y = prediction.float(), target.float()
    horizontal = F.l1_loss(x[..., 1:] - x[..., :-1], y[..., 1:] - y[..., :-1])
    vertical = F.l1_loss(x[..., 1:, :] - x[..., :-1, :], y[..., 1:, :] - y[..., :-1, :])
    return .5 * (horizontal + vertical)


def vq_loss(prediction, target, codebook, commitment, settings):
    l1 = F.l1_loss(prediction.float(), target.float())
    structural = ssim_loss(prediction, target) if settings["ssim_weight"] else l1.new_zeros(())
    gradient = edge_loss(prediction, target) if settings["edge_weight"] else l1.new_zeros(())
    reconstruction = settings["l1_weight"] * l1 + settings["ssim_weight"] * structural + settings["edge_weight"] * gradient
    total = reconstruction + settings["codebook_weight"] * codebook + settings["commitment_weight"] * commitment
    return total, {"reconstruction_loss": reconstruction, "l1": l1, "ssim_loss": structural,
                   "edge_loss": gradient, "codebook_loss": codebook, "commitment_loss": commitment}


def codebook_statistics(indices):
    q = np.asarray(indices)
    if q.ndim != 3 or tuple(q.shape[1:]) != (8, 8) or not len(q) or q.dtype.kind not in "iu" or q.min() < 0 or q.max() > 255:
        raise ValueError("Expected nonempty N x 8 x 8 indices in [0,255]")
    counts = np.bincount(q.reshape(-1).astype(np.int64), minlength=256)
    p = counts[counts > 0] / counts.sum()
    active = int(np.count_nonzero(counts))
    perplexity = float(np.exp(-(p * np.log(p)).sum()))
    positions = np.stack([np.bincount(q[:, y, x].astype(np.int64), minlength=256) for y in range(8) for x in range(8)])
    return {"active_entries": active, "perplexity": perplexity, "unused_percent": 100 * (256 - active) / 256,
            "constant_positions": int(np.sum(np.count_nonzero(positions, axis=1) == 1)),
            "unique_payloads": len(set(row.astype(np.uint8).tobytes() for row in q))}, counts, positions


def severe_collapse(stats, epoch, grace=5, min_active=8, min_perplexity=4.):
    return epoch > grace and (stats["active_entries"] < min_active or stats["perplexity"] < min_perplexity
                              or stats.get("all_payloads_identical", False))


@torch.inference_mode()
def initialize_codebook(model, dataset, device, batch_size, seed):
    """One documented train-only initialization; never reseed during training."""
    model.eval()
    features = []
    count = min(len(dataset), max(4, min(256, 2 * batch_size)))
    for start in range(0, count, batch_size):
        x = torch.stack([dataset[i] for i in range(start, min(start + batch_size, count))]).to(device)
        features.append(model.encoder(x).float().permute(0, 2, 3, 1).reshape(-1, 64).cpu())
    pool = torch.cat(features)
    if len(pool) < 256:
        raise ValueError("Need at least four training images to initialize 256 vectors")
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed + 1))[:256]
    model.quantizer.embedding.weight.copy_(pool[order].to(device))
    return {"method": "256 sampled encoder vectors before optimization, training files only",
            "training_images": count, "feature_vectors": len(pool), "seed": seed + 1}


def plot_training(history, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for split in ("train", "validation"):
        axes[0, 0].plot([r["epoch"] for r in history], [r[f"{split}_loss"] for r in history], label=split)
    axes[0, 0].set_ylabel("Total VQ objective")
    axes[0, 0].legend()
    for ax, key in zip(list(axes.flat)[1:], ("validation_mse", "validation_ssim", "validation_perplexity", "validation_active_entries", "learning_rate")):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history])
        ax.set_ylabel(key)
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_vq(dataset=DEFAULT_DATASET, output=ROOT / "models/vq64", samples=50000,
             validation_samples=2000, epochs=100, batch_size=128, learning_rate=.0002,
             seed=42, workers=None, device="auto", amp=False, patience=15,
             l1_weight=1., ssim_weight=.2, edge_weight=.2, codebook_weight=1., commitment_weight=.25,
             max_grad_norm=1., resume=None, base_split=DEFAULT_BASE_SPLIT, split_manifest=None,
             collapse_grace=5, collapse_patience=3, collapse_min_active=8, collapse_min_perplexity=4.,
             experiment_observer=None):
    weights = (l1_weight, ssim_weight, edge_weight, codebook_weight, commitment_weight)
    if (epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0 or max_grad_norm <= 0
            or not all(math.isfinite(v) for v in (*weights, learning_rate, max_grad_norm, collapse_min_perplexity))
            or min(weights) < 0 or sum(weights[:3]) <= 0 or min(weights[3:]) <= 0
            or collapse_grace < 0 or collapse_patience < 1 or not 1 <= collapse_min_active <= 256
            or not 1 <= collapse_min_perplexity <= 256):
        raise ValueError("Invalid VQ training settings")
    device = resolve_device(device)
    workers = min(4, os.cpu_count() or 1) if workers is None else workers
    if workers < 0:
        raise ValueError("Workers must be nonnegative")
    output = vq_output(Path(output) / "config.json", dataset).parent
    if not resume and any(output.iterdir()):
        raise ValueError("VQ run directory is not empty; choose a new --output or resume")
    settings = {"batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
                "l1_weight": l1_weight, "ssim_weight": ssim_weight, "edge_weight": edge_weight,
                "codebook_weight": codebook_weight, "commitment_weight": commitment_weight,
                "max_grad_norm": max_grad_norm, "amp": bool(amp and device.type == "cuda"),
                "collapse_grace": collapse_grace, "collapse_patience": collapse_patience,
                "collapse_min_active": collapse_min_active, "collapse_min_perplexity": collapse_min_perplexity,
                "scheduler": "none", "patience": patience}
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(seed)
    start, best, stale, collapse_streak = 0, float("inf"), 0, 0
    history, frequencies, checkpoint = [], [], None
    if resume:
        model, checkpoint = load_vq(resume, device)
        required = {"optimizer", "scaler", "loader_rng", "torch_rng", "python_rng", "cuda_rng", "history", "initialization"}
        if required - checkpoint.keys():
            raise ValueError("VQ checkpoint is not resumable")
        if checkpoint.get("checkpoint_role") != "latest":
            raise ValueError("Resume VQ training from last.pt, not the selected weights checkpoint")
        if checkpoint["training_settings"] != settings:
            raise ValueError("VQ resume training settings differ")
        if checkpoint.get("stop_reason") == "severe_codebook_collapse":
            raise ValueError("This run stopped for severe collapse; inspect it before choosing a new experiment")
        manifest = validate_manifest(checkpoint["split_manifest"], dataset)
        disk = Path(resume).parent / "split.json"
        if disk.exists() and json.loads(disk.read_text()) != manifest:
            raise ValueError("VQ checkpoint and saved split differ")
        start, best, stale = checkpoint["epoch"], checkpoint["best_loss"], checkpoint["stale_epochs"]
        collapse_streak = checkpoint["collapse_streak"]
        history, frequencies = checkpoint["history"], checkpoint["frequencies"]
        initialization = checkpoint["initialization"]
        if start >= epochs:
            raise ValueError("--epochs is the total target and must exceed the resumed epoch")
    else:
        manifest = make_vq_split(Path(dataset), samples, validation_samples, seed, base_split, split_manifest)
        model = VQVAE().to(device)
        initialization = initialize_codebook(model, FaceDataset(dataset, manifest["train"]), device, batch_size, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if "numpy_rng" in checkpoint:
            state = checkpoint["numpy_rng"]
            np.random.set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
                                 state["position"], state["has_gauss"], state["cached_gaussian"]))
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if checkpoint.get("best_state") is None:
            raise ValueError("VQ resume is missing the best-model state")
        current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(checkpoint["best_state"])
        save_vq(output / "decoder.pt", model, decoder_only=True)
        # Resuming to another directory preserves the original best checkpoint.
        save_vq(output / "best.pt", model, epoch=checkpoint["best_epoch"], validation_loss=best,
                split_manifest=manifest, training_settings=settings, initialization=initialization,
                checkpoint_role="selection")
        model.load_state_dict(current)
    best_state = checkpoint.get("best_state") if checkpoint else None
    best_epoch = checkpoint.get("best_epoch", 0) if checkpoint else 0
    monitor_names = [manifest["validation"][i] for i in np.random.default_rng(seed + 1).choice(len(manifest["validation"]), min(20, len(manifest["validation"])), replace=False)]
    # Optional experiment reporting only; the observer must not update weights or RNGs.
    if experiment_observer is not None:
        experiment_observer("initialized", model=model, optimizer=optimizer, manifest=manifest,
                            monitor_names=monitor_names, initialization=initialization)
    (output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({"model": model.config.__dict__, "training": settings,
                                                   "initialization": initialization, "monitor_names": monitor_names}, indent=2) + "\n")
    loaders = {split: DataLoader(FaceDataset(dataset, manifest[split]), batch_size=batch_size, shuffle=split == "train",
                                num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                                worker_init_fn=seed_worker, generator=generator) for split in ("train", "validation")}
    print(f"VQ device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, "
          f"train={len(manifest['train'])}, validation={len(manifest['validation'])}, grid=8x8, entries=256", flush=True)
    for epoch in range(start + 1, epochs + 1):
        began = time.perf_counter()
        row = {"epoch": epoch}
        validation_original, validation_prediction = [], []
        epoch_positions = {}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            total, count, codes = {}, 0, []
            with torch.set_grad_enabled(training):
                for batch in tqdm(loader, desc=f"VQ {epoch}/{epochs} {split}"):
                    batch = batch.to(device, non_blocking=device.type == "cuda")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=settings["amp"]):
                        prediction, indices, codebook, commitment = model(batch)
                    loss, components = vq_loss(prediction, batch, codebook, commitment, settings)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite VQ loss; no silent restart")
                    if training:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=not settings["amp"])
                        scaler.step(optimizer)
                        scaler.update()
                    for key, value in {"loss": loss, **components}.items():
                        total[key] = total.get(key, 0.) + value.detach().item() * len(batch)
                    count += len(batch)
                    codes.append(indices.detach().cpu().numpy().astype(np.uint8))
                    if not training:
                        validation_original.append(batch.cpu().numpy().reshape(len(batch), -1))
                        validation_prediction.append(prediction.float().cpu().numpy().reshape(len(batch), -1))
            indices = np.concatenate(codes)
            stats, counts, positions = codebook_statistics(indices)
            epoch_positions[split] = positions
            row.update({f"{split}_{key}": value / count for key, value in total.items()})
            row.update({f"{split}_{key}": value for key, value in stats.items()})
            frequencies.extend({"epoch": epoch, "split": split, "entry": j, "count": int(counts[j]),
                                "fraction": float(counts[j] / counts.sum())} for j in range(256))
        original, prediction = np.concatenate(validation_original), np.concatenate(validation_prediction)
        measured = [quality_metrics(x, y) for x, y in zip(original, prediction)]
        row.update(validation_mse=float(np.mean([m["mse"] for m in measured])),
                   validation_ssim=float(np.mean([m["ssim"] for m in measured])),
                   learning_rate=optimizer.param_groups[0]["lr"], epoch_seconds=time.perf_counter() - began)
        # Select on the explicitly defined reconstruction objective. Codebook and
        # commitment losses are logged separately and never treated as quality.
        selected_loss = row["validation_reconstruction_loss"]
        if selected_loss < best:
            best, best_epoch, stale = selected_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        current_stats = {k: row[f"validation_{k}"] for k in ("active_entries", "perplexity")}
        current_stats["all_payloads_identical"] = len(manifest["validation"]) > 1 and row["validation_unique_payloads"] == 1
        collapse_streak = collapse_streak + 1 if severe_collapse(current_stats, epoch, collapse_grace, collapse_min_active, collapse_min_perplexity) else 0
        stop = "severe_codebook_collapse" if collapse_streak >= collapse_patience else "early_stopping" if stale >= patience else "max_epochs" if epoch == epochs else None
        row.update(best_reconstruction_loss=best, collapse_streak=collapse_streak)
        history.append(row)
        numpy_state = np.random.get_state()
        extra = dict(epoch=epoch, best_epoch=best_epoch, best_loss=best, validation_loss=selected_loss,
                     stale_epochs=stale, collapse_streak=collapse_streak, stop_reason=stop,
                     optimizer=optimizer.state_dict(), best_state=best_state, checkpoint_role="latest",
                     scaler=scaler.state_dict(), split_manifest=manifest, training_settings=settings,
                     initialization=initialization, history=history, frequencies=frequencies,
                     loader_rng=generator.get_state(), torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     numpy_rng={"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(), "position": numpy_state[2],
                                "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        save_vq(output / "last.pt", model, **extra)
        if epoch == best_epoch:
            save_vq(output / "best.pt", model, epoch=epoch, validation_loss=best, split_manifest=manifest,
                    training_settings=settings, initialization=initialization, checkpoint_role="selection")
            save_vq(output / "decoder.pt", model, decoder_only=True)
        write_csv(output / "history.csv", history)
        write_csv(output / "codebook_usage.csv", frequencies)
        np.savez_compressed(output / f"positions_{epoch:03}.npz", **epoch_positions)
        plot_training(history, output / "training.png")
        if epoch % 5 == 0 or stop:
            positions = [manifest["validation"].index(name) for name in monitor_names]
            panels, labels = [], []
            for i, name in zip(positions, monitor_names):
                panels.extend([original[i], prediction[i]])
                labels.extend([Path(name).name + " original", "VQ reconstruction"])
            sheet(output / f"validation_{epoch:03}.png", panels, labels, 8, f"VQ validation epoch {epoch}")
            fig, ax = plt.subplots(figsize=(13, 4))
            im = ax.imshow(epoch_positions["validation"] / len(manifest["validation"]), aspect="auto", cmap="magma")
            ax.set(xlabel="Codebook index", ylabel="Spatial position y*8+x")
            fig.colorbar(im, label="Frequency at each position")
            fig.tight_layout()
            fig.savefig(output / "codebook_positions.png", dpi=150)
            plt.close(fig)
        print(f"VQ epoch {epoch}: val reconstruction={selected_loss:.6f}, MSE={row['validation_mse']:.6f}, "
              f"SSIM={row['validation_ssim']:.4f}, active={row['validation_active_entries']}/256, "
              f"perplexity={row['validation_perplexity']:.2f}, lr={row['learning_rate']:g}", flush=True)
        if experiment_observer is not None:
            experiment_observer("epoch", epoch=epoch, row=row, stop=stop,
                                original=original, prediction=prediction, monitor_names=monitor_names,
                                validation_names=manifest["validation"])
        if stop:
            if stop == "severe_codebook_collapse":
                print("STOP: severe codebook collapse. Checkpoint and usage diagnostics saved; no reseeding or architecture change.", flush=True)
            break
    return history
