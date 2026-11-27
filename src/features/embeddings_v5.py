"""
v5 frozen feature blocks (EXPLORATORY, Deviation 8) -- style and dinov2.

Two label-free blocks for every processed crop, cached like the v1
resnet18_<family>.npz files (src/features/embeddings.py, which is not
changed):

  style   ImageNet ResNet18 (embeddings.load_backbone), same input path as
          embeddings.embed_crops (eval transform, no resize). Forward hooks
          on SignalV5Config.style_layers; each layer gives the per-channel
          mean and std over H x W. layer1-layer3 -> (64 + 128 + 256) x 2 =
          896 dims. These are texture statistics (stroke width, ink density,
          irregularity), not object features.
  dinov2  DINOv2 from torch.hub (SignalV5Config.dinov2_repo / dinov2_model),
          eval, no gradients. The processed crop is un-inverted to
          dark-on-white when it is stored inverted (same rule as
          embeddings_handwriting.py), resized so its long side is
          dinov2_long_side with the aspect ratio kept, padded with white on
          the right and bottom to a multiple of dinov2_patch, repeated to 3
          channels and ImageNet-normalized. Output = CLS token and the mean
          of the patch tokens, concatenated (2 x 384 = 768 dims for ViT-S/14).

Caches: results/embeddings/style_<family>.npz and dinov2_<family>.npz with
arrays embeddings (n, d) float32, participant_id, cell, cache_key, and for
dinov2 also weights_sha256 (hash of the loaded parameters, since torch.hub
code is not pinned to a commit). A matching cache_key skips recomputation.
No scaler or PCA is fit here; that happens inside CV folds (Stage F rules).

    python -m src.features.embeddings_v5 --blocks style,dinov2 --device auto
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.transforms import NORMALIZE_MEAN, NORMALIZE_STD, get_eval_transform
from src.features.embeddings import (
    FAMILIES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _cache_matches,
    _to_imagenet_input,
    load_backbone,
    resolve_device,
)
from src.utils.config import config
from src.utils.paths import read_manifest

BLOCKS = ("style", "dinov2")
WHITE = 255


def output_path(block: str, family: str, cfg=config) -> Path:
    out = cfg.paths.results_dir / cfg.embedding.output_subdir
    return out / f"{block}_{family}.npz"


# ---------------------------------------------------------------- style


def style_dim(model: torch.nn.Module, cfg=config) -> int:
    layers = cfg.signal_v5.style_layers
    channels = [getattr(model, name)[-1].conv2.out_channels for name in layers]
    return 2 * sum(channels)


def style_features(
    paths: list, task_family: str, device: str, model=None, cfg=config
) -> np.ndarray:
    """(len(paths), 896) channel mean/std of the frozen ResNet18 style layers."""
    device = resolve_device(device)
    if model is None:
        model = load_backbone(device)
    layers = list(cfg.signal_v5.style_layers)
    grabbed = {}
    hooks = [
        getattr(model, name).register_forward_hook(
            lambda _m, _i, out, name=name: grabbed.__setitem__(name, out)
        )
        for name in layers
    ]
    transform = get_eval_transform(cfg, task_family)
    batch_size = cfg.signal_v5.batch_size
    out = np.zeros((len(paths), style_dim(model, cfg)), dtype=np.float32)
    try:
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch = []
                for path in paths[start : start + batch_size]:
                    with Image.open(path) as img:
                        batch.append(_to_imagenet_input(transform(img.convert("L"))))
                model(torch.stack(batch).to(device))
                parts = []
                for name in layers:
                    act = grabbed[name].flatten(2)
                    parts += [act.mean(dim=2), act.std(dim=2, unbiased=False)]
                feats = torch.cat(parts, dim=1)
                out[start : start + len(batch)] = feats.cpu().numpy()
    finally:
        for hook in hooks:
            hook.remove()
    return out


# ---------------------------------------------------------------- dinov2


def load_dinov2(device: str, cfg=config) -> torch.nn.Module:
    v5 = cfg.signal_v5
    model = torch.hub.load(v5.dinov2_repo, v5.dinov2_model, trust_repo=True)
    for param in model.parameters():
        param.requires_grad_(False)
    return model.eval().to(resolve_device(device))


def weights_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dinov2_size(height: int, width: int, cfg=config) -> tuple:
    """((resized_h, resized_w), (padded_h, padded_w)) for one canvas size."""
    v5 = cfg.signal_v5
    scale = v5.dinov2_long_side / max(height, width)
    h = max(1, round(height * scale))
    w = max(1, round(width * scale))
    p = v5.dinov2_patch
    return (h, w), (-(-h // p) * p, -(-w // p) * p)


def dinov2_input(path: Path, cfg=config) -> torch.Tensor:
    """Processed crop -> (3, H, W) dark-on-white, ImageNet-normalized."""
    with Image.open(path) as img:
        gray = np.array(img.convert("L"))
    if cfg.signal_v5.uninvert and cfg.preprocessing.invert:
        gray = WHITE - gray
    (h, w), (ph, pw) = dinov2_size(*gray.shape, cfg)
    resized = np.array(Image.fromarray(gray).resize((w, h), Image.BILINEAR))
    canvas = np.full((ph, pw), WHITE, dtype=np.uint8)
    canvas[:h, :w] = resized
    x = torch.from_numpy(canvas).float().div(255.0).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def dinov2_features(paths: list, device: str, model=None, cfg=config) -> np.ndarray:
    """(len(paths), 2 x embed_dim) CLS token + mean patch token."""
    device = resolve_device(device)
    if model is None:
        model = load_dinov2(device, cfg)
    batch_size = cfg.signal_v5.batch_size
    chunks = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack(
                [dinov2_input(Path(p), cfg) for p in paths[start : start + batch_size]]
            )
            out = model.forward_features(batch.to(device))
            cls = out["x_norm_clstoken"]
            patches = out["x_norm_patchtokens"].mean(dim=1)
            chunks.append(torch.cat([cls, patches], dim=1).cpu().numpy())
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# ---------------------------------------------------------------- cache


def cache_key(block: str, family_df: pd.DataFrame, cfg=config) -> str:
    v5 = cfg.signal_v5
    h = hashlib.sha256()
    if block == "style":
        settings = (
            block,
            str(cfg.embedding.weights),
            tuple(v5.style_layers),
            IMAGENET_MEAN,
            IMAGENET_STD,
            NORMALIZE_MEAN,
            NORMALIZE_STD,
        )
    else:
        settings = (
            block,
            v5.dinov2_repo,
            v5.dinov2_model,
            v5.dinov2_long_side,
            v5.dinov2_patch,
            v5.uninvert,
            cfg.preprocessing.invert,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )
    h.update(repr(settings).encode())
    for pid, cell, path in zip(
        family_df["participant_id"], family_df["cell"], family_df["processed_path"]
    ):
        h.update(f"{pid}|{cell}|".encode())
        h.update(hashlib.sha256(Path(path).read_bytes()).digest())
    return h.hexdigest()


def run(blocks=BLOCKS, device: str | None = None, cfg=config) -> list:
    """Compute (or reuse) every requested block for every task family."""
    unknown = set(blocks) - set(BLOCKS)
    if unknown:
        raise ValueError(f"unknown blocks {sorted(unknown)}; choose from {BLOCKS}")
    device = resolve_device(device or cfg.embedding.device)
    manifest_path = cfg.paths.metadata_dir / "processed_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found; run python -m src.preprocessing.pipeline."
        )
    manifest = read_manifest(manifest_path)
    manifest = manifest.sort_values(["participant_id", "cell"]).reset_index(drop=True)
    models = {}
    written = []
    for block in blocks:
        for family in FAMILIES:
            family_df = manifest[manifest["task_family"] == family]
            if family_df.empty:
                continue
            path = output_path(block, family, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            key = cache_key(block, family_df, cfg)
            if _cache_matches(path, key):
                print(f"{block}/{family}: cache up to date ({len(family_df)} crops)")
                written.append(path)
                continue
            began = time.perf_counter()
            paths = [Path(p) for p in family_df["processed_path"]]
            extra = {}
            if block == "style":
                models.setdefault("style", load_backbone(device))
                emb = style_features(paths, family, device, models["style"], cfg)
            else:
                if "dinov2" not in models:
                    models["dinov2"] = load_dinov2(device, cfg)
                    models["dinov2_sha"] = weights_sha256(models["dinov2"])
                emb = dinov2_features(paths, device, models["dinov2"], cfg)
                extra["weights_sha256"] = np.array(models["dinov2_sha"])
            np.savez_compressed(
                path,
                embeddings=emb,
                participant_id=family_df["participant_id"].to_numpy(dtype=str),
                cell=family_df["cell"].to_numpy(dtype=str),
                cache_key=np.array(key),
                **extra,
            )
            print(
                f"{block}/{family}: {emb.shape} on {device} in "
                f"{time.perf_counter() - began:.1f} s -> {path}",
                flush=True,
            )
            written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache v5 style/dinov2 features.")
    parser.add_argument("--blocks", default=",".join(BLOCKS))
    parser.add_argument("--device", default=None, help="cpu | cuda | auto")
    args = parser.parse_args()
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    try:
        run(blocks, device=args.device)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
