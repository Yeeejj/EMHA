"""
Frozen ImageNet ResNet18 embeddings for every processed crop — Stage D.

torchvision resnet18 with ResNet18_Weights.IMAGENET1K_V1 (multi-weight API),
fc replaced by Identity -> 512-dim global-average-pooled features, eval mode,
no gradients. Inputs are the processed crops (DATASET/processed, listed in
processed_manifest.csv) through the project's eval transform; its
mean=0.5/std=0.5 normalization is undone, the channel is repeated to 3, and
ImageNet mean/std (taken from the weights' own transform) are applied. The
weights' resize/center-crop is NOT used: crops keep their fixed per-family
canvas size (adaptive pooling handles any size), so nothing is stretched.

No scaler or PCA is fit here (EmbeddingConfig.pca_components is applied
later, inside CV folds, on training participants only).

Cache: results/embeddings/resnet18_<task_family>.npz with arrays
embeddings (n, 512) float32, participant_id, cell, and cache_key. The key
hashes the family's manifest rows (participant_id, cell), the bytes of every
processed image, and the model/transform settings; a matching key skips
recomputation.

Run from the project root:

    python -m src.features.embeddings
    python -m src.features.embeddings --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from src.data.transforms import NORMALIZE_MEAN, NORMALIZE_STD, get_eval_transform
from src.utils.config import config

ARCH = "resnet18"
EMBED_DIM = 512
FAMILIES = ("drawing", "word", "cursive")

# ImageNet statistics from the weights' bundled inference transform (the
# same for every ResNet18_Weights member; available without a download).
_IMAGENET_TRANSFORM = ResNet18_Weights.IMAGENET1K_V1.transforms()
IMAGENET_MEAN = list(_IMAGENET_TRANSFORM.mean)
IMAGENET_STD = list(_IMAGENET_TRANSFORM.std)


def resolve_device(device: str | None) -> str:
    """'auto'/None -> 'cuda' if available else 'cpu'; otherwise unchanged."""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_backbone(device: str, weights: str | None = "config") -> torch.nn.Module:
    """Frozen ResNet18 feature extractor: fc = Identity, eval, no grad.

    weights defaults to config.embedding.weights (a ResNet18_Weights member
    name, e.g. "IMAGENET1K_V1"); None gives random init (tests, no download).
    """
    if weights == "config":
        weights = config.embedding.weights
    model = resnet18(weights=ResNet18_Weights[weights] if weights else None)
    model.fc = torch.nn.Identity()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.eval().to(resolve_device(device))


def _to_imagenet_input(x: torch.Tensor) -> torch.Tensor:
    """Eval-transform output (1, H, W) -> ImageNet-normalized (3, H, W)."""
    x = x * NORMALIZE_STD[0] + NORMALIZE_MEAN[0]  # back to [0, 1]
    x = x.repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def embed_crops(
    paths: list,
    task_family: str,
    device: str,
    model: torch.nn.Module | None = None,
) -> np.ndarray:
    """(len(paths), 512) float32 embeddings, in the order of paths.

    All crops of one task family share one canvas size, so they batch
    directly. Loads the backbone if model is not given.
    """
    device = resolve_device(device)
    if model is None:
        model = load_backbone(device)
    transform = get_eval_transform(config, task_family)
    batch_size = config.embedding.batch_size

    out = np.zeros((len(paths), EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as img:
                    batch.append(_to_imagenet_input(transform(img.convert("L"))))
            feats = model(torch.stack(batch).to(device))
            out[start : start + len(batch)] = feats.cpu().numpy()
    return out


def cache_key(family_df: pd.DataFrame) -> str:
    """Hash of the family's manifest rows, processed image bytes, and settings."""
    h = hashlib.sha256()
    settings = (
        ARCH,
        str(config.embedding.weights),
        IMAGENET_MEAN,
        IMAGENET_STD,
        NORMALIZE_MEAN,
        NORMALIZE_STD,
    )
    h.update(repr(settings).encode())
    for pid, cell, path in zip(
        family_df["participant_id"], family_df["cell"], family_df["processed_path"]
    ):
        h.update(f"{pid}|{cell}|".encode())
        h.update(hashlib.sha256(Path(path).read_bytes()).digest())
    return h.hexdigest()


def _cache_matches(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as cached:
        return "cache_key" in cached.files and str(cached["cache_key"]) == key


def run(device: str | None = None) -> list:
    """Embed every processed crop, one cache file per task family.

    Returns the list of npz paths (cached or freshly written).
    """
    device = resolve_device(device or config.embedding.device)
    manifest_path = config.paths.metadata_dir / "processed_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found; run python -m src.preprocessing.pipeline."
        )
    manifest = pd.read_csv(manifest_path, dtype={"participant_id": str})
    manifest = manifest.sort_values(["participant_id", "cell"]).reset_index(drop=True)

    out_dir = config.paths.results_dir / config.embedding.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    written = []
    for family in FAMILIES:
        family_df = manifest[manifest["task_family"] == family]
        if family_df.empty:
            continue
        out_path = out_dir / f"{ARCH}_{family}.npz"
        key = cache_key(family_df)
        if _cache_matches(out_path, key):
            print(f"{family:8s}: cache up to date ({len(family_df)} crops) {out_path}")
            written.append(out_path)
            continue

        if model is None:
            model = load_backbone(device)
        paths = [Path(p) for p in family_df["processed_path"]]
        embeddings = embed_crops(paths, family, device, model=model)
        np.savez_compressed(
            out_path,
            embeddings=embeddings,
            participant_id=family_df["participant_id"].to_numpy(dtype=str),
            cell=family_df["cell"].to_numpy(dtype=str),
            cache_key=np.array(key),
        )
        print(f"{family:8s}: embedded {embeddings.shape} on {device} -> {out_path}")
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache frozen ResNet18 embeddings.")
    parser.add_argument(
        "--device", default=None, help="cpu | cuda (default: config, auto)"
    )
    args = parser.parse_args()
    try:
        run(device=args.device)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
