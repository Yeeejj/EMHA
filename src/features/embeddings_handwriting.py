"""
EXPLORATORY handwriting-encoder embeddings for word and cursive crops — Stage D.

*** EXPLORATORY ONLY. Not part of the pre-registered analysis and never an
input to the pre-registered ensemble (LR-handcrafted + LR-ResNet18 +
CNN-HMM). Any result from these features is reported as exploratory. ***

Why exploratory: the encoder's own image processor resizes every crop to a
fixed square (384x384 for TrOCR) without preserving aspect ratio. That
changes absolute size and stretches word crops ~3.3x and cursive crops ~6.6x
vertically, discarding exactly the size, slant, and spacing signal the
pre-registered pipeline is designed to keep (CLAUDE.md Accuracy Strategy).

Model (HandwritingEmbeddingConfig): microsoft/trocr-base-handwritten, MIT
licence, ViT-B/16 encoder fine-tuned on IAM. Only the encoder is kept: the
VisionEncoderDecoderModel checkpoint is loaded and its .encoder retained
(the checkpoint is a single file, so the full file is downloaded once).
Inputs are the processed crops, un-inverted to dark-on-white (TrOCR's
training polarity) when they are stored inverted, then the model's own
AutoImageProcessor. Embedding = mean of the last hidden state over patch
tokens (CLS/distillation prefix tokens excluded). Frozen, eval, no_grad.
No scaler or PCA is fit here.

Cache: results/embeddings/handwriting_<task_family>.npz with embeddings,
participant_id, cell, cache_key, model_name, model_commit, and
exploratory=True. Requires the optional dependency `transformers`
(requirements-optional.txt); it is imported only when the encoder loads.

Run from the project root:

    python -m src.features.embeddings_handwriting
    python -m src.features.embeddings_handwriting --device cuda
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

from src.features.embeddings import resolve_device
from src.utils.config import config

EXPLORATORY = True


def load_encoder(name: str, device: str) -> tuple:
    """(encoder, image_processor, commit) for a Hub vision-encoder-decoder model.

    The full VisionEncoderDecoderModel is loaded and only its encoder kept
    (the decoder is dropped). Frozen and in eval mode.
    """
    from transformers import AutoImageProcessor, VisionEncoderDecoderModel

    revision = config.handwriting_embedding.model_revision
    processor = AutoImageProcessor.from_pretrained(name, revision=revision)
    full = VisionEncoderDecoderModel.from_pretrained(name, revision=revision)
    commit = getattr(full.config, "_commit_hash", None) or revision
    encoder = full.encoder
    del full
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder.eval().to(resolve_device(device)), processor, commit


def _num_prefix_tokens(encoder: torch.nn.Module, seq_len: int) -> int:
    """Non-patch tokens at the start of the sequence (CLS, distillation)."""
    cfg = getattr(encoder, "config", None)
    if cfg is None or not hasattr(cfg, "image_size"):
        return 0
    n_patches = (cfg.image_size // cfg.patch_size) ** 2
    return max(0, seq_len - n_patches)


def _load_rgb(path: Path) -> Image.Image:
    """Processed crop as dark-on-white RGB (un-inverted if stored inverted)."""
    with Image.open(path) as img:
        gray = np.array(img.convert("L"))
    if config.handwriting_embedding.uninvert and config.preprocessing.invert:
        gray = 255 - gray
    return Image.fromarray(gray).convert("RGB")


def embed_crops(
    paths: list,
    device: str,
    encoder: torch.nn.Module | None = None,
    processor=None,
) -> np.ndarray:
    """(len(paths), hidden_size) float32 mean-pooled encoder embeddings.

    EXPLORATORY. Loads the configured encoder if encoder/processor are not
    given.
    """
    device = resolve_device(device)
    if encoder is None or processor is None:
        encoder, processor, _ = load_encoder(
            config.handwriting_embedding.model_name, device
        )
    batch_size = config.handwriting_embedding.batch_size

    chunks = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = [_load_rgb(Path(p)) for p in paths[start : start + batch_size]]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            hidden = encoder(pixel_values=pixels.to(device)).last_hidden_state
            skip = _num_prefix_tokens(encoder, hidden.shape[1])
            chunks.append(hidden[:, skip:, :].mean(dim=1).cpu().numpy())
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def cache_key(family_df: pd.DataFrame) -> str:
    """Hash of the family's manifest rows, processed image bytes, and settings."""
    hcfg = config.handwriting_embedding
    h = hashlib.sha256()
    settings = (
        hcfg.model_name,
        hcfg.model_revision,
        hcfg.uninvert,
        config.preprocessing.invert,
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
    """EXPLORATORY: embed processed word and cursive crops, one cache per family.

    Returns the list of npz paths (cached or freshly written).
    """
    hcfg = config.handwriting_embedding
    device = resolve_device(device or hcfg.device)
    manifest_path = config.paths.metadata_dir / "processed_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found; run python -m src.preprocessing.pipeline."
        )
    manifest = pd.read_csv(manifest_path, dtype={"participant_id": str})
    manifest = manifest.sort_values(["participant_id", "cell"]).reset_index(drop=True)

    out_dir = config.paths.results_dir / hcfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("EXPLORATORY handwriting embeddings -- not part of the pre-registered set.")
    encoder = processor = commit = None
    written = []
    for family in hcfg.task_families:
        family_df = manifest[manifest["task_family"] == family]
        if family_df.empty:
            continue
        out_path = out_dir / f"{hcfg.output_prefix}_{family}.npz"
        key = cache_key(family_df)
        if _cache_matches(out_path, key):
            print(f"{family:8s}: cache up to date ({len(family_df)} crops) {out_path}")
            written.append(out_path)
            continue

        if encoder is None:
            encoder, processor, commit = load_encoder(hcfg.model_name, device)
        paths = [Path(p) for p in family_df["processed_path"]]
        embeddings = embed_crops(paths, device, encoder=encoder, processor=processor)
        np.savez_compressed(
            out_path,
            embeddings=embeddings,
            participant_id=family_df["participant_id"].to_numpy(dtype=str),
            cell=family_df["cell"].to_numpy(dtype=str),
            cache_key=np.array(key),
            model_name=np.array(hcfg.model_name),
            model_commit=np.array(str(commit)),
            exploratory=np.array(EXPLORATORY),
        )
        print(f"{family:8s}: embedded {embeddings.shape} on {device} -> {out_path}")
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EXPLORATORY: cache handwriting-encoder embeddings."
    )
    parser.add_argument(
        "--device", default=None, help="cpu | cuda (default: config, auto)"
    )
    args = parser.parse_args()
    try:
        run(device=args.device)
    except ImportError:
        print(
            "ERROR: transformers is not installed (optional dependency); "
            "pip install -r requirements-optional.txt"
        )
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
