"""
Model-input preprocessing pipeline — Stage D.

Preserves ink darkness, absolute size, slant, and baseline slope: grayscale
with background flattening only. No Otsu binarization, no deskew, no
square/stretch resize anywhere in the model-input path (CLAUDE.md Accuracy
Strategy). ink_mask_for_stats() is the only function that may use Otsu
thresholding, and it exists solely for descriptive statistics — it never
touches the image written to disk.

Reads crops from DATASET/raw-3Page / raw-4Page (read-only, never written
to) and writes DATASET/processed/<participant_id>/<cell>.png plus
DATASET/metadata/processed_manifest.csv.

Run from the project root:

    python -m src.preprocessing.pipeline
    python -m src.preprocessing.pipeline --batch 001-050
    python -m src.preprocessing.pipeline --overwrite
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import config


def flatten_background(img: np.ndarray, ksize: int) -> np.ndarray:
    """Flatten uneven paper background by dividing by a large-kernel blur.

    Rescales so paper areas land near 255, preserving ink darkness
    relative to the local background. No binarization, no thresholding.
    """
    img_f = img.astype(np.float32)
    k = ksize if ksize % 2 == 1 else ksize + 1
    background = cv2.GaussianBlur(img_f, (k, k), 0)
    background = np.clip(background, 1.0, 255.0)
    flattened = (img_f / background) * 255.0
    return np.clip(flattened, 0, 255).astype(np.uint8)


def ink_mask_for_stats(img: np.ndarray) -> np.ndarray:
    """Binary ink mask for descriptive statistics ONLY.

    Otsu thresholding is acceptable here -- this mask is never used to
    build the model-input image, only to compute statistics about it.
    """
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def preprocess_crop(img: np.ndarray, task_family: str, cfg) -> np.ndarray:
    """Full per-crop preprocessing. Never binarizes, deskews, or stretches.

    Order: flatten background -> optional median denoise -> fixed-factor
    resize (INTER_AREA, aspect ratio preserved) -> place on a white canvas
    (pad, never stretch) -> optional invert. Returns a uint8 image sized
    exactly cfg.canvas_size[task_family].
    """
    flattened = flatten_background(img, cfg.background_blur_ksize)

    if cfg.denoise == "median3":
        denoised = cv2.medianBlur(flattened, 3)
    else:
        denoised = flattened

    scale = cfg.scale_factor[task_family]
    new_w = max(1, round(denoised.shape[1] * scale))
    new_h = max(1, round(denoised.shape[0] * scale))
    resized = cv2.resize(denoised, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas_w, canvas_h = cfg.canvas_size[task_family]
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    h = min(resized.shape[0], canvas_h)
    w = min(resized.shape[1], canvas_w)
    y_off = (canvas_h - h) // 2
    x_off = (canvas_w - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = resized[:h, :w]

    if cfg.invert:
        canvas = 255 - canvas

    return canvas


def _load_qc_dropped(meta_dir: Path) -> set:
    """Return {(participant_id, cell)} marked dropped=True in qc_log.csv, if present."""
    qc_log_path = meta_dir / "qc_log.csv"
    if not qc_log_path.is_file():
        return set()
    qc_log = pd.read_csv(qc_log_path, dtype={"participant_id": str})
    if "dropped" not in qc_log.columns:
        return set()
    dropped = qc_log[qc_log["dropped"].astype(bool)]
    return set(zip(dropped["participant_id"], dropped["cell"]))


def _filter_batch(df: pd.DataFrame, batch: str) -> pd.DataFrame:
    start_s, end_s = batch.split("-")
    start, end = int(start_s), int(end_s)
    numeric_id = pd.to_numeric(df["participant_id"], errors="coerce")
    return df[numeric_id.between(start, end)].reset_index(drop=True)


def _contact_sheet(pairs: list, out_path: Path) -> None:
    """pairs: list of (label, before_img, after_img) uint8 2D arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    n = len(pairs)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.2 * n), squeeze=False)
    for i, (label, before, after) in enumerate(pairs):
        axes[i][0].imshow(before, cmap="gray", vmin=0, vmax=255)
        axes[i][0].axis("off")
        axes[i][1].imshow(after, cmap="gray", vmin=0, vmax=255)
        axes[i][1].axis("off")
        axes[i][0].set_title(f"{label} before" if i == 0 else "", fontsize=8)
        axes[i][1].set_title(f"{label} after" if i == 0 else "", fontsize=8)
        axes[i][0].text(
            -0.1, 0.5, label, transform=axes[i][0].transAxes, fontsize=7, ha="right"
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(batch: str = None, overwrite: bool = False) -> Path:
    """Preprocess every (non-dropped) crop in crop_manifest.csv.

    Writes DATASET/processed/<participant_id>/<cell>.png and
    DATASET/metadata/processed_manifest.csv (crop_manifest columns plus
    processed_path). Never writes inside raw-3Page or raw-4Page. Returns
    the path to processed_manifest.csv.
    """
    cfg = config.preprocessing
    meta_dir = config.paths.metadata_dir

    manifest_path = meta_dir / "crop_manifest.csv"
    if not manifest_path.is_file():
        print(f"ERROR: {manifest_path} not found.")
        print("  Run python -m src.data.crop_manifest first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path, dtype={"participant_id": str})
    if batch:
        df = _filter_batch(df, batch)

    dropped = _load_qc_dropped(meta_dir)
    df = df[
        ~df.apply(lambda r: (r["participant_id"], r["cell"]) in dropped, axis=1)
    ].reset_index(drop=True)

    processed_dir = config.paths.processed_dir
    processed_paths = []
    before_after_samples: dict = {}
    rng = random.Random(config.training.seed)
    sample_participants = set(
        rng.sample(
            sorted(df["participant_id"].unique()),
            k=min(10, df["participant_id"].nunique()),
        )
    )

    for _, row in df.iterrows():
        pid = row["participant_id"]
        cell = row["cell"]
        task_family = row["task_family"]

        out_dir = processed_dir / pid
        out_path = out_dir / f"{cell}.png"

        if out_path.is_file() and not overwrite:
            processed_paths.append(str(out_path))
            continue

        with Image.open(row["path"]) as img:
            gray = np.array(img.convert("L"))

        processed = preprocess_crop(gray, task_family, cfg)

        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(processed).save(out_path)
        processed_paths.append(str(out_path))

        if pid in sample_participants and pid not in before_after_samples:
            before_after_samples[pid] = (f"{pid}/{cell}", gray, processed)

    df = df.copy()
    df["processed_path"] = processed_paths

    out_manifest_path = meta_dir / "processed_manifest.csv"
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest_path, index=False)

    qc_dir = config.paths.results_dir / "qc"
    _contact_sheet(
        list(before_after_samples.values()), qc_dir / "preprocess_before_after.png"
    )

    return out_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess crops for model input.")
    parser.add_argument("--batch", help="Participant code range, e.g. 001-050")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess crops even if an output file already exists.",
    )
    args = parser.parse_args()

    out_path = run(batch=args.batch, overwrite=args.overwrite)
    print(f"Processed manifest written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
