"""
Crop preview helper — visually tune crop coordinates.

Given one participant_id, this loads the 24 extracted crops from
DATA/EXTRACTED/{participant_id}/ and tiles them into a single composite image
(each panel titled with its task code), saved to
DATA/METADATA/crop_preview_{participant_id}.png.

Open that PNG in Cursor to check alignment, then adjust DRAWING_CROPS /
WORD_CROPS / CURSIVE_CROPS in src/utils/config.py and re-run extraction.

Run from the project root:

    python -m src.data.preview_crops P001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.config import (  # noqa: E402
    config,
    CURSIVE_CROPS,
    DRAWING_CROPS,
    WORD_CROPS,
)

# All 24 task codes in extraction order: drawing, word table, cursive.
TASK_CODES = (
    list(DRAWING_CROPS) + list(WORD_CROPS) + list(CURSIVE_CROPS)
)

N_COLS = 6
N_ROWS = 4  # 6 x 4 = 24 panels


def imread_unicode(path: Path) -> "np.ndarray | None":
    """Read an image via numpy so non-ASCII Windows paths work. None on fail."""
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def build_preview(pid: str, extracted_dir: Path, out_path: Path) -> int:
    """Tile the participant's crops into one titled composite. Returns missing count."""
    crop_dir = extracted_dir / pid

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(18, 13))
    axes = axes.ravel()
    missing = 0

    for ax, task_code in zip(axes, TASK_CODES):
        crop_path = crop_dir / f"{pid}_{task_code}.png"
        image = imread_unicode(crop_path)
        ax.set_title(task_code, fontsize=8)
        ax.axis("off")
        if image is None:
            missing += 1
            ax.text(
                0.5,
                0.5,
                "MISSING",
                ha="center",
                va="center",
                color="red",
                fontsize=10,
            )
            ax.set_facecolor("0.9")
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Hide any unused panels (none for 24, but defensive).
    for ax in axes[len(TASK_CODES):]:
        ax.axis("off")

    fig.suptitle(f"Crop preview — {pid}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a titled composite of a participant's 24 crops."
    )
    parser.add_argument("participant_id", help="e.g. P001")
    args = parser.parse_args()
    pid = args.participant_id

    extracted_dir = Path(config.data.extracted_dir)
    if not (extracted_dir / pid).is_dir():
        print(
            f"ERROR: no extracted crops for '{pid}' at "
            f"{extracted_dir / pid}. Run extract_content first."
        )
        return 1

    out_path = Path(config.data.metadata_dir) / f"crop_preview_{pid}.png"
    missing = build_preview(pid, extracted_dir, out_path)

    print(f"Preview written : {out_path}")
    print(f"Panels          : {len(TASK_CODES)}")
    print(f"Missing crops   : {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
