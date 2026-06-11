"""
Phase 4 — Extract content: cut 24 task crops from each participant's pages.

Reads DATA/METADATA/page_manifest.csv and, for every participant, extracts:

  * 4 drawing crops from page_drawing (index 2) using config.DRAWING_CROPS
  * 15 word-table crops from page_writing (index 3) using config.WORD_CROPS
  * 5 cursive crops from page_writing (index 3) using config.CURSIVE_CROPS

Each crop is saved to
DATA/EXTRACTED/{participant_id}/{participant_id}_{task_code}.png and every
attempt is recorded in DATA/METADATA/extraction_log.csv.

Crop boxes are placeholders sized to the 1700x2200 scans (config.SCAN_*_PX);
tune them visually with src/data/preview_crops.py.

Run from the project root (all participants, or a subset by id):

    python -m src.data.extract_content
    python -m src.data.extract_content P001
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import (
    config,
    CURSIVE_CROPS,
    DRAWING_CROPS,
    WORD_CROPS,
)

DRAWING_ROLE = "page_drawing"
WRITING_ROLE = "page_writing"

# (page_role, crop map) pairs in the order their crops are produced.
SOURCE_SPECS = [
    (DRAWING_ROLE, DRAWING_CROPS),
    (WRITING_ROLE, {**WORD_CROPS, **CURSIVE_CROPS}),
]


def imread_unicode(path: Path) -> "np.ndarray | None":
    """Read an image via numpy so non-ASCII Windows paths work. None on fail."""
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: "np.ndarray") -> bool:
    """Write an image via numpy so non-ASCII Windows paths work."""
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def crop_and_save(
    image: "np.ndarray", box: tuple, out_path: Path
) -> tuple[bool, str]:
    """Crop *box* from *image* (clamped to bounds) and save it to out_path."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    cx1, cx2 = max(0, min(x1, width)), max(0, min(x2, width))
    cy1, cy2 = max(0, min(y1, height)), max(0, min(y2, height))

    if cx2 <= cx1 or cy2 <= cy1:
        return False, (
            f"empty crop after clamping box {box} to image "
            f"{width}x{height}"
        )

    crop = image[cy1:cy2, cx1:cx2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(out_path, crop):
        return False, "cv2 encode/write failed"
    return True, ""


def read_manifest(manifest_path: Path) -> "OrderedDict[str, dict]":
    """Return participant_id -> {page_role: png_path}, in manifest order."""
    participants: "OrderedDict[str, dict]" = OrderedDict()
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row["participant_id"]
            participants.setdefault(pid, {})[row["page_role"]] = row["png_path"]
    return participants


def extract_participant(
    pid: str, pages: dict, extracted_dir: Path
) -> list[dict]:
    """Extract all 24 crops for one participant; return log rows."""
    log_rows: list[dict] = []
    out_dir = extracted_dir / pid

    for role, crop_map in SOURCE_SPECS:
        png_path = pages.get(role)
        image = imread_unicode(Path(png_path)) if png_path else None

        for task_code, box in crop_map.items():
            out_path = out_dir / f"{pid}_{task_code}.png"

            if not png_path:
                status, reason = "FAIL", f"no {role} row in manifest"
            elif image is None:
                status, reason = "FAIL", f"could not read {role} image"
            else:
                ok, reason = crop_and_save(image, box, out_path)
                status = "SUCCESS" if ok else "FAIL"

            log_rows.append(
                {
                    "participant_id": pid,
                    "task_code": task_code,
                    "output_path": str(out_path),
                    "status": status,
                    "reason": reason,
                }
            )
    return log_rows


def extract_content(
    manifest_path: Path,
    extracted_dir: Path,
    log_path: Path,
    only: "list[str] | None",
) -> None:
    """Extract crops for all (or selected) participants and write the log."""
    print("=" * 60)
    print("Phase 4 - EXTRACT content")
    print("=" * 60)

    if not manifest_path.is_file():
        print(f"ERROR: page manifest not found: {manifest_path}")
        sys.exit(1)

    participants = read_manifest(manifest_path)
    if only:
        missing = [pid for pid in only if pid not in participants]
        for pid in missing:
            print(f"WARNING: requested participant '{pid}' not in manifest.")
        participants = OrderedDict(
            (pid, participants[pid]) for pid in only if pid in participants
        )

    print(f"Page manifest : {manifest_path}")
    print(f"Participants  : {len(participants)}")

    all_rows: list[dict] = []
    for pid, pages in participants.items():
        all_rows.extend(extract_participant(pid, pages, extracted_dir))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "participant_id",
        "task_code",
        "output_path",
        "status",
        "reason",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    attempted = len(all_rows)
    succeeded = sum(1 for r in all_rows if r["status"] == "SUCCESS")
    failed = attempted - succeeded

    print(f"Extraction log : {log_path}")
    print(f"Attempted      : {attempted}")
    print(f"Succeeded      : {succeeded}")
    print(f"Failed         : {failed}")

    if failed:
        print("\nFailures:")
        for r in all_rows:
            if r["status"] == "FAIL":
                print(f"  {r['participant_id']} {r['task_code']}: {r['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract 24 task crops.")
    parser.add_argument(
        "participant_ids",
        nargs="*",
        help="Optional participant ids to process (default: all).",
    )
    args = parser.parse_args()

    metadata_dir = Path(config.data.metadata_dir)
    manifest_path = metadata_dir / "page_manifest.csv"
    log_path = metadata_dir / "extraction_log.csv"
    extracted_dir = Path(config.data.extracted_dir)

    extract_content(
        manifest_path,
        extracted_dir,
        log_path,
        args.participant_ids or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
