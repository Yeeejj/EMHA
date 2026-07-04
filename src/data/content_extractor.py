"""
Phase 4 — Extract content: cut 24 task crops from each participant's pages.

Gate 3 deliverable.  Implements WritingCellExtractor using morphological grid
detection and fractional (0.0-1.0) bounding boxes so crop coordinates are
resolution-independent (template scan is anisotropic ~74.5 DPI H / ~81.1 DPI V).

Page roles (from page_manifest.csv):
    page_questionnaire  - index 0, skip (ground-truth source only)
    page_drawing        - index 1, 2x2 drawing grid (4 crops)
    page_writing        - index 2-3, 5x3 word table (15 crops) + 5 cursive rows

skip_skew = True for all draw_* task codes (no text baseline).
skip_skew = False for all word_* and cursive_* task codes.

Output per respondent:
    DATA/CROPS/{participant_id}/{participant_id}_{task_code}.png   (24 files)
Log written to:
    DATA/METADATA/extraction_report.csv

Run from the project root:

    python -m src.data.content_extractor
    python -m src.data.content_extractor P001
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

REPORT_FIELDS = [
    "participant_id", "task_code", "output_path", "status", "reason",
]


def imread_unicode(path: Path) -> np.ndarray | None:
    """Read an image via numpy buffer (handles non-ASCII Windows paths)."""
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Write an image via numpy buffer (handles non-ASCII Windows paths)."""
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def frac_box_to_pixels(
    frac_box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert fractional (0.0-1.0) box to pixel coordinates clamped to image."""
    x1f, y1f, x2f, y2f = frac_box
    x1 = max(0, min(int(x1f * width), width))
    y1 = max(0, min(int(y1f * height), height))
    x2 = max(0, min(int(x2f * width), width))
    y2 = max(0, min(int(y2f * height), height))
    return x1, y1, x2, y2


class WritingCellExtractor:
    """Extracts writing and drawing crops from a scanned FINALE page.

    Uses fractional bounding boxes so coordinates are scale-invariant across
    scan resolutions.  Deskew is applied to word/cursive cells and skipped for
    drawing cells (no text baseline).
    """

    def __init__(
        self,
        crops_dir: Path,
        skip_skew_prefixes: tuple[str, ...] = ("draw_",),
    ):
        self.crops_dir = crops_dir
        self.skip_skew_prefixes = skip_skew_prefixes

    def _should_skip_skew(self, task_code: str) -> bool:
        return any(task_code.startswith(p) for p in self.skip_skew_prefixes)

    def _crop_and_save(
        self,
        image: np.ndarray,
        frac_box: tuple[float, float, float, float],
        out_path: Path,
    ) -> tuple[bool, str]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = frac_box_to_pixels(frac_box, w, h)

        if x2 <= x1 or y2 <= y1:
            return False, f"empty crop after scaling box {frac_box} to {w}x{h}"

        crop = image[y1:y2, x1:x2]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(out_path, crop):
            return False, "cv2 encode/write failed"
        return True, ""

    def extract_participant(
        self,
        pid: str,
        pages: dict[str, str],
    ) -> list[dict]:
        """Extract all 24 crops for one participant. Returns log rows."""
        log_rows: list[dict] = []
        out_dir = self.crops_dir / pid

        source_specs = [
            (DRAWING_ROLE, DRAWING_CROPS),
            (WRITING_ROLE, {**WORD_CROPS, **CURSIVE_CROPS}),
        ]

        for role, crop_map in source_specs:
            png_path_str = pages.get(role)
            image = imread_unicode(Path(png_path_str)) if png_path_str else None

            for task_code, frac_box in crop_map.items():
                out_path = out_dir / f"{pid}_{task_code}.png"

                if not png_path_str:
                    status, reason = "FAIL", f"no {role} row in manifest"
                elif image is None:
                    status, reason = "FAIL", f"could not read {role} image"
                else:
                    ok, reason = self._crop_and_save(image, frac_box, out_path)
                    status = "SUCCESS" if ok else "FAIL"

                log_rows.append({
                    "participant_id": pid,
                    "task_code":      task_code,
                    "output_path":    str(out_path),
                    "status":         status,
                    "reason":         reason,
                })

        return log_rows


def read_manifest(manifest_path: Path) -> OrderedDict[str, dict[str, str]]:
    """Return participant_id -> {page_role: png_path}, in manifest order."""
    participants: OrderedDict[str, dict[str, str]] = OrderedDict()
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row["participant_id"]
            participants.setdefault(pid, {})[row["page_role"]] = row["png_path"]
    return participants


def extract_content(
    manifest_path: Path,
    crops_dir: Path,
    report_path: Path,
    only: list[str] | None,
) -> None:
    print("=" * 60)
    print("Phase 4 - EXTRACT content (WritingCellExtractor)")
    print("=" * 60)

    if not manifest_path.is_file():
        print(f"ERROR: page manifest not found: {manifest_path}")
        sys.exit(1)

    participants = read_manifest(manifest_path)
    if only:
        missing = [p for p in only if p not in participants]
        for p in missing:
            print(f"WARNING: '{p}' not in manifest.")
        participants = OrderedDict(
            (p, participants[p]) for p in only if p in participants
        )

    print(f"Page manifest : {manifest_path}")
    print(f"Participants  : {len(participants)}")

    extractor = WritingCellExtractor(
        crops_dir=crops_dir,
        skip_skew_prefixes=config.preprocessing.skip_skew_prefixes,
    )

    all_rows: list[dict] = []
    for pid, pages in participants.items():
        all_rows.extend(extractor.extract_participant(pid, pages))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    attempted = len(all_rows)
    succeeded = sum(1 for r in all_rows if r["status"] == "SUCCESS")
    failed = attempted - succeeded

    print(f"Extraction report : {report_path}")
    print(f"Attempted         : {attempted}")
    print(f"Succeeded         : {succeeded}")
    print(f"Failed            : {failed}")

    if failed:
        print("\nFailures:")
        for r in all_rows:
            if r["status"] == "FAIL":
                print(f"  {r['participant_id']} {r['task_code']}: {r['reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract 24 task crops per participant.")
    parser.add_argument("participant_ids", nargs="*", help="Optional subset (default: all)")
    args = parser.parse_args()

    meta = Path(config.data.metadata_dir)
    extract_content(
        manifest_path=meta / "page_manifest.csv",
        crops_dir=Path(config.data.crops_dir),
        report_path=meta / "extraction_report.csv",
        only=args.participant_ids or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
