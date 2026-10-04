"""Auto-crop the P4 Writing Exercise page into its 20 cells.

Reads every PNG sitting directly in the source folder, cuts the 20 fixed
regions, and saves each one into its own subfolder (W1_LH ... CS5) as
EMHA-P4_WritingExercise_<number code>_<label>.png.

The source scans are only read, never modified or moved.

Usage (from the project root):
    python -m src.cropping.p4.crop_p4 --dry-run
    python -m src.cropping.p4.crop_p4
    python -m src.cropping.p4.crop_p4 --range 001-100
    python -m src.cropping.p4.crop_p4 --code 057
    python -m src.cropping.p4.crop_p4 --code 005 017 123
    python -m src.cropping.p4.crop_p4 --overwrite
    python -m src.cropping.p4.crop_p4 --src "D:\\other\\folder"

Requires: pip install pillow
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# (left, top, width, height) in pixels
Box = Tuple[int, int, int, int]


def _default_boxes() -> Dict[str, Box]:
    word_w, word_h = 368, 118
    strip_w, strip_h = 1342, 174
    return {
        "W1_LH": (395, 265, word_w, word_h),
        "W1_RH": (769, 265, word_w, word_h),
        "W1_UC": (1149, 265, word_w, word_h),
        "W2_LH": (395, 383, word_w, word_h),
        "W2_RH": (772, 383, word_w, word_h),
        "W2_UC": (1151, 383, word_w, word_h),
        "W3_LH": (395, 500, word_w, word_h),
        "W3_RH": (772, 500, word_w, word_h),
        "W3_UC": (1151, 500, word_w, word_h),
        "W4_LH": (395, 620, word_w, word_h),
        "W4_RH": (772, 620, word_w, word_h),
        "W4_UC": (1151, 620, word_w, word_h),
        "W5_LH": (395, 741, word_w, word_h),
        "W5_RH": (772, 741, word_w, word_h),
        "W5_UC": (1151, 741, word_w, word_h),
        "CS1": (182, 956, strip_w, strip_h),
        "CS2": (182, 1180, strip_w, strip_h),
        "CS3": (182, 1398, strip_w, strip_h),
        "CS4": (182, 1613, strip_w, strip_h),
        "CS5": (182, 1838, strip_w, strip_h),
    }


@dataclass
class P4CropConfig:
    src_dir: Path = Path(r"D:\EMHA_Thesis\EMHA-1\DATASET\raw-4Page")
    prefix: str = "EMHA-P4_WritingExercise"
    # Set to (width, height) to pin the expected scan size. If None, the
    # first scan read becomes the reference and any scan of a different
    # size is skipped and reported.
    page_size: Optional[Tuple[int, int]] = None
    boxes: Dict[str, Box] = field(default_factory=_default_boxes)


def number_code(stem: str, prefix: str) -> str:
    """Pull the respondent number code out of a scan's file name.

    'EMHA-P4_WritingExercise_0123' -> '0123'
    '0123'                         -> '0123'
    """
    match = re.fullmatch(rf"{re.escape(prefix)}[_-]?(.+)", stem)
    return match.group(1) if match else stem


def _as_int(code: str) -> Optional[int]:
    return int(code) if code.isdigit() else None


def parse_range(text: str) -> Tuple[int, int]:
    """'001-100' -> (1, 100)."""
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", text)
    if not match:
        raise argparse.ArgumentTypeError("use START-END, e.g. 001-100")
    start, end = int(match.group(1)), int(match.group(2))
    if start > end:
        raise argparse.ArgumentTypeError("range start is larger than its end")
    return start, end


def same_code(a: str, b: str) -> bool:
    """'57' and '057' count as the same respondent."""
    if a == b:
        return True
    return _as_int(a) is not None and _as_int(a) == _as_int(b)


def crop_page(
    path: Path, cfg: P4CropConfig, overwrite: bool, dry_run: bool
) -> Tuple[int, int]:
    """Crop one scan. Returns (saved, skipped_existing)."""
    code = number_code(path.stem, cfg.prefix)
    saved = skipped = 0
    with Image.open(path) as page:
        for label, (left, top, width, height) in cfg.boxes.items():
            out_dir = cfg.src_dir / label
            out_path = out_dir / f"{cfg.prefix}_{code}_{label}.png"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            if not dry_run:
                out_dir.mkdir(exist_ok=True)
                cell = page.crop((left, top, left + width, top + height))
                cell.save(out_path)
            saved += 1
    return saved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, help="folder holding the P4 scans")
    parser.add_argument(
        "--range",
        type=parse_range,
        metavar="START-END",
        help="only crop number codes in this range, e.g. 001-100",
    )
    parser.add_argument(
        "--code",
        nargs="+",
        metavar="CODE",
        help="only crop these number codes, e.g. 057 or 005 017 123",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="redo crops that already exist"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, write nothing"
    )
    args = parser.parse_args()

    cfg = P4CropConfig()
    if args.src:
        cfg.src_dir = args.src

    # Top level only, so the W1_LH ... CS5 output folders are never re-read.
    scans = sorted(
        p for p in cfg.src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
    )
    if not scans:
        raise SystemExit(f"No PNG files found in {cfg.src_dir}")

    found = len(scans)
    if args.range or args.code:
        wanted = args.code or []
        kept = []
        for path in scans:
            code = number_code(path.stem, cfg.prefix)
            num = _as_int(code)
            in_range = (
                args.range is not None
                and num is not None
                and args.range[0] <= num <= args.range[1]
            )
            if in_range or any(same_code(code, w) for w in wanted):
                kept.append(path)
        missing = [
            w
            for w in wanted
            if not any(same_code(number_code(p.stem, cfg.prefix), w) for p in kept)
        ]
        scans = kept
        print(f"Selected {len(scans)} of {found} scans")
        if missing:
            print(f"No scan found for code(s): {', '.join(missing)}")
        if not scans:
            raise SystemExit("Nothing to crop for that selection.")

    need_w = max(left + w for left, _, w, _ in cfg.boxes.values())
    need_h = max(top + h for _, top, _, h in cfg.boxes.values())

    expected = cfg.page_size
    total_saved = total_skipped = 0
    problems: List[str] = []

    for path in scans:
        with Image.open(path) as page:
            size = page.size
        if expected is None:
            expected = size
            print(f"Reference page size: {size[0]}x{size[1]} (from {path.name})")
        if size[0] < need_w or size[1] < need_h:
            problems.append(
                f"{path.name}: {size[0]}x{size[1]} is smaller than the crop "
                f"area ({need_w}x{need_h})"
            )
            continue
        if size != expected:
            problems.append(
                f"{path.name}: {size[0]}x{size[1]}, expected "
                f"{expected[0]}x{expected[1]}"
            )
            continue
        saved, skipped = crop_page(path, cfg, args.overwrite, args.dry_run)
        total_saved += saved
        total_skipped += skipped

    verb = "Would save" if args.dry_run else "Saved"
    print(f"\nScans processed: {len(scans)}")
    print(f"{verb}: {total_saved} crops")
    print(f"Already existed (left alone): {total_skipped}")
    if problems:
        print(f"\nSkipped {len(problems)} scan(s) - check these by hand:")
        for line in problems:
            print(f"  {line}")


if __name__ == "__main__":
    main()