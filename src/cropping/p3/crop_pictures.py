"""
Crop the 4 drawings out of each scanned page and save them into D1-D4 folders.

Output:
    D:\\EMHA_Thesis\\EMHA-1\\DATASET\\raw-3Page\\D1\\EMHA-P3_DrawingExercise_<CODE>_D1.png
    ... same for D2, D3, D4

Scans are read from:
    D:\\EMHA_Thesis\\EMHA-1\\DATASET\\raw-3Page

Usage (from the project root, D:\\EMHA_Thesis\\EMHA-1):
    python -m src.cropping.p3.crop_pictures                    every scan
    python -m src.cropping.p3.crop_pictures --range 001-100    codes 001 to 100
    python -m src.cropping.p3.crop_pictures --only 007         one code
    python -m src.cropping.p3.crop_pictures --only 007 023 150 several codes

Add --dry-run to any of these to preview without saving anything.
--range and --only can be combined. Leading zeros are optional (7 = 007).

Scans are read from the top level of the folder only, so the crops already
saved inside D1-D4 are never picked up as scans.

The number code is the last group of digits in the scan's file name
(EMHA-P3_DrawingExercise_001.png -> 001).

Requires:  pip install pillow      (plus: pip install pymupdf, for PDF scans)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

# Folder holding the scans. D1-D4 folders are created inside it.
SCAN_DIR = Path(r"D:\EMHA_Thesis\EMHA-1\DATASET\raw-3Page")
OUT_ROOT = SCAN_DIR
NAME_TEMPLATE = "EMHA-P3_DrawingExercise_{code}_{label}.png"

# label: (x, y, width, height) in pixels from the top-left corner of the page
BOXES = {
    "D1": (182, 254, 674, 669),
    "D2": (859, 254, 662, 674),
    "D3": (177, 996, 675, 1016),
    "D4": (859, 996, 664, 1016),
}

SCAN_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}


def load_page(path: Path, dpi: int) -> Image.Image:
    """Open an image directly, or render page 1 of a PDF at the given DPI."""
    if path.suffix.lower() != ".pdf":
        return Image.open(path)

    import fitz  # PyMuPDF

    with fitz.open(path) as doc:
        pix = doc[0].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def number_code(path: Path) -> str | None:
    """Last group of digits in the file name, leading zeros kept."""
    groups = re.findall(r"\d+", path.stem)
    return groups[-1] if groups else None


def wanted_codes(args: argparse.Namespace) -> set[int] | None:
    """Codes picked with --range / --only, or None when every scan is wanted."""
    if not args.range and not args.only:
        return None

    wanted: set[int] = set()
    if args.range:
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", args.range)
        if not match:
            raise ValueError(f"--range must look like 001-100, got: {args.range}")
        first, last = sorted(int(n) for n in match.groups())
        wanted.update(range(first, last + 1))
    for item in args.only or []:
        for part in item.split(","):
            if not part.strip().isdigit():
                raise ValueError(f"--only takes number codes, got: {part}")
            wanted.add(int(part))
    return wanted


def process(scan: Path, code: str, args: argparse.Namespace) -> tuple[int, int]:
    """Crop one scan. Returns (pictures saved, pictures already there)."""
    page = load_page(scan, args.dpi)
    print(f"{scan.name}  ->  code {code}  ({page.width} x {page.height} px)")

    saved, existing = 0, 0
    for label, (x, y, w, h) in BOXES.items():
        target = args.out_root / label / NAME_TEMPLATE.format(code=code, label=label)

        if x + w > page.width or y + h > page.height:
            print(f"    {label}: SKIPPED, box goes outside the page")
            continue
        if target.exists() and not args.overwrite:
            print(f"    {label}: already exists, left as is: {target.name}")
            existing += 1
            continue

        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            page.crop((x, y, x + w, y + h)).save(target)
        print(f"    {label}: {target}")
        saved += 1
    return saved, existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=SCAN_DIR,
        help="a scan, or a folder of scans (default: the raw-3Page folder)",
    )
    parser.add_argument("--range", metavar="FIRST-LAST", help="e.g. 001-100")
    parser.add_argument("--only", nargs="+", metavar="CODE", help="e.g. 007 023")
    parser.add_argument("--set-code", metavar="CODE", help="name a single scan by hand")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--dpi", type=int, default=200, help="PDF render resolution")
    parser.add_argument("--dry-run", action="store_true", help="save nothing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"not found: {args.source}")

    try:
        wanted = wanted_codes(args)
    except ValueError as error:
        parser.error(str(error))

    if args.source.is_dir():
        if args.set_code:
            parser.error("--set-code only works with a single scan, not a folder")
        scans = sorted(
            p for p in args.source.iterdir() if p.suffix.lower() in SCAN_TYPES
        )
    else:
        scans = [args.source]

    total, done, found, problems = 0, 0, set(), []
    for scan in scans:
        code = args.set_code or number_code(scan)
        if code is None:
            if wanted is None:
                problems.append(f"{scan.name}: no digits in the file name")
            continue
        if wanted is not None:
            if int(code) not in wanted:
                continue
            found.add(int(code))
        done += 1
        try:
            saved, existing = process(scan, code, args)
        except Exception as error:  # keep going, report at the end
            problems.append(f"{scan.name}: {error}")
            continue
        total += saved
        if saved + existing < len(BOXES):
            problems.append(
                f"{scan.name}: {len(BOXES) - saved - existing} box(es) skipped"
            )

    missing = sorted(wanted - found) if wanted is not None else []

    verb = "would be saved" if args.dry_run else "saved"
    print(f"\n{done} scan(s), {total} picture(s) {verb}.")
    if missing:
        codes = ", ".join(f"{n:03d}" for n in missing)
        print(f"{len(missing)} requested code(s) have no scan in the folder: {codes}")
    if problems:
        print(f"{len(problems)} scan(s) need a look:")
        for line in problems:
            print(f"    {line}")


if __name__ == "__main__":
    main()