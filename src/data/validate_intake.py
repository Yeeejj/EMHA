"""
Phase 0 — Intake: copy raw data to staging and validate respondent folders.

STEP 1 (COPY): Mirror the READ-ONLY raw dataset into DATA/STAGING/ with
shutil.copytree(dirs_exist_ok=True) so re-running is idempotent. The raw
dataset (config.data.raw_data_dir) is never written to (CLAUDE.md Rule 1).

STEP 2 (VALIDATE): For every respondent folder under DATA/STAGING/, confirm a
PNG and a PDF subfolder exist, that exactly four PNG files are present, that
every PNG filename matches EMHA<8 digits>_<8 digits>.png, and that no file is
zero bytes. Results are written to DATA/METADATA/validation_report.csv.

Run from the project root:

    python -m src.data.validate_intake
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

from src.utils.config import config

# EMHA{YYYYMMDD}_{HHMMSSff}.png  ->  "EMHA" + 8 digits + "_" + 8 digits + ".png"
PNG_NAME_PATTERN = re.compile(r"^EMHA\d{8}_\d{8}\.png$")
EXPECTED_PNG_COUNT = 4


def _count_subfolders(directory: Path) -> int:
    """Return the number of immediate subdirectories of *directory*."""
    return sum(1 for child in directory.iterdir() if child.is_dir())


def copy_raw_to_staging(raw_dir: Path, staging_dir: Path) -> bool:
    """Copy the raw dataset into staging, preserving the folder structure.

    Returns True if the staging folder count matches the raw folder count,
    False otherwise (caller should stop on False).
    """
    print("=" * 60)
    print("STEP 1 - COPY raw dataset to staging")
    print("=" * 60)

    if not raw_dir.is_dir():
        print(f"ERROR: raw data dir does not exist: {raw_dir}")
        return False

    raw_count = _count_subfolders(raw_dir)
    print(f"Raw data dir : {raw_dir}")
    print(f"Staging dir  : {staging_dir}")
    print(f"Found {raw_count} respondent folders in raw. Copying...")

    staging_dir.mkdir(parents=True, exist_ok=True)
    # dirs_exist_ok=True makes re-runs safe (overwrites existing files).
    shutil.copytree(raw_dir, staging_dir, dirs_exist_ok=True)

    staging_count = _count_subfolders(staging_dir)
    print(f"Copied. Staging now holds {staging_count} respondent folders.")

    if staging_count != raw_count:
        print(
            "WARNING: staging folder count "
            f"({staging_count}) does not match raw folder count "
            f"({raw_count}). Stopping."
        )
        return False

    print(f"Verified: staging matches raw ({raw_count} folders).")
    return True


def validate_folder(folder: Path) -> dict:
    """Validate a single respondent folder.

    Returns a row dict with keys: folder_name, status, reason, file_count,
    notes.
    """
    png_dir = folder / "PNG"
    pdf_dir = folder / "PDF"

    reasons: list[str] = []
    notes: list[str] = []

    if not pdf_dir.is_dir():
        reasons.append("missing PDF subfolder")

    png_files: list[Path] = []
    if not png_dir.is_dir():
        reasons.append("missing PNG subfolder")
    else:
        png_files = sorted(p for p in png_dir.iterdir() if p.is_file())

        if len(png_files) != EXPECTED_PNG_COUNT:
            reasons.append(
                f"expected {EXPECTED_PNG_COUNT} PNG files, " f"found {len(png_files)}"
            )

        bad_names = [p.name for p in png_files if not PNG_NAME_PATTERN.match(p.name)]
        if bad_names:
            reasons.append("filename pattern mismatch: " + ", ".join(bad_names))

        zero_byte = [p.name for p in png_files if p.stat().st_size == 0]
        if zero_byte:
            reasons.append("zero-byte file(s): " + ", ".join(zero_byte))

    file_count = len(png_files)
    status = "PASS" if not reasons else "FAIL"

    return {
        "folder_name": folder.name,
        "status": status,
        "reason": "; ".join(reasons),
        "file_count": file_count,
        "notes": "; ".join(notes),
    }


def validate_staging(staging_dir: Path, report_path: Path) -> None:
    """Validate every respondent folder in staging and write the CSV report."""
    print()
    print("=" * 60)
    print("STEP 2 - VALIDATE staging folders")
    print("=" * 60)

    folders = sorted(
        (child for child in staging_dir.iterdir() if child.is_dir()),
        key=lambda p: p.name,
    )

    rows = [validate_folder(folder) for folder in folders]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["folder_name", "status", "reason", "file_count", "notes"]
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = len(rows) - passed

    print(f"Validation report written to: {report_path}")
    print(f"Total scanned : {len(rows)}")
    print(f"Passed        : {passed}")
    print(f"Failed        : {failed}")

    if failed:
        print("\nFailed folders:")
        for r in rows:
            if r["status"] == "FAIL":
                print(f"  {r['folder_name']}: {r['reason']}")


def main() -> int:
    raw_dir = Path(config.data.raw_data_dir)
    staging_dir = Path(config.data.staging_dir)
    report_path = Path(config.data.metadata_dir) / "validation_report.csv"

    if not copy_raw_to_staging(raw_dir, staging_dir):
        return 1

    validate_staging(staging_dir, report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
