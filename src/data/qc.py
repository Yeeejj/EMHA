"""
Phase 6 — Quality control: per-crop checks on included samples.

Reads DATA/METADATA/crop_index.csv.  For every row where excluded is False,
four checks are applied in order; the first failure sets excluded=True:

    missing_file    - file does not exist or has zero bytes
    corrupt_image   - PIL cannot open or fully load the file
    low_resolution  - width or height < config.preprocessing.min_crop_size
    blank_content   - grayscale mean pixel > config.preprocessing.blank_threshold

Rows already excluded (UNKNOWN) pass through unchanged.
Exclusion is per-crop; a failure in one crop does not exclude others for that
participant.

Outputs:
    DATA/METADATA/crop_index.csv  (updated with exclusion flags)
    DATA/METADATA/qc_report.csv   (summary statistics)
    DATA/METADATA/exclusions.csv  (excluded crop rows with reasons)

Run from the project root:

    python -m src.data.qc
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils.config import config

CROP_INDEX_FIELDS = [
    "crop_path", "participant_id", "task_code", "task_type",
    "label", "p_happy", "excluded", "exclusion_reason",
]
REPORT_FIELDS = ["metric", "value"]
EXCLUSIONS_FIELDS = CROP_INDEX_FIELDS

FAILURE_REASONS = (
    "missing_file",
    "corrupt_image",
    "low_resolution",
    "blank_content",
)


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def check_image(crop_path: str) -> tuple[bool, str]:
    """Run all four quality checks. Return (passed, reason)."""
    cfg = config.preprocessing
    path = Path(crop_path)

    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing_file"

    try:
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError):
        return False, "corrupt_image"

    min_w, min_h = cfg.min_crop_size
    if img.width < min_w or img.height < min_h:
        return False, "low_resolution"

    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.mean() > cfg.blank_threshold:
        return False, "blank_content"

    return True, ""


def run_qc(rows: list[dict]) -> list[dict]:
    """Apply quality checks to every included row; return updated rows."""
    updated: list[dict] = []
    for row in rows:
        r = dict(row)
        r.setdefault("exclusion_reason", "")

        if _parse_bool(r.get("excluded", "False")):
            updated.append(r)
            continue

        passed, reason = check_image(r["crop_path"])
        if not passed:
            r["excluded"] = True
            r["exclusion_reason"] = reason

        updated.append(r)
    return updated


def build_report(
    original_rows: list[dict],
    updated_rows: list[dict],
) -> list[dict]:
    originally_included = [
        r for r in original_rows if not _parse_bool(r.get("excluded", "False"))
    ]
    total_checked = len(originally_included)

    failure_counts: Counter = Counter()
    passed = 0
    for orig, upd in zip(original_rows, updated_rows):
        if _parse_bool(orig.get("excluded", "False")):
            continue
        reason = upd.get("exclusion_reason", "")
        if reason:
            failure_counts[reason] += 1
        else:
            passed += 1

    final_usable = [
        r for r in updated_rows if not _parse_bool(str(r.get("excluded", "False")))
    ]
    label_counts: Counter = Counter(r["label"] for r in final_usable)

    report = [
        {"metric": "total_checked", "value": total_checked},
        {"metric": "passed", "value": passed},
    ]
    for reason in FAILURE_REASONS:
        report.append({"metric": f"failed_{reason}", "value": failure_counts[reason]})
    for label in ("HAPPY", "SAD"):
        report.append({"metric": f"usable_{label}", "value": label_counts[label]})
    report.append(
        {"metric": "total_usable", "value": label_counts["HAPPY"] + label_counts["SAD"]}
    )
    return report


def read_crop_index(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_crop_index(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CROP_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, report: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report)


def write_exclusions(path: Path, rows: list[dict]) -> None:
    excluded = [r for r in rows if _parse_bool(str(r.get("excluded", "False")))]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXCLUSIONS_FIELDS)
        writer.writeheader()
        writer.writerows(excluded)


def print_summary(report: list[dict]) -> None:
    kv = {r["metric"]: r["value"] for r in report}
    print("\nQuality gate results:")
    print(f"  Checked          : {kv['total_checked']}")
    print(f"  Passed           : {kv['passed']}")
    for reason in FAILURE_REASONS:
        print(f"  Failed ({reason:<20}): {kv[f'failed_{reason}']}")
    print("\nFinal usable pool (Chapter 4):")
    print(f"  HAPPY : {kv['usable_HAPPY']}")
    print(f"  SAD   : {kv['usable_SAD']}")
    print(f"  Total : {kv['total_usable']}")


def validate_quality(
    index_path: Path,
    report_path: Path,
    exclusions_path: Path,
) -> None:
    print("=" * 60)
    print("Phase 6 - QUALITY CONTROL")
    print("=" * 60)

    if not index_path.is_file():
        print(f"ERROR: crop index not found: {index_path}")
        sys.exit(1)

    print(f"Crop index : {index_path}")

    original_rows = read_crop_index(index_path)
    updated_rows = run_qc(original_rows)
    report = build_report(original_rows, updated_rows)

    write_crop_index(index_path, updated_rows)
    print(f"Crop index updated : {index_path}")

    write_report(report_path, report)
    print(f"QC report          : {report_path}")

    write_exclusions(exclusions_path, updated_rows)
    print(f"Exclusions         : {exclusions_path}")

    print_summary(report)


def main() -> int:
    meta = Path(config.data.metadata_dir)
    validate_quality(
        index_path=meta / "crop_index.csv",
        report_path=meta / "qc_report.csv",
        exclusions_path=meta / "exclusions.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
