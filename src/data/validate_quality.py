"""
Phase 6 — Validate quality: run per-file checks on included samples.

Reads DATA/METADATA/samples_manifest.csv.  For every row where
excluded is False, four checks are applied in order; the first
failure sets excluded = True and records the reason:

  missing_file    — file does not exist or has zero bytes
  corrupt_image   — PIL cannot open or fully load the file
  low_resolution  — width or height < config.preprocessing.min_crop_size
  blank_content   — grayscale mean pixel value > config.preprocessing.blank_threshold
                    (participant left the section empty)

Rows already excluded (NEUTRAL / UNKNOWN) are passed through unchanged.
Exclusion is per file; a failure in one crop does not exclude the other
crops of the same participant.

Writes back to DATA/METADATA/samples_manifest.csv (adds exclusion_reason
column), and writes a summary to DATA/METADATA/quality_report.csv.

Run from the project root:

    python -m src.data.validate_quality
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils.config import config

MANIFEST_FIELDS = [
    "participant_id",
    "task_code",
    "image_path",
    "label",
    "excluded",
    "exclusion_reason",
]

REPORT_FIELDS = ["metric", "value"]

FAILURE_REASONS = (
    "missing_file",
    "corrupt_image",
    "low_resolution",
    "blank_content",
)


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def check_image(image_path: str) -> tuple[bool, str]:
    """Run all four quality checks. Return (passed, reason)."""
    cfg = config.preprocessing
    path = Path(image_path)

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


def validate_manifest(rows: list[dict]) -> list[dict]:
    """
    Run quality checks on every included row.  Returns updated rows with
    the exclusion_reason field populated.
    """
    updated: list[dict] = []
    for row in rows:
        r = dict(row)
        r.setdefault("exclusion_reason", "")

        if _parse_bool(r.get("excluded", "False")):
            updated.append(r)
            continue

        passed, reason = check_image(r["image_path"])
        if not passed:
            r["excluded"] = True
            r["exclusion_reason"] = reason

        updated.append(r)

    return updated


def build_report(
    original_rows: list[dict],
    updated_rows: list[dict],
) -> list[dict]:
    """Return key/value rows for quality_report.csv."""
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
    class_counts: Counter = Counter(r["label"] for r in final_usable)

    report = [
        {"metric": "total_checked", "value": total_checked},
        {"metric": "passed", "value": passed},
    ]
    for reason in FAILURE_REASONS:
        report.append({"metric": f"failed_{reason}", "value": failure_counts[reason]})
    for label in ("HAPPY", "SAD"):
        report.append({"metric": f"usable_{label}", "value": class_counts[label]})
    report.append(
        {"metric": "total_usable", "value": class_counts["HAPPY"] + class_counts["SAD"]}
    )
    return report


def read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report_path: Path, report: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report)


def print_summary(report: list[dict]) -> None:
    kv = {r["metric"]: r["value"] for r in report}
    print("\nQuality gate results:")
    print(f"  Checked          : {kv['total_checked']}")
    print(f"  Passed           : {kv['passed']}")
    for reason in FAILURE_REASONS:
        print(f"  Failed ({reason:<20}): {kv[f'failed_{reason}']}")
    print("\nFinal class distribution for Chapter 4:")
    print(f"  HAPPY (usable)   : {kv['usable_HAPPY']}")
    print(f"  SAD   (usable)   : {kv['usable_SAD']}")
    print(f"  Total for training: {kv['total_usable']}")


def validate_quality(
    manifest_path: Path,
    report_path: Path,
) -> None:
    print("=" * 60)
    print("Phase 6 - VALIDATE quality")
    print("=" * 60)

    if not manifest_path.is_file():
        print(f"ERROR: samples manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Manifest : {manifest_path}")

    original_rows = read_manifest(manifest_path)
    updated_rows = validate_manifest(original_rows)
    report = build_report(original_rows, updated_rows)

    write_manifest(manifest_path, updated_rows)
    print(f"Manifest updated : {manifest_path}")

    write_report(report_path, report)
    print(f"Quality report   : {report_path}")

    print_summary(report)


def main() -> int:
    metadata_dir = Path(config.data.metadata_dir)
    validate_quality(
        manifest_path=metadata_dir / "samples_manifest.csv",
        report_path=metadata_dir / "quality_report.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
