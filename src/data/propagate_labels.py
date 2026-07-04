"""
Phase 5 — Propagate labels: join extraction log with participant labels.

Reads DATA/METADATA/extraction_log.csv and DATA/METADATA/labels.csv.
For every extraction row whose status is SUCCESS:

  HAPPY  -> included (excluded = False)
  SAD    -> included (excluded = False)
  NEUTRAL -> excluded (excluded = True)
  not found in labels.csv -> excluded = True, label = UNKNOWN, warning printed

Writes DATA/METADATA/samples_manifest.csv with columns:
    participant_id, task_code, image_path, label, excluded

Run from the project root:

    python -m src.data.propagate_labels
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

from src.utils.config import config

INCLUDED_LABELS = {"HAPPY", "SAD"}
EXCLUDED_LABEL = "NEUTRAL"
UNKNOWN_LABEL = "UNKNOWN"


def load_labels(labels_path: Path) -> dict[str, str]:
    """Return participant_id -> label from labels.csv."""
    label_map: dict[str, str] = {}
    with labels_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("participant_id") or "").strip()
            label = (row.get("label") or "").strip().upper()
            if pid:
                label_map[pid] = label
    return label_map


def build_manifest(
    log_path: Path,
    label_map: dict[str, str],
) -> tuple[list[dict], set[str]]:
    """Return (manifest_rows, unknown_pids) for all SUCCESS extractions."""
    rows: list[dict] = []
    unknown_pids: set[str] = set()

    with log_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("status") or "").strip().upper() != "SUCCESS":
                continue

            pid = (row.get("participant_id") or "").strip()
            task_code = (row.get("task_code") or "").strip()
            image_path = (row.get("output_path") or "").strip()

            if pid in label_map:
                label = label_map[pid]
                excluded = label not in INCLUDED_LABELS
            else:
                label = UNKNOWN_LABEL
                excluded = True
                unknown_pids.add(pid)

            rows.append(
                {
                    "participant_id": pid,
                    "task_code": task_code,
                    "image_path": image_path,
                    "label": label,
                    "excluded": excluded,
                }
            )

    return rows, unknown_pids


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["participant_id", "task_code", "image_path", "label", "excluded"]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    total = len(rows)
    happy = sum(1 for r in rows if r["label"] == "HAPPY" and not r["excluded"])
    sad = sum(1 for r in rows if r["label"] == "SAD" and not r["excluded"])
    neutral = sum(1 for r in rows if r["label"] == EXCLUDED_LABEL and r["excluded"])
    unknown = sum(1 for r in rows if r["label"] == UNKNOWN_LABEL and r["excluded"])
    for_training = happy + sad

    print(f"  Total files processed : {total}")
    print(f"  HAPPY    (included)   : {happy}")
    print(f"  SAD      (included)   : {sad}")
    print(f"  NEUTRAL  (excluded)   : {neutral}")
    print(f"  UNKNOWN  (excluded)   : {unknown}")
    print(f"  Total for training    : {for_training}")


def propagate_labels(
    log_path: Path,
    labels_path: Path,
    manifest_path: Path,
) -> None:
    print("=" * 60)
    print("Phase 5 - PROPAGATE labels")
    print("=" * 60)

    if not log_path.is_file():
        print(f"ERROR: extraction log not found: {log_path}")
        sys.exit(1)

    if not labels_path.is_file():
        print(f"ERROR: labels file not found: {labels_path}")
        sys.exit(1)

    label_map = load_labels(labels_path)
    print(f"Labels loaded  : {len(label_map)} participants")

    rows, unknown_pids = build_manifest(log_path, label_map)

    for pid in sorted(unknown_pids):
        warnings.warn(
            f"Participant '{pid}' not found in labels.csv; "
            "assigned UNKNOWN and excluded.",
            UserWarning,
            stacklevel=2,
        )

    write_manifest(manifest_path, rows)
    print(f"Manifest written: {manifest_path}")
    print_summary(rows)


def main() -> int:
    metadata_dir = Path(config.data.metadata_dir)
    propagate_labels(
        log_path=metadata_dir / "extraction_log.csv",
        labels_path=metadata_dir / "labels.csv",
        manifest_path=metadata_dir / "samples_manifest.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
