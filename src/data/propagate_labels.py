"""
Phase 5 — Propagate labels: join extraction report with participant labels.

Reads:
    DATA/METADATA/extraction_report.csv  (from Phase 4)
    DATA/METADATA/labels.csv             (from Phase 3)

For every SUCCESS row in the extraction report, looks up the participant's
label and p_happy from labels.csv. There is NO NEUTRAL class (Section A):
every labelled participant is included. Participants absent from labels.csv
are assigned UNKNOWN and excluded with a warning.

Writes DATA/METADATA/crop_index.csv with columns:
    crop_path, participant_id, task_code, task_type, label, p_happy,
    excluded, exclusion_reason

task_type is derived from the task_code prefix:
    draw_*    -> drawing
    word_*    -> word
    cursive_* -> cursive

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
UNKNOWN_LABEL = "UNKNOWN"

CROP_INDEX_FIELDS = [
    "crop_path", "participant_id", "task_code", "task_type",
    "label", "p_happy", "excluded", "exclusion_reason",
]


def _task_type(task_code: str) -> str:
    if task_code.startswith("draw_"):
        return "drawing"
    if task_code.startswith("word_"):
        return "word"
    if task_code.startswith("cursive_"):
        return "cursive"
    return "unknown"


def load_labels(labels_path: Path) -> dict[str, dict]:
    """Return participant_id -> {label, p_happy, ...} from labels.csv."""
    label_map: dict[str, dict] = {}
    with labels_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("participant_id") or "").strip()
            if pid:
                label_map[pid] = {
                    "label":   (row.get("label") or "").strip().upper(),
                    "p_happy": row.get("p_happy", ""),
                }
    return label_map


def build_crop_index(
    report_path: Path,
    label_map: dict[str, dict],
) -> tuple[list[dict], set[str]]:
    """Return (crop_index_rows, unknown_pids) for all SUCCESS extractions."""
    rows: list[dict] = []
    unknown_pids: set[str] = set()

    with report_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("status") or "").strip().upper() != "SUCCESS":
                continue

            pid = (row.get("participant_id") or "").strip()
            task_code = (row.get("task_code") or "").strip()
            crop_path = (row.get("output_path") or "").strip()

            if pid in label_map:
                label = label_map[pid]["label"]
                p_happy = label_map[pid]["p_happy"]
                excluded = False
                exclusion_reason = ""
            else:
                label = UNKNOWN_LABEL
                p_happy = ""
                excluded = True
                exclusion_reason = "participant not in labels.csv"
                unknown_pids.add(pid)

            rows.append({
                "crop_path":        crop_path,
                "participant_id":   pid,
                "task_code":        task_code,
                "task_type":        _task_type(task_code),
                "label":            label,
                "p_happy":          p_happy,
                "excluded":         excluded,
                "exclusion_reason": exclusion_reason,
            })

    return rows, unknown_pids


def write_crop_index(index_path: Path, rows: list[dict]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CROP_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    from collections import Counter
    included = [r for r in rows if not r["excluded"]]
    label_counts = Counter(r["label"] for r in included)
    unknown = sum(1 for r in rows if r["label"] == UNKNOWN_LABEL)

    print(f"  Total crops processed  : {len(rows)}")
    print(f"  HAPPY (included)       : {label_counts['HAPPY']}")
    print(f"  SAD   (included)       : {label_counts['SAD']}")
    print(f"  UNKNOWN (excluded)     : {unknown}")
    print(f"  Total for training     : {label_counts['HAPPY'] + label_counts['SAD']}")
    print("  (No NEUTRAL class - Section A)")


def propagate_labels(
    report_path: Path,
    labels_path: Path,
    index_path: Path,
) -> None:
    print("=" * 60)
    print("Phase 5 - PROPAGATE labels -> crop_index.csv")
    print("=" * 60)

    if not report_path.is_file():
        print(f"ERROR: extraction report not found: {report_path}")
        sys.exit(1)
    if not labels_path.is_file():
        print(f"ERROR: labels file not found: {labels_path}")
        sys.exit(1)

    label_map = load_labels(labels_path)
    print(f"Labels loaded  : {len(label_map)} participants")

    rows, unknown_pids = build_crop_index(report_path, label_map)

    for pid in sorted(unknown_pids):
        warnings.warn(
            f"Participant '{pid}' not in labels.csv; assigned UNKNOWN.",
            UserWarning,
            stacklevel=2,
        )

    write_crop_index(index_path, rows)
    print(f"Crop index written: {index_path}")
    print_summary(rows)


def main() -> int:
    meta = Path(config.data.metadata_dir)
    propagate_labels(
        report_path=meta / "extraction_report.csv",
        labels_path=meta / "labels.csv",
        index_path=meta / "crop_index.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
