"""
Phase 8 — Split dataset: participant-level stratified train/val/test split.

HARD CONSTRAINT: all files from one participant must land in the same split.
File-level splitting leaks the participant's handwriting style into both
training and evaluation sets.

Procedure:
  1. Read DATA/METADATA/samples_manifest.csv.
  2. Collect unique participants with label HAPPY or SAD and at least
     one included file (excluded == False).
  3. Stratified split preserving class balance (stratify on label,
     random_state from config.training.random_state):
       a. Split off test  (config.data.test_ratio,  default 15 %)
       b. Split remainder into train (70 %) and val (15 %)
  4. Copy each participant's included processed files from
     DATA/PROCESSED/{label}/ to DATA/SPLITS/{split}/{label}/.
  5. Write DATA/METADATA/split_manifest.csv with columns:
         participant_id, label, split  (one row per participant)
  6. Print split table: participants and file counts per split/class.
  7. Print TEST SPLIT LOCKED notice.
  8. Verify no participant_id appears in more than one split.

Run from the project root:

    python -m src.data.split_dataset
"""

from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.utils.config import config

SPLIT_MANIFEST_FIELDS = ["participant_id", "label", "split"]
INCLUDED_LABELS = {"HAPPY", "SAD"}
SPLITS = ("train", "val", "test")


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def load_included(
    manifest_path: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Return (label_of, files_of) from the samples manifest.

    label_of : participant_id -> label  (HAPPY/SAD; excluded rows dropped)
    files_of : participant_id -> list of "{pid}_{task_code}.png" basenames
    """
    label_of: dict[str, str] = {}
    files_of: defaultdict[str, list[str]] = defaultdict(list)

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if _parse_bool(row.get("excluded", "False")):
                continue
            label = (row.get("label") or "").strip().upper()
            if label not in INCLUDED_LABELS:
                continue
            pid = row["participant_id"].strip()
            task = row["task_code"].strip()
            label_of[pid] = label
            files_of[pid].append(f"{pid}_{task}.png")

    return label_of, dict(files_of)


def compute_assignment(label_of: dict[str, str]) -> dict[str, str]:
    """Return participant_id -> split ('train' | 'val' | 'test')."""
    pids = sorted(label_of)  # sorted for reproducibility
    labels = [label_of[p] for p in pids]

    cfg = config.data
    rs = config.training.random_state
    val_of_remainder = cfg.val_ratio / (1.0 - cfg.test_ratio)

    pids_tv, pids_test, labels_tv, _ = train_test_split(
        pids,
        labels,
        test_size=cfg.test_ratio,
        stratify=labels,
        random_state=rs,
    )
    pids_train, pids_val = train_test_split(
        pids_tv,
        test_size=val_of_remainder,
        stratify=labels_tv,
        random_state=rs,
    )

    assignment: dict[str, str] = {}
    for pid in pids_train:
        assignment[pid] = "train"
    for pid in pids_val:
        assignment[pid] = "val"
    for pid in pids_test:
        assignment[pid] = "test"
    return assignment


def copy_splits(
    assignment: dict[str, str],
    label_of: dict[str, str],
    files_of: dict[str, list[str]],
    processed_dir: Path,
    splits_dir: Path,
) -> tuple[int, int]:
    """Copy processed files to split directories.

    Returns (copied, missing) — missing counts source files not yet on disk.
    """
    copied = 0
    missing = 0
    for pid, split in assignment.items():
        label = label_of[pid]
        src_dir = processed_dir / label
        dst_dir = splits_dir / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in files_of.get(pid, []):
            src = src_dir / fname
            dst = dst_dir / fname
            if src.is_file():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
    return copied, missing


def write_split_manifest(
    manifest_path: Path,
    assignment: dict[str, str],
    label_of: dict[str, str],
) -> None:
    rows = [
        {
            "participant_id": pid,
            "label": label_of[pid],
            "split": split,
        }
        for pid, split in sorted(assignment.items())
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SPLIT_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_split_table(
    assignment: dict[str, str],
    label_of: dict[str, str],
    files_of: dict[str, list[str]],
) -> None:
    p_count: Counter = Counter()
    f_count: Counter = Counter()
    for pid, split in assignment.items():
        label = label_of[pid]
        key = (split, label)
        p_count[key] += 1
        f_count[key] += len(files_of.get(pid, []))

    header = f"\n{'Split':<8} {'Label':<8} {'Participants':>13} {'Files':>8}"
    sep = "-" * 42
    print(header)
    print(sep)
    total_p = total_f = 0
    for split in SPLITS:
        for label in ("HAPPY", "SAD"):
            key = (split, label)
            p = p_count[key]
            f = f_count[key]
            total_p += p
            total_f += f
            print(f"{split:<8} {label:<8} {p:>13} {f:>8}")
    print(sep)
    print(f"{'TOTAL':<8} {'':8} {total_p:>13} {total_f:>8}")


def verify_no_leakage(assignment: dict[str, str]) -> bool:
    """Verify each participant_id appears in exactly one split."""
    split_sets: dict[str, set[str]] = {s: set() for s in SPLITS}
    for pid, split in assignment.items():
        split_sets[split].add(pid)

    passed = True
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1 :]:
            overlap = split_sets[s1] & split_sets[s2]
            if overlap:
                print(
                    f"  LEAKAGE: {len(overlap)} participant(s) in "
                    f"both {s1} and {s2}: {sorted(overlap)}"
                )
                passed = False
    return passed


def split_dataset(
    manifest_path: Path,
    processed_dir: Path,
    splits_dir: Path,
    split_manifest_path: Path,
) -> None:
    print("=" * 60)
    print("Phase 8 - SPLIT dataset")
    print("=" * 60)

    if not manifest_path.is_file():
        print(f"ERROR: samples manifest not found: {manifest_path}")
        sys.exit(1)

    label_of, files_of = load_included(manifest_path)

    if not label_of:
        print("ERROR: no included HAPPY/SAD participants found.")
        sys.exit(1)

    class_counts = Counter(label_of.values())
    print(f"Eligible participants: {len(label_of)}")
    for lbl in ("HAPPY", "SAD"):
        print(f"  {lbl}: {class_counts[lbl]}")

    min_class = min(class_counts.values())
    if min_class < 3:
        print(
            "ERROR: too few participants in smallest class "
            f"({min_class}) for a 3-way stratified split."
        )
        sys.exit(1)

    assignment = compute_assignment(label_of)

    copy_counts = Counter(assignment.values())
    for split in SPLITS:
        print(f"  {split}: {copy_counts[split]} participants")

    copied, missing = copy_splits(
        assignment, label_of, files_of, processed_dir, splits_dir
    )
    print(f"\nFiles copied : {copied}")
    if missing:
        print(f"Files missing from DATA/PROCESSED (run Phase 7 first): " f"{missing}")

    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_split_manifest(split_manifest_path, assignment, label_of)
    print(f"Split manifest: {split_manifest_path}")

    print_split_table(assignment, label_of, files_of)

    print(
        "\n*** TEST SPLIT LOCKED — do not evaluate on test data "
        "until final evaluation (Phase 10). ***"
    )

    print("\nLeakage check:")
    if verify_no_leakage(assignment):
        print("  PASSED — no participant_id appears in more than one split.")
    else:
        print("  FAILED — see details above.")


def main() -> int:
    metadata_dir = Path(config.data.metadata_dir)
    split_dataset(
        manifest_path=metadata_dir / "samples_manifest.csv",
        processed_dir=Path(config.data.processed_dir),
        splits_dir=Path(config.data.splits_dir),
        split_manifest_path=metadata_dir / "split_manifest.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
