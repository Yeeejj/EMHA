"""
Phase 8 — Participant-aware splitting.

HARD CONSTRAINT (Section A Rule 3): All 24 crops of one participant must
stay in the same split/fold.  File-level splitting causes data leakage.

Reads DATA/METADATA/crop_index.csv (after Phase 6 QC).
Collects unique participants that have at least one non-excluded crop.
Performs stratified splits at the RESPONDENT level, then emits crop lists.

Writes DATA/METADATA/splits.json:
    {
      "train": ["P001", ...],
      "val":   ["P010", ...],
      "test":  ["P020", ...],
      "folds": [
          {"train": [...], "val": [...]},
          ...
      ],
      "label_map": {"P001": "HAPPY", ...}
    }

Run from the project root:

    python -m src.data.participant_aware_splitter
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.config import config

INCLUDED_LABELS = {"HAPPY", "SAD"}


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def load_usable(index_path: Path) -> dict[str, str]:
    """Return participant_id -> label for participants with at least 1 usable crop."""
    label_of: dict[str, str] = {}
    with index_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if _parse_bool(row.get("excluded", "False")):
                continue
            label = (row.get("label") or "").strip().upper()
            if label not in INCLUDED_LABELS:
                continue
            pid = row["participant_id"].strip()
            label_of[pid] = label
    return label_of


def stratified_split(
    label_of: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Return (train_pids, val_pids, test_pids) stratified at participant level."""
    cfg_d = config.data
    cfg_t = config.training
    pids = sorted(label_of)
    labels = [label_of[p] for p in pids]

    val_of_remainder = cfg_d.val_ratio / (1.0 - cfg_d.test_ratio)

    pids_tv, pids_test, labels_tv, _ = train_test_split(
        pids, labels,
        test_size=cfg_d.test_ratio,
        stratify=labels,
        random_state=cfg_t.random_state,
    )
    pids_train, pids_val = train_test_split(
        pids_tv, labels_tv,
        test_size=val_of_remainder,
        stratify=labels_tv,
        random_state=cfg_t.random_state,
    )
    return pids_train, pids_val, pids_test


def build_folds(pids_train: list[str], label_of: dict[str, str]) -> list[dict]:
    """Return list of {train, val} fold dicts stratified within the train split."""
    cfg_t = config.training
    pids = sorted(pids_train)
    labels = [label_of[p] for p in pids]

    skf = StratifiedKFold(
        n_splits=cfg_t.n_folds,
        shuffle=True,
        random_state=cfg_t.random_state,
    )
    folds = []
    for train_idx, val_idx in skf.split(pids, labels):
        folds.append({
            "train": [pids[i] for i in train_idx],
            "val":   [pids[i] for i in val_idx],
        })
    return folds


def verify_no_leakage(
    pids_train: list[str],
    pids_val: list[str],
    pids_test: list[str],
) -> bool:
    sets = {"train": set(pids_train), "val": set(pids_val), "test": set(pids_test)}
    names = list(sets.keys())
    passed = True
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap = sets[n1] & sets[n2]
            if overlap:
                print(f"  LEAKAGE: {len(overlap)} participant(s) in both {n1} and {n2}")
                passed = False
    return passed


def split_participants(index_path: Path, splits_path: Path) -> None:
    print("=" * 60)
    print("Phase 8 - PARTICIPANT-AWARE SPLIT -> splits.json")
    print("=" * 60)

    if not index_path.is_file():
        print(f"ERROR: crop index not found: {index_path}")
        sys.exit(1)

    label_of = load_usable(index_path)
    if not label_of:
        print("ERROR: no usable HAPPY/SAD participants found.")
        sys.exit(1)

    counts = Counter(label_of.values())
    print(f"Eligible participants: {len(label_of)}")
    print(f"  HAPPY: {counts['HAPPY']}  SAD: {counts['SAD']}")

    if min(counts.values()) < config.training.n_folds:
        print("ERROR: smallest class too small for stratified CV.")
        sys.exit(1)

    pids_train, pids_val, pids_test = stratified_split(label_of)
    folds = build_folds(pids_train, label_of)

    print("\nSplit sizes (participants):")
    for name, pids in (("train", pids_train), ("val", pids_val), ("test", pids_test)):
        c = Counter(label_of[p] for p in pids)
        print(f"  {name:6s}: {len(pids):3d}  (HAPPY={c['HAPPY']}, SAD={c['SAD']})")

    print("\nLeakage check: ", end="")
    if verify_no_leakage(pids_train, pids_val, pids_test):
        print("PASSED")
    else:
        print("FAILED")

    payload = {
        "train":        sorted(pids_train),
        "val":          sorted(pids_val),
        "test":         sorted(pids_test),
        "folds":        folds,
        "label_map":    {p: label_of[p] for p in sorted(label_of)},
        "n_folds":      config.training.n_folds,
        "random_state": config.training.random_state,
    }

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nsplits.json written: {splits_path}")
    print("*** TEST SPLIT LOCKED - do not evaluate until Phase 10. ***")


def main() -> int:
    meta = Path(config.data.metadata_dir)
    split_participants(
        index_path=meta / "crop_index.csv",
        splits_path=meta / "splits.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
