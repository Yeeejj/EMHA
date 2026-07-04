"""
Phase 3 — Score questionnaire: compute affect composites and emotion labels.

Reads DATA/METADATA/questionnaire_scores.csv (filled in manually), with
columns participant_id and item_01 .. item_24, each a 1-5 Likert response.

Scoring (see config.LabelingConfig):
    happiness_score = sum of the 12 happiness items   (range 12-60)
    sadness_score   = sum of the 12 sadness items     (range 12-60)

Items are summed in their own keyed direction; no within-subscale reversal is
applied, so higher happiness_score = happier and higher sadness_score = sadder.

Labels (thresholds from config.LabelingConfig):
    HAPPY:   happiness_score >= happiness_high AND sadness_score < sadness_threshold
    SAD:     sadness_score >= sadness_threshold OR happiness_score < happiness_low
    NEUTRAL: otherwise (excluded from training)

Every input row is validated: all 24 items present and each value an integer
in 1-5. Invalid rows are skipped with a warning. Results are written to
DATA/METADATA/labels.csv.

Run from the project root:

    python -m src.data.score_questionnaire
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from src.utils.config import config

ITEM_COLUMNS = [f"item_{i:02d}" for i in range(1, 25)]


def parse_item_value(raw: str | None) -> int | None:
    """Return the 1-5 integer value of a cell, or None if invalid/missing."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    cfg = config.labeling
    if value < cfg.likert_min or value > cfg.likert_max:
        return None
    return value


def validate_row(row: dict) -> tuple[dict | None, str]:
    """Validate one CSV row.

    Returns (values, "") on success where values maps item number -> int, or
    (None, reason) when the row is invalid.
    """
    values: dict[int, int] = {}
    for index, column in enumerate(ITEM_COLUMNS, start=1):
        value = parse_item_value(row.get(column))
        if value is None:
            return None, f"{column} missing or not in 1-5"
        values[index] = value
    return values, ""


def score_row(values: dict[int, int]) -> tuple[int, int]:
    """Return (happiness_score, sadness_score) from validated item values."""
    cfg = config.labeling
    happiness_score = sum(values[i] for i in cfg.happiness_items)
    sadness_score = sum(values[i] for i in cfg.sadness_items)
    return happiness_score, sadness_score


def assign_label(happiness_score: int, sadness_score: int) -> str:
    """Apply the HAPPY / SAD / NEUTRAL labeling rules."""
    cfg = config.labeling
    if happiness_score >= cfg.happiness_high and sadness_score < cfg.sadness_threshold:
        return "HAPPY"
    if sadness_score >= cfg.sadness_threshold or happiness_score < cfg.happiness_low:
        return "SAD"
    return "NEUTRAL"


def score_questionnaire(scores_path: Path, labels_path: Path) -> None:
    """Score every valid questionnaire row and write the labels file."""
    print("=" * 60)
    print("Phase 3 - SCORE questionnaire")
    print("=" * 60)

    if not scores_path.is_file():
        print(f"ERROR: questionnaire scores not found: {scores_path}")
        sys.exit(1)

    rows_out: list[dict] = []
    skipped = 0

    with scores_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            participant_id = (row.get("participant_id") or "").strip()
            if not participant_id:
                print("WARNING: row with empty participant_id; skipping.")
                skipped += 1
                continue

            values, reason = validate_row(row)
            if values is None:
                print(f"WARNING: {participant_id}: {reason}; skipping.")
                skipped += 1
                continue

            happiness_score, sadness_score = score_row(values)
            label = assign_label(happiness_score, sadness_score)
            rows_out.append(
                {
                    "participant_id": participant_id,
                    "happiness_score": happiness_score,
                    "sadness_score": sadness_score,
                    "label": label,
                }
            )

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["participant_id", "happiness_score", "sadness_score", "label"]
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    distribution = Counter(r["label"] for r in rows_out)
    print(f"Labels written : {labels_path}")
    print(f"Scored         : {len(rows_out)}")
    print(f"Skipped        : {skipped}")
    print("Class distribution:")
    for label in ("HAPPY", "SAD", "NEUTRAL"):
        print(f"  {label:8s}: {distribution.get(label, 0)}")


def main() -> int:
    metadata_dir = Path(config.data.metadata_dir)
    scores_path = metadata_dir / "questionnaire_scores.csv"
    labels_path = metadata_dir / "labels.csv"
    score_questionnaire(scores_path, labels_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
