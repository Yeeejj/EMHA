"""
Phase 3 — Score questionnaire: compute FINALE composites and binary labels.

Reads DATA/METADATA/questionnaire_scores.csv with columns:
    participant_id, age, gender, item_01 .. item_24   (Likert 1-5)

Scoring — ProcessPipeline.txt Section A (non-negotiable):
    happiness_items = (2,4,6,8,10,12,13,16,19,20,21,23)  kept raw
    sadness_items   = (1,3,5,7,9,11,14,15,17,18,22,24)   kept raw
    happiness_sum   = sum of happiness items              range 12-60
    sadness_sum     = sum of sadness items                range 12-60
    adjusted_total  = happiness_sum + (72 - sadness_sum)  range 24-120
    label           = HAPPY if adjusted_total >= 72 else SAD
    mean_adjusted   = adjusted_total / 24
    p_happy         = (mean_adjusted - 1) / 4             stored only

NO NEUTRAL class. All 400 respondents receive a label. Integer comparisons
only at the >= 72 boundary.

Writes DATA/METADATA/labels.csv:
    participant_id, age, gender, happiness_sum, sadness_sum,
    adjusted_total, mean_adjusted, p_happy, label

Run from the project root:

    python -m src.data.questionnaire_scorer
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from src.utils.config import config

ITEM_COLUMNS = [f"item_{i:02d}" for i in range(1, 25)]

LABELS_FIELDS = [
    "participant_id", "age", "gender",
    "happiness_sum", "sadness_sum", "adjusted_total",
    "mean_adjusted", "p_happy", "label",
]


def parse_item(raw: str | None) -> int | None:
    """Return the 1-5 integer value of a Likert cell, or None if invalid."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        return None
    cfg = config.labeling
    return v if cfg.likert_min <= v <= cfg.likert_max else None


def validate_items(row: dict) -> tuple[dict[int, int] | None, str]:
    """Validate all 24 item cells; return (item_map, "") or (None, reason)."""
    items: dict[int, int] = {}
    for idx, col in enumerate(ITEM_COLUMNS, start=1):
        v = parse_item(row.get(col))
        if v is None:
            return None, f"{col} missing or not in 1-5"
        items[idx] = v
    return items, ""


def score(items: dict[int, int]) -> tuple[int, int, int, float, float]:
    """Compute composites from validated items.

    Returns (happiness_sum, sadness_sum, adjusted_total, mean_adjusted, p_happy).
    """
    cfg = config.labeling
    h_sum = sum(items[i] for i in cfg.happiness_items)
    s_sum = sum(items[i] for i in cfg.sadness_items)
    adj = h_sum + (72 - s_sum)
    mean_adj = adj / 24
    p_happy = (mean_adj - 1) / 4
    return h_sum, s_sum, adj, round(mean_adj, 6), round(p_happy, 6)


def assign_label(adjusted_total: int) -> str:
    """Binary label: HAPPY if adjusted_total >= 72, SAD otherwise."""
    return "HAPPY" if adjusted_total >= config.labeling.adjusted_total_threshold else "SAD"


def score_questionnaire(scores_path: Path, labels_path: Path) -> None:
    print("=" * 60)
    print("Phase 3 - SCORE questionnaire (Section A)")
    print("=" * 60)

    if not scores_path.is_file():
        print(f"ERROR: questionnaire scores not found: {scores_path}")
        sys.exit(1)

    rows_out: list[dict] = []
    skipped = 0

    with scores_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("participant_id") or "").strip()
            if not pid:
                print("WARNING: empty participant_id; skipping.")
                skipped += 1
                continue

            items, reason = validate_items(row)
            if items is None:
                print(f"WARNING: {pid}: {reason}; skipping.")
                skipped += 1
                continue

            h_sum, s_sum, adj, mean_adj, p_happy = score(items)
            label = assign_label(adj)

            rows_out.append({
                "participant_id": pid,
                "age":            (row.get("age") or "").strip(),
                "gender":         (row.get("gender") or "").strip(),
                "happiness_sum":  h_sum,
                "sadness_sum":    s_sum,
                "adjusted_total": adj,
                "mean_adjusted":  mean_adj,
                "p_happy":        p_happy,
                "label":          label,
            })

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABELS_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    dist = Counter(r["label"] for r in rows_out)
    print(f"Labels written : {labels_path}")
    print(f"Scored         : {len(rows_out)}")
    print(f"Skipped        : {skipped}")
    print(f"HAPPY          : {dist['HAPPY']}")
    print(f"SAD            : {dist['SAD']}")
    print("(No NEUTRAL class - all respondents labelled)")


def main() -> int:
    meta = Path(config.data.metadata_dir)
    score_questionnaire(
        scores_path=meta / "questionnaire_scores.csv",
        labels_path=meta / "labels.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
