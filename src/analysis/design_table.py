"""
Design table — Stage B. Previews the extreme-groups middle-band tradeoff
across candidate fractions, using labels.csv only (never an image,
feature, or model output) — so the band choice is made from score data
alone, before any model result exists.

Run from the project root:

    python -m src.analysis.design_table
"""

from __future__ import annotations

import sys

import pandas as pd

from src.utils.config import config

BAND_TABLE_FIELDS = [
    "fraction",
    "n_middle_band",
    "n_kept_happy",
    "n_kept_sad",
    "n_kept_total",
    "happy_score_min",
    "happy_score_max",
    "sad_score_min",
    "sad_score_max",
]


def band_table(labels: pd.DataFrame, fractions: list) -> pd.DataFrame:
    """For each candidate middle_band_fraction, preview the extreme-groups split.

    labels must have participant_id, total_score, label columns (exactly
    what labels.csv provides). Mirrors LabelLoader.write()'s selection rule
    exactly (round(), stable sort by boundary_distance from
    config.labeling.cutoff) so this preview matches what writing that
    fraction would actually produce.
    """
    cutoff = config.labeling.cutoff
    distance = (labels["total_score"] - cutoff).abs()

    rows = []
    for fraction in fractions:
        n_middle = round(len(labels) * fraction)
        middle_ids = set(
            labels.assign(_distance=distance)
            .sort_values("_distance", kind="stable")
            .head(n_middle)["participant_id"]
        )
        kept = labels[~labels["participant_id"].isin(middle_ids)]

        happy = kept.loc[kept["label"] == "HAPPY", "total_score"]
        sad = kept.loc[kept["label"] == "SAD", "total_score"]

        rows.append(
            {
                "fraction": fraction,
                "n_middle_band": len(middle_ids),
                "n_kept_happy": len(happy),
                "n_kept_sad": len(sad),
                "n_kept_total": len(kept),
                "happy_score_min": float(happy.min()) if len(happy) else float("nan"),
                "happy_score_max": float(happy.max()) if len(happy) else float("nan"),
                "sad_score_min": float(sad.min()) if len(sad) else float("nan"),
                "sad_score_max": float(sad.max()) if len(sad) else float("nan"),
            }
        )

    return pd.DataFrame(rows, columns=BAND_TABLE_FIELDS)


def main() -> int:
    labels_path = config.paths.metadata_dir / "labels.csv"
    if not labels_path.is_file():
        print(f"ERROR: {labels_path} not found. Run python -m src.data.labeler first.")
        return 1

    labels = pd.read_csv(labels_path, dtype={"participant_id": str})
    fractions = [0.20, 0.25, 0.30, 0.35, 0.40]
    table = band_table(labels, fractions)

    print(table.to_string(index=False))
    print()
    print(f"min_per_class = {config.labeling.min_per_class}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
