"""
Phase 13 — Psychometrics: validate the FINALE instrument.

Computes Cronbach's alpha for the 24-item FINALE scale (overall and per
subscale), item-total correlations, and score distribution statistics.
Supervised by the four licensed psychometricians listed in SYSTEMS_GUIDE.md.

Reads DATA/METADATA/questionnaire_scores.csv (item_01..item_24).
Writes results to DATA/METADATA/psychometrics_report.csv.

Run from the project root:

    python -m src.analysis.psychometrics
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import config

ITEM_COLS = [f"item_{i:02d}" for i in range(1, 25)]


def cronbach_alpha(df: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for item columns in df."""
    k = df.shape[1]
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def item_total_correlations(df: pd.DataFrame) -> pd.Series:
    """Pearson correlation of each item with the scale total."""
    totals = df.sum(axis=1)
    return df.corrwith(totals)


def run_psychometrics(scores_path: Path, report_path: Path) -> None:
    print("=" * 60)
    print("Phase 13 - PSYCHOMETRICS")
    print("=" * 60)

    if not scores_path.is_file():
        print(f"ERROR: questionnaire scores not found: {scores_path}")
        sys.exit(1)

    df_raw = pd.read_csv(scores_path)
    df_items = df_raw[ITEM_COLS].apply(pd.to_numeric, errors="coerce").dropna()

    cfg = config.labeling

    # Overall alpha
    alpha_overall = cronbach_alpha(df_items)

    # Subscale alpha
    h_cols = [f"item_{i:02d}" for i in cfg.happiness_items]
    s_cols = [f"item_{i:02d}" for i in cfg.sadness_items]
    alpha_happiness = cronbach_alpha(df_items[h_cols])
    alpha_sadness = cronbach_alpha(df_items[s_cols])

    # Item-total correlations
    itc = item_total_correlations(df_items)

    # Score distribution
    h_sums = df_items[h_cols].sum(axis=1)
    s_sums = df_items[s_cols].sum(axis=1)
    adj_totals = h_sums + (72 - s_sums)

    print(f"\nN valid respondents : {len(df_items)}")
    print(f"Cronbach alpha (overall 24-item) : {alpha_overall:.4f}")
    print(f"Cronbach alpha (happiness 12-item): {alpha_happiness:.4f}")
    print(f"Cronbach alpha (sadness  12-item) : {alpha_sadness:.4f}")
    print(f"\nadjusted_total: mean={adj_totals.mean():.2f} "
          f"sd={adj_totals.std():.2f} "
          f"min={adj_totals.min():.0f} max={adj_totals.max():.0f}")

    report_rows = [
        {"metric": "n_valid",              "value": len(df_items)},
        {"metric": "alpha_overall",        "value": round(alpha_overall, 4)},
        {"metric": "alpha_happiness",      "value": round(alpha_happiness, 4)},
        {"metric": "alpha_sadness",        "value": round(alpha_sadness, 4)},
        {"metric": "adj_total_mean",       "value": round(float(adj_totals.mean()), 4)},
        {"metric": "adj_total_sd",         "value": round(float(adj_totals.std()), 4)},
        {"metric": "adj_total_min",        "value": int(adj_totals.min())},
        {"metric": "adj_total_max",        "value": int(adj_totals.max())},
    ]
    for item_col in ITEM_COLS:
        report_rows.append({
            "metric": f"itc_{item_col}",
            "value":  round(float(itc[item_col]), 4),
        })

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nPsychometrics report: {report_path}")


def main() -> int:
    meta = Path(config.data.metadata_dir)
    run_psychometrics(
        scores_path=meta / "questionnaire_scores.csv",
        report_path=meta / "psychometrics_report.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
