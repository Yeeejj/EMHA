"""
Questionnaire reliability report — Stage B.

Reports Cronbach's alpha (with a 95% bootstrap CI and alpha-if-item-deleted
per item), class balance, and score distribution for the FINALE
instrument. Never writes to the export or to labels.csv, and never changes
a score or label (CLAUDE.md Non-Negotiable 3) — this module only reads and
reports.

Run from the project root:

    python -m src.analysis.questionnaire_report
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.labeler import LabelLoader, _normalize_id  # noqa: E402
from src.utils.config import config  # noqa: E402

# The FINALE instrument's fixed 1-5 Likert response range.
LIKERT_MIN = 1
LIKERT_MAX = 5


def _reverse_score(series: pd.Series) -> pd.Series:
    return (LIKERT_MIN + LIKERT_MAX) - series


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha for the columns of items (one column per scale item)."""
    k = items.shape[1]
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def alpha_if_item_deleted(items: pd.DataFrame) -> pd.Series:
    """Alpha recomputed with each single item dropped, indexed by item name."""
    return pd.Series({col: cronbach_alpha(items.drop(columns=[col])) for col in items})


def bootstrap_alpha_ci(items: pd.DataFrame, n: int, seed: int) -> tuple:
    """95% bootstrap CI for alpha, resampling participants with replacement."""
    rng = np.random.default_rng(seed)
    n_rows = len(items)
    alphas = np.empty(n)
    for i in range(n):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        alphas[i] = cronbach_alpha(items.iloc[sample_idx])
    lower, upper = np.percentile(alphas, [2.5, 97.5])
    return float(lower), float(upper)


def _load_items(participant_ids) -> pd.DataFrame:
    """Read and reverse-score the raw item columns for the given participant_ids."""
    lbl = config.labeling
    rep = config.report
    source_csv = Path(
        lbl.source_csv or (config.paths.metadata_dir / "questionnaire_export.csv")
    )
    raw = pd.read_csv(source_csv, dtype=str, keep_default_na=False)

    raw["participant_id"] = raw[lbl.id_column].map(_normalize_id)
    wanted = set(participant_ids)
    raw = raw[raw["participant_id"].isin(wanted)]

    items = raw[list(rep.item_columns)].apply(pd.to_numeric, errors="coerce")
    for col in rep.reverse_items:
        items[col] = _reverse_score(items[col])
    return items.dropna()


def _plot_histogram(df: pd.DataFrame, cutoff: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["total_score"], bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(cutoff, color="red", linestyle="--", label=f"cutoff = {cutoff:g}")
    ax.set_xlabel("total_score")
    ax.set_ylabel("Participants")
    ax.set_title("Score distribution")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_report() -> Path:
    """Compute reliability/balance/distribution and write results/questionnaire/.

    Asserts that label counts here match labels.csv exactly (labels.csv
    must already exist, produced by python -m src.data.labeler) — raises
    AssertionError loudly if the two disagree, e.g. because the export
    changed since labels.csv was last written. Never writes to the export
    or to labels.csv.
    """
    lbl = config.labeling
    rep = config.report

    df = LabelLoader.load()  # participant_id, total_score, label (verbatim)

    labels_path = config.paths.metadata_dir / "labels.csv"
    if not labels_path.is_file():
        print(f"ERROR: {labels_path} not found. Run python -m src.data.labeler first.")
        sys.exit(1)
    labels_csv = pd.read_csv(labels_path, dtype={"participant_id": str})

    fresh_counts = df["label"].value_counts().sort_index()
    csv_counts = labels_csv["label"].value_counts().sort_index()
    if not fresh_counts.equals(csv_counts):
        raise AssertionError(
            f"Label counts from the export ({fresh_counts.to_dict()}) do not match "
            f"labels.csv ({csv_counts.to_dict()}) — labels.csv is stale. "
            "Re-run python -m src.data.labeler."
        )

    items = _load_items(df["participant_id"])
    alpha = cronbach_alpha(items)
    ci_low, ci_high = bootstrap_alpha_ci(
        items, n=rep.n_bootstrap, seed=config.training.seed
    )
    deleted = alpha_if_item_deleted(items)

    out_dir = config.paths.results_dir / "questionnaire"
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_histogram(df, lbl.cutoff, out_dir / "score_distribution.png")

    overall_counts = labels_csv["label"].value_counts()
    primary = labels_csv[labels_csv["in_primary_analysis"]]
    primary_counts = primary["label"].value_counts()
    middle_band_count = int(labels_csv["in_middle_band"].sum())

    score_ranges = {}
    for label in ("HAPPY", "SAD"):
        subset = primary.loc[primary["label"] == label, "total_score"]
        if len(subset):
            score_ranges[label] = (float(subset.min()), float(subset.max()))
        else:
            score_ranges[label] = (float("nan"), float("nan"))

    lines = [
        "QUESTIONNAIRE RELIABILITY REPORT",
        "=" * 40,
        "",
        f"N participants (labeled, complete items) : {len(items)}",
        f"Cronbach's alpha ({items.shape[1]} items)          : {alpha:.4f}",
        f"  95% bootstrap CI (n={rep.n_bootstrap})          : "
        f"[{ci_low:.4f}, {ci_high:.4f}]",
        "",
        "Alpha if item deleted:",
    ]
    for item, value in deleted.items():
        lines.append(f"  {item:>5s}: {value:.4f}")
    lines += [
        "",
        "Class balance (overall):",
        f"  HAPPY: {overall_counts.get('HAPPY', 0)}  "
        f"SAD: {overall_counts.get('SAD', 0)}",
        "Class balance (primary analysis set):",
        f"  HAPPY: {primary_counts.get('HAPPY', 0)}  "
        f"SAD: {primary_counts.get('SAD', 0)}",
        "",
        f"Middle-band count : {middle_band_count}",
        "Score range kept per class (primary set):",
    ]
    for label, (lo, hi) in score_ranges.items():
        lines.append(f"  {label}: [{lo:g}, {hi:g}]")
    lines += [
        "",
        f"Cutoff           : {lbl.cutoff}",
        f"Analysis design  : {lbl.analysis_design}",
        "",
    ]

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> int:
    report_path = build_report()
    print(f"Report written: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
