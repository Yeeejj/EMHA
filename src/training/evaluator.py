"""
Participant-level evaluation of every model under the frozen protocol — Stage H1.

Reads the runners' predictions (shared schema: analysis, model, feature_set,
fold, participant_id, label, prob_sad, pred, in_middle_band):

    results/baselines/<name>/predictions.csv
    results/cnn/<name>/predictions.csv
    results/hybrid/<name>/predictions.csv
    results/ensemble/<analysis>/predictions.csv    (only when no subset)

with <name> = <analysis>[_<subset>] (EvalConfig.sources). A missing file is
reported, never invented. Every file is validated: schema, one analysis, labels
and preds in {HAPPY, SAD}, pred == SAD iff prob_sad >= AggregateConfig.threshold,
one prediction per participant per (analysis, model, feature_set), and one
label per participant across models.

Per analysis set (primary first) and (model, feature_set), following
EVALUATION_PROTOCOL.md sections 2 and 4:

  - The primary analysis drops in_middle_band rows (the --predict-middle
    extras, reported by H3); the full analysis keeps every row.
  - A (model, feature_set) that does not cover all CVConfig.n_splits folds is
    listed as skipped, not evaluated.
  - macro-F1, accuracy, balanced accuracy, precision and recall per class
    (SAD and HAPPY), and ROC-AUC (SAD positive): pooled over the out-of-fold
    participants, and per fold (mean +/- sample std, ddof=1).
  - 95% percentile CIs from EvalConfig.n_bootstrap participant resamples of
    the pooled predictions (seed TrainingConfig.seed). Rows are sorted by
    participant_id first, so every model of an analysis is resampled with the
    same participants and the paired macro-F1 difference against
    EvalConfig.reference_model / reference_feature_set uses the same
    resamples.
  - The majority baseline (training-fold majority label, ties -> SAD, as in
    src.utils.run_log) is shown beside every accuracy.

Outputs, results/final[_<subset>]/:

    model_comparison.csv / .txt     one row per analysis, model, feature_set
    per_fold_metrics.csv            every fold, every model
    confusion_matrices.csv
    confusion_matrices_<analysis>.png, roc_curves_<analysis>.png
    reliability_<model>_<analysis>.png   the pre-registered primary model
                                         (EvalConfig.reliability_model)

Each (analysis, model, feature_set) appends a row to results/RUN_LOG.csv with
macro_f1_ci_low/high filled (src.utils.run_log.log_run). Refuses to run on
real data before the protocol-frozen tag.

    python -m src.training.evaluator [--analysis primary|full]
        [--subset pilot] [--stage smoke|pilot|full]
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.training.run_baselines import EXPLORATORY_MODELS, PRED_COLUMNS
from src.training.splits import SUBSETS
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic
from src.utils.run_log import STAGES, command_line, log_run, resolve_stage

plt.switch_backend("Agg")

SAD, HAPPY = "SAD", "HAPPY"
GROUP = ["analysis", "model", "feature_set"]
METRICS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "precision_sad",
    "recall_sad",
    "precision_happy",
    "recall_happy",
    "roc_auc",
)

# Chart tokens: light chart surface, ink, and one sequential blue hue.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SERIES = "#2a78d6"
_SEQUENTIAL = LinearSegmentedColormap.from_list("emha_blue", ["#cde2fb", "#104281"])
_PNG_META = {"Software": None}  # no version string -> byte-identical reruns


# ─── loading and validation ───────────────────────────────────────────────────


def _analyses(analysis: str | None, cfg=config) -> tuple:
    if analysis is None:
        return tuple(cfg.evaluation.analyses)
    if analysis not in cfg.evaluation.analyses:
        raise ValueError(f"analysis must be one of {cfg.evaluation.analyses}")
    return (analysis,)


def prediction_paths(
    analysis: str | None = None, subset: str | None = None, cfg=config
) -> list:
    """(analysis, source, path) for every predictions.csv the evaluator reads."""
    out = []
    for name in _analyses(analysis, cfg):
        for source in cfg.evaluation.sources:
            if source == "ensemble":
                if subset is not None:
                    continue  # run_ensemble has no subset runs
                folder = name
            else:
                folder = name if subset is None else f"{name}_{subset}"
            path = cfg.paths.results_dir / source / folder / "predictions.csv"
            out.append((name, source, path))
    return out


def _as_bool(column: pd.Series, where: str) -> pd.Series:
    mapping = {True: True, False: False, "True": True, "False": False}
    values = column.map(lambda v: mapping.get(v, None))
    if values.isna().any():
        raise ValueError(f"{where}: in_middle_band must be True/False")
    return values.astype(bool)


def validate(frame: pd.DataFrame, analysis: str, where: str, cfg=config):
    """Check one predictions file; return it with clean dtypes."""
    missing = set(PRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{where}: missing columns {sorted(missing)}")
    frame = frame[list(PRED_COLUMNS)].copy()
    if frame.isna().any().any():
        raise ValueError(f"{where}: empty cells")
    if set(frame["analysis"]) != {analysis}:
        raise ValueError(
            f"{where}: analysis {sorted(set(frame['analysis']))}, expected {analysis}"
        )
    for col in ("label", "pred"):
        bad = set(frame[col]) - {SAD, HAPPY}
        if bad:
            raise ValueError(f"{where}: {col} values {sorted(bad)} not HAPPY/SAD")
    prob = frame["prob_sad"].astype(float)
    if ((prob < 0) | (prob > 1)).any():
        raise ValueError(f"{where}: prob_sad outside [0, 1]")
    thr = cfg.aggregate.threshold
    if (frame["pred"] != np.where(prob >= thr, SAD, HAPPY)).any():
        raise ValueError(f"{where}: pred disagrees with prob_sad >= {thr}")
    frame["prob_sad"] = prob
    frame["fold"] = frame["fold"].astype(int)
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["in_middle_band"] = _as_bool(frame["in_middle_band"], where)
    dup = frame[frame.duplicated(GROUP + ["participant_id"], keep=False)]
    if not dup.empty:
        first = dup.iloc[0]
        raise ValueError(
            f"{where}: more than one prediction for participant "
            f"{first['participant_id']} in {first['model']}/{first['feature_set']}"
        )
    return frame


def load_predictions(
    analysis: str | None = None, subset: str | None = None, cfg=config
) -> pd.DataFrame:
    """All available, validated predictions for the requested analysis sets."""
    frames = []
    for name, source, path in prediction_paths(analysis, subset, cfg):
        if not path.is_file():
            print(f"evaluator: {path} not found; {source}/{name} not evaluated")
            continue
        raw = pd.read_csv(path, dtype={"participant_id": str})
        frames.append(validate(raw, name, str(path), cfg))
    if not frames:
        raise FileNotFoundError(
            f"no predictions.csv under {cfg.paths.results_dir} for "
            f"{_analyses(analysis, cfg)} / subset {subset}"
        )
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(GROUP + ["participant_id"])
    if dup.any():
        first = df[dup].iloc[0]
        raise ValueError(
            f"{first['model']}/{first['feature_set']} ({first['analysis']}) "
            "appears in more than one predictions file"
        )
    labels = df.groupby(["analysis", "participant_id"])["label"].nunique()
    if (labels > 1).any():
        pid = labels[labels > 1].index[0]
        raise ValueError(f"models disagree on the label of participant {pid}")
    return df


def analysis_frame(df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """Rows evaluated for one analysis: primary drops the middle band."""
    frame = df[df["analysis"] == analysis]
    if analysis == "primary":
        frame = frame[~frame["in_middle_band"]]
    return frame


# ─── metrics ──────────────────────────────────────────────────────────────────


def _arrays(df: pd.DataFrame) -> tuple:
    y = (df["label"] == SAD).to_numpy()
    p = (df["pred"] == SAD).to_numpy()
    return y, p, df["prob_sad"].to_numpy(dtype=float)


def metrics(df: pd.DataFrame) -> dict:
    """Participant-level metrics of one prediction frame (SAD positive)."""
    y, p, prob = _arrays(df)
    both = [False, True]
    prec = precision_score(y, p, labels=both, average=None, zero_division=0)
    rec = recall_score(y, p, labels=both, average=None, zero_division=0)
    two_classes = len(set(y)) == 2
    return {
        "macro_f1": float(
            f1_score(y, p, labels=both, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, p)) if two_classes else math.nan
        ),
        "precision_sad": float(prec[1]),
        "recall_sad": float(rec[1]),
        "precision_happy": float(prec[0]),
        "recall_happy": float(rec[0]),
        "roc_auc": float(roc_auc_score(y, prob)) if two_classes else math.nan,
    }


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros(len(num), dtype=float)
    return np.divide(num, den, out=out, where=den > 0)


def _resampled_metrics(df: pd.DataFrame, idx: np.ndarray) -> dict:
    """metrics() for every row of idx (n_resamples x n_rows) at once."""
    y_all, p_all, prob_all = _arrays(df)
    y, p, prob = y_all[idx], p_all[idx], prob_all[idx]
    tp = (y & p).sum(axis=1)
    fp = (~y & p).sum(axis=1)
    fn = (y & ~p).sum(axis=1)
    tn = (~y & ~p).sum(axis=1)
    rec_sad, rec_happy = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    n_pos = y.sum(axis=1)
    n_neg = y.shape[1] - n_pos
    two_classes = (n_pos > 0) & (n_neg > 0)
    rank_sum = (rankdata(prob, axis=1) * y).sum(axis=1)
    auc = np.full(len(idx), math.nan)
    auc[two_classes] = (
        rank_sum[two_classes] - n_pos[two_classes] * (n_pos[two_classes] + 1) / 2
    ) / (n_pos[two_classes] * n_neg[two_classes])
    return {
        "macro_f1": (
            _ratio(2 * tp, 2 * tp + fp + fn) + _ratio(2 * tn, 2 * tn + fn + fp)
        )
        / 2,
        "accuracy": (tp + tn) / y.shape[1],
        "balanced_accuracy": np.where(two_classes, (rec_sad + rec_happy) / 2, math.nan),
        "precision_sad": _ratio(tp, tp + fp),
        "recall_sad": rec_sad,
        "precision_happy": _ratio(tn, tn + fn),
        "recall_happy": rec_happy,
        "roc_auc": auc,
    }


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in participant order, so equal participant sets resample alike."""
    return df.sort_values("participant_id", kind="mergesort").reset_index(drop=True)


def _bootstrap(df: pd.DataFrame, n: int, seed: int) -> dict:
    df = _sorted(df)
    idx = np.random.default_rng(seed).integers(0, len(df), size=(n, len(df)))
    return _resampled_metrics(df, idx)


def _percentile_ci(values: np.ndarray, cfg=config) -> tuple:
    finite = values[~np.isnan(values)]
    if not finite.size:
        return math.nan, math.nan
    tail = (1 - cfg.evaluation.ci_level) / 2 * 100
    lo, hi = np.percentile(finite, [tail, 100 - tail])
    return float(lo), float(hi)


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def bootstrap_ci(df: pd.DataFrame, metric: str, n: int, seed: int) -> tuple:
    """Percentile CI of metric over n participant resamples."""
    _check_metric(metric)
    return _percentile_ci(_bootstrap(df, n, seed)[metric])


def _check_paired(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    ids_a, ids_b = list(_sorted(a)["participant_id"]), list(
        _sorted(b)["participant_id"]
    )
    if ids_a != ids_b:
        diff = sorted(set(ids_a) ^ set(ids_b))
        raise ValueError(
            f"{what}: participant sets differ ({len(diff)} not shared, "
            f"e.g. {diff[:3]}); a paired comparison needs the same participants"
        )


def _paired(boot_a: dict, boot_b: dict, metric: str) -> tuple:
    diff = boot_a[metric] - boot_b[metric]
    lo, hi = _percentile_ci(diff)
    return float(np.nanmean(diff)), lo, hi


def paired_diff_ci(
    df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, n: int, seed: int
) -> tuple:
    """(mean, ci_low, ci_high) of metric(a) - metric(b) on shared resamples."""
    _check_metric(metric)
    _check_paired(df_a, df_b, "paired_diff_ci")
    return _paired(_bootstrap(df_a, n, seed), _bootstrap(df_b, n, seed), metric)


def majority_hits(df: pd.DataFrame, cfg=config) -> pd.Series:
    """Per row: is the fold's training-majority label (ties -> SAD) correct?"""
    hits = pd.Series(False, index=df.index)
    for fold, part in df.groupby("fold"):
        train = df[df["fold"] != fold]
        share = float((train["label"] == SAD).mean()) if len(train) else 0.0
        majority = SAD if share >= cfg.aggregate.threshold else HAPPY
        hits.loc[part.index] = part["label"] == majority
    return hits


# ─── evaluation ───────────────────────────────────────────────────────────────


@dataclass
class Evaluation:
    comparison: pd.DataFrame
    per_fold: pd.DataFrame
    confusion: pd.DataFrame
    groups: dict = field(default_factory=dict)  # (analysis, model, fs) -> frame
    skipped: list = field(default_factory=list)


def _per_fold(group: pd.DataFrame, hits: pd.Series, key: tuple) -> pd.DataFrame:
    rows = []
    for fold, part in group.groupby("fold"):
        rows.append(
            {
                **dict(zip(GROUP, key)),
                "fold": int(fold),
                "n_participants": len(part),
                **metrics(part),
                "majority_baseline": float(hits.loc[part.index].mean()),
            }
        )
    return pd.DataFrame(rows)


def _confusion(group: pd.DataFrame, key: tuple) -> dict:
    y, p, _ = _arrays(group)
    return {
        **dict(zip(GROUP, key)),
        "tn": int((~y & ~p).sum()),
        "fp": int((~y & p).sum()),
        "fn": int((y & ~p).sum()),
        "tp": int((y & p).sum()),
    }


def evaluate(df: pd.DataFrame, cfg=config) -> Evaluation:
    """Pooled, per-fold, bootstrap and paired results for every group."""
    ev = cfg.evaluation
    seed = cfg.training.seed
    all_folds = set(range(cfg.cv.n_splits))
    ref_key = (ev.reference_model, ev.reference_feature_set)
    rows, fold_frames, confusion = [], [], []
    result = Evaluation(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    present = set(df["analysis"])
    for analysis in [a for a in ev.analyses if a in present]:
        groups = {}
        frame = analysis_frame(df, analysis)
        for (model, fs), g in frame.groupby(["model", "feature_set"], sort=False):
            got = set(g["fold"])
            if got != all_folds:
                result.skipped.append(
                    f"{analysis} {model}/{fs}: folds {sorted(got)}, "
                    f"need all {cfg.cv.n_splits}"
                )
                continue
            groups[(model, fs)] = _sorted(g)
        boots = {k: _bootstrap(g, ev.n_bootstrap, seed) for k, g in groups.items()}

        for (model, fs), g in groups.items():
            key = (analysis, model, fs)
            result.groups[key] = g
            hits = majority_hits(g, cfg)
            pf = _per_fold(g, hits, key)
            fold_frames.append(pf)
            confusion.append(_confusion(g, key))
            n_sad = int((g["label"] == SAD).sum())
            row = {
                **dict(zip(GROUP, key)),
                "exploratory": model in EXPLORATORY_MODELS,
                "n_participants": len(g),
                "n_sad": n_sad,
                "n_happy": len(g) - n_sad,
                "majority_baseline": float(hits.mean()),
                "majority_baseline_fold_mean": float(pf["majority_baseline"].mean()),
            }
            point = metrics(g)
            for m in METRICS:
                lo, hi = _percentile_ci(boots[(model, fs)][m], cfg)
                row[m] = point[m]
                row[f"{m}_ci_low"] = lo
                row[f"{m}_ci_high"] = hi
                row[f"{m}_fold_mean"] = float(pf[m].mean())
                row[f"{m}_fold_std"] = float(pf[m].std(ddof=1))
            row["reference"] = "/".join(ref_key) if ref_key in groups else ""
            delta = (math.nan, math.nan, math.nan)
            if ref_key in groups and (model, fs) != ref_key:
                _check_paired(g, groups[ref_key], f"{analysis} {model}/{fs}")
                delta = _paired(boots[(model, fs)], boots[ref_key], "macro_f1")
            (
                row["delta_macro_f1_vs_reference"],
                row["delta_macro_f1_ci_low"],
                row["delta_macro_f1_ci_high"],
            ) = delta
            rows.append(row)

    result.comparison = pd.DataFrame(rows)
    result.per_fold = (
        pd.concat(fold_frames, ignore_index=True) if fold_frames else (pd.DataFrame())
    )
    result.confusion = pd.DataFrame(confusion)
    return result


# ─── text report ──────────────────────────────────────────────────────────────


def _ci(row, m: str) -> str:
    return f"{row[m]:.3f} [{row[f'{m}_ci_low']:.3f}, {row[f'{m}_ci_high']:.3f}]"


def _text_report(result: Evaluation, missing: list, subset, cfg=config) -> str:
    ev = cfg.evaluation
    pct = round(ev.ci_level * 100)
    lines = [
        "EMHA participant-level evaluation (EVALUATION_PROTOCOL.md sections 2, 4)",
        f"Subset: {subset or 'all participants'}",
        f"Bootstrap: {ev.n_bootstrap} participant resamples of the pooled "
        f"out-of-fold predictions, seed {cfg.training.seed}, {pct}% percentile CIs.",
        f"Paired comparison: macro-F1 difference vs {ev.reference_model}/"
        f"{ev.reference_feature_set} on the same resamples (mean [CI]).",
        "Primary analysis excludes in_middle_band rows; full analysis keeps all.",
        f"Majority baseline: training-fold majority label per outer fold, ties "
        f"-> SAD. Fold values: mean +/- sample std over {cfg.cv.n_splits} folds.",
        "* = exploratory, never part of a confirmatory claim.",
        "",
    ]
    comp = result.comparison
    for analysis in [a for a in ev.analyses if a in set(comp.get("analysis", []))]:
        part = comp[comp["analysis"] == analysis]
        first = part.iloc[0]
        lines += [
            f"== {analysis} ==  n = {first['n_participants']} participants "
            f"(SAD {first['n_sad']}, HAPPY {first['n_happy']}), majority "
            f"baseline accuracy {first['majority_baseline']:.3f}",
            "",
        ]
        for _, row in part.iterrows():
            name = f"{row['model']}/{row['feature_set']}"
            name += " *" if row["exploratory"] else ""
            if row["reference"] and not math.isnan(row["delta_macro_f1_vs_reference"]):
                delta = (
                    f"{row['delta_macro_f1_vs_reference']:+.3f} "
                    f"[{row['delta_macro_f1_ci_low']:+.3f}, "
                    f"{row['delta_macro_f1_ci_high']:+.3f}]"
                )
            elif row["reference"]:
                delta = "reference"
            else:
                delta = "no reference model"
            lines += [
                f"{name}  (n = {row['n_participants']})",
                f"  macro-F1          {_ci(row, 'macro_f1')}   folds "
                f"{row['macro_f1_fold_mean']:.3f} +/- {row['macro_f1_fold_std']:.3f}",
                f"  accuracy          {_ci(row, 'accuracy')}   majority "
                f"{row['majority_baseline']:.3f}   folds "
                f"{row['accuracy_fold_mean']:.3f} +/- {row['accuracy_fold_std']:.3f}",
                f"  balanced acc.     {_ci(row, 'balanced_accuracy')}",
                f"  precision SAD     {_ci(row, 'precision_sad')}   recall SAD   "
                f"{_ci(row, 'recall_sad')}",
                f"  precision HAPPY   {_ci(row, 'precision_happy')}   recall HAPPY "
                f"{_ci(row, 'recall_happy')}",
                f"  ROC-AUC           {_ci(row, 'roc_auc')}",
                f"  d macro-F1 vs ref {delta}",
                "",
            ]
    if result.skipped:
        lines += ["Skipped (incomplete folds):"] + [f"  {s}" for s in result.skipped]
        lines.append("")
    if missing:
        lines += ["Prediction files not found:"] + [f"  {m}" for m in missing]
        lines.append("")
    return "\n".join(lines)


# ─── figures ──────────────────────────────────────────────────────────────────


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)
    ax.tick_params(colors=_INK_2, labelsize=7)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)
    ax.title.set_color(_INK)


def _grid(n: int) -> tuple:
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False
    )
    fig.patch.set_facecolor(_SURFACE)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def _save(fig, path: Path, cfg=config) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.evaluation.dpi, metadata=_PNG_META)
    plt.close(fig)


def _plot_confusion(result: Evaluation, analysis: str, path: Path, cfg=config):
    rows = result.confusion[result.confusion["analysis"] == analysis]
    fig, axes = _grid(len(rows))
    for ax, (_, r) in zip(axes, rows.iterrows()):
        counts = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap=_SEQUENTIAL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{counts[i, j]}\n{share[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if share[i, j] > 0.5 else _INK,
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=[HAPPY, SAD],
            yticklabels=[HAPPY, SAD],
            xlabel="Predicted",
            ylabel="True",
        )
        ax.set_title(f"{r['model']} / {r['feature_set']}", fontsize=8)
        _style(ax)
    fig.suptitle(f"Confusion matrices, {analysis} set (row %)", color=_INK)
    _save(fig, path, cfg)


def _plot_roc(result: Evaluation, analysis: str, path: Path, cfg=config) -> None:
    keys = [k for k in result.groups if k[0] == analysis]
    fig, axes = _grid(len(keys))
    for ax, key in zip(axes, keys):
        y, _, prob = _arrays(result.groups[key])
        ax.plot([0, 1], [0, 1], color=_MUTED, linestyle="--", linewidth=0.8)
        title = f"{key[1]} / {key[2]}"
        if len(set(y)) == 2:
            fpr, tpr, _ = roc_curve(y, prob)
            ax.plot(fpr, tpr, color=_SERIES, linewidth=1.5)
            title += f"\nAUC {roc_auc_score(y, prob):.3f}"
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - specificity", ylabel="Recall SAD")
        ax.grid(color=_GRID, linewidth=0.5)
        ax.set_title(title, fontsize=8)
        _style(ax)
    fig.suptitle(f"ROC curves, {analysis} set (pooled out-of-fold)", color=_INK)
    _save(fig, path, cfg)


def _plot_reliability(group: pd.DataFrame, key: tuple, path: Path, cfg=config):
    bins = cfg.evaluation.reliability_bins
    y, _, prob = _arrays(group)
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=bins, strategy="uniform")
    counts, edges = np.histogram(prob, bins=np.linspace(0, 1, bins + 1))
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(4.5, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(_SURFACE)
    top.plot([0, 1], [0, 1], color=_MUTED, linestyle="--", linewidth=0.8)
    top.plot(mean_pred, frac_pos, color=_SERIES, linewidth=1.5, marker="o", ms=4)
    top.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed SAD fraction")
    top.grid(color=_GRID, linewidth=0.5)
    top.set_title(
        f"Reliability, {key[1]} / {key[2]} ({key[0]} set, n = {len(group)})",
        fontsize=9,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    bottom.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=_SERIES)
    bottom.set(xlabel="Predicted prob_sad", ylabel="Participants")
    for ax in (top, bottom):
        _style(ax)
    _save(fig, path, cfg)


# ─── run ──────────────────────────────────────────────────────────────────────


def output_dir(subset: str | None = None, cfg=config) -> Path:
    name = cfg.evaluation.output_subdir
    return cfg.paths.results_dir / (name if subset is None else f"{name}_{subset}")


def run(
    analysis: str | None = None, subset: str | None = None, stage: str | None = None
) -> Path:
    """Evaluate every available model; return the model_comparison.csv path.

    analysis None evaluates every EvalConfig.analyses set (primary first).
    """
    require_frozen_or_synthetic(config, "evaluator")
    stage = resolve_stage(stage, subset)
    if subset is not None and subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    ev = config.evaluation
    missing = [
        str(p) for _, _, p in prediction_paths(analysis, subset) if not p.is_file()
    ]
    result = evaluate(load_predictions(analysis, subset))
    if result.comparison.empty:
        raise ValueError("no model covers all folds; nothing to evaluate")

    out = output_dir(subset)
    out.mkdir(parents=True, exist_ok=True)
    fmt = "%.6f"
    path = out / "model_comparison.csv"
    result.comparison.to_csv(path, index=False, float_format=fmt)
    result.per_fold.to_csv(out / "per_fold_metrics.csv", index=False, float_format=fmt)
    result.confusion.to_csv(out / "confusion_matrices.csv", index=False)
    (out / "model_comparison.txt").write_text(
        _text_report(result, missing, subset), encoding="utf-8"
    )

    for name in [a for a in ev.analyses if a in set(result.comparison["analysis"])]:
        _plot_confusion(result, name, out / f"confusion_matrices_{name}.png")
        _plot_roc(result, name, out / f"roc_curves_{name}.png")
        key = (name, ev.reliability_model, ev.reliability_feature_set)
        if key in result.groups:
            _plot_reliability(
                result.groups[key], key, out / f"reliability_{key[1]}_{name}.png"
            )
        else:
            print(f"evaluator: no {key[1]}/{key[2]} for {name}; no reliability curve")

    command = command_line(
        "src.training.evaluator", analysis=analysis, subset=subset, stage=stage
    )
    for _, row in result.comparison.iterrows():
        log_run(
            stage,
            command,
            row["analysis"],
            subset,
            {
                "model": row["model"],
                "feature_set": row["feature_set"],
                "n_participants": int(row["n_participants"]),
                "macro_f1": float(row["macro_f1"]),
                "accuracy": float(row["accuracy"]),
                "majority_baseline": float(row["majority_baseline"]),
                "macro_f1_ci_low": float(row["macro_f1_ci_low"]),
                "macro_f1_ci_high": float(row["macro_f1_ci_high"]),
            },
        )
        print(
            f"{row['analysis']:8s} {row['model'] + '/' + row['feature_set']:28s} "
            f"macro-F1 {_ci(row, 'macro_f1')}  acc {row['accuracy']:.3f} "
            f"(majority {row['majority_baseline']:.3f})"
        )
    print(f"Written: {out}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Participant-level evaluation with bootstrap CIs (Stage H1)."
    )
    parser.add_argument(
        "--analysis",
        choices=config.evaluation.analyses,
        default=None,
        help="Evaluate one analysis set (default: all, primary first).",
    )
    parser.add_argument("--subset", choices=SUBSETS, default=None)
    parser.add_argument("--stage", choices=STAGES, default=None)
    args = parser.parse_args()
    try:
        run(args.analysis, args.subset, args.stage)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
