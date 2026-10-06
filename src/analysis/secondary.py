"""
Pre-registered secondary and exploratory analyses — Stage H3.

EVALUATION_PROTOCOL.md section 6. Nothing here changes, re-selects, or
re-thresholds the primary result: it only reads the runners' predictions and
outputs, and it never writes the H1 model_comparison files.

Sections of results/final/secondary.txt (machine-readable rows in
secondary_metrics.csv):

  1. Per task family: macro-F1 [95% CI] of every model on drawing, word,
     cursive, and the combined set (LR all / concat, CNN *_fused).
  2. Primary vs full sample side by side; middle band: models run with
     --predict-middle, scored on the set-aside middle-band participants
     against their export labels, next to the training-fold majority
     baseline (training participants = the primary-set participants of the
     other folds).
  3. Ablations, each as a paired macro-F1 difference [95% CI] against a
     named reference on the same participant resamples (evaluator):
       backbone      simple CNN head and ResNet18 head vs CNN-HMM (fused)
       aggregation   mean_logit and majority_vote vs mean_prob (protocol
                     section 6.3), per family and fused (equal weights) for
                     the CNN head and the CNN-HMM, from crop_predictions.csv
       ensemble      each component vs the ensemble
       EXPLORATORY   TrOCR handwriting embeddings vs ResNet18 embeddings
  4. Top handcrafted features: lr_handcrafted/all standardized coefficients
     (lr_coefficients.csv) averaged over the outer folds; the
     EvalConfig.top_features largest by |mean|, with fold std, min and max.
     Positive = towards SAD.
  5. QC exclusions: label balance of labeled participants whose
     participants.csv status is not qc_passed vs qc_passed (chi-square with
     Yates' correction), if any were excluded.

PNGs (300 dpi, titled "Secondary analysis: ..." or "Exploratory: ..."):
secondary_family_<analysis>.png, secondary_ablations_<analysis>.png,
secondary_top_features.png.

    python -m src.analysis.secondary
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.data.dataloader import labels_csv_path
from src.training.aggregate import FAMILIES, METHODS, PRIMARY_METHOD
from src.training.aggregate import aggregate_crops, fuse_task_families
from src.training.evaluator import (
    analysis_frame,
    bootstrap_ci,
    evaluate,
    load_predictions,
    majority_hits,
    metrics,
    output_dir,
    paired_diff_ci,
    validate,
)
from src.training.run_baselines import EXPLORATORY_MODELS, PRED_COLUMNS
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic

plt.switch_backend("Agg")

SAD = "SAD"
SECONDARY, EXPLORATORY = "Secondary analysis", "Exploratory"
COMBINED = "combined"
ROW_COLUMNS = (
    "block",
    "analysis",
    "model",
    "feature_set",
    "variant",
    "row_label",
    "exploratory",
    "n_participants",
    "macro_f1",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "accuracy",
    "majority_baseline",
    "reference",
    "delta_macro_f1",
    "delta_ci_low",
    "delta_ci_high",
)
# Row label -> {family or COMBINED: (model, feature_set)} for section 1.
FAMILY_TABLE = {
    "lr_handcrafted": {
        **{f: ("lr_handcrafted", f) for f in FAMILIES},
        COMBINED: ("lr_handcrafted", "all"),
    },
    "lr_embeddings": {
        **{f: ("lr_embeddings", f) for f in FAMILIES},
        COMBINED: ("lr_embeddings", "concat"),
    },
    "cnn_head (ResNet18)": {
        **{f: ("cnn_head", f) for f in FAMILIES},
        COMBINED: ("cnn_head_fused", "fused"),
    },
    "cnn_head (simple CNN)": {
        **{f: ("cnn_head_simple", f) for f in FAMILIES},
        COMBINED: ("cnn_head_simple_fused", "fused"),
    },
    "cnn_hmm": {
        **{f: ("cnn_hmm", f) for f in FAMILIES},
        COMBINED: ("cnn_hmm_fused", "fused"),
    },
}
# (block, reference, compared, exploratory) for section 3 (non-aggregation).
ABLATIONS = (
    ("backbone", ("cnn_hmm_fused", "fused"), ("cnn_head_fused", "fused"), False),
    (
        "backbone",
        ("cnn_hmm_fused", "fused"),
        ("cnn_head_simple_fused", "fused"),
        False,
    ),
    ("ensemble", ("ensemble", "all"), ("lr_handcrafted", "all"), False),
    ("ensemble", ("ensemble", "all"), ("lr_embeddings", "concat"), False),
    ("ensemble", ("ensemble", "all"), ("cnn_hmm_fused", "fused"), False),
    ("embeddings", ("lr_embeddings", "concat"), ("lr_handwriting", "concat"), True),
)
# Crop-level models: (source folder, crop model, fused model, fused families).
CROP_MODELS = (
    ("cnn", "cnn_head", "cnn_head_fused", FAMILIES),
    ("hybrid", "cnn_hmm", "cnn_hmm_fused", tuple(config.hybrid.families)),
)

_SURFACE, _INK, _INK_2, _MUTED, _GRID, _SERIES = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#2a78d6",
)
_PNG_META = {"Software": None}


# ─── loading ──────────────────────────────────────────────────────────────────


def _complete(frame: pd.DataFrame, cfg=config) -> bool:
    return set(frame["fold"]) == set(range(cfg.cv.n_splits))


def load_simple_cnn(cfg=config) -> pd.DataFrame:
    """The simple-CNN ablation (run_cnn --backbone simple), models renamed."""
    frames = []
    for analysis in cfg.evaluation.analyses:
        path = cfg.paths.results_dir / "cnn" / f"{analysis}_simple" / "predictions.csv"
        if path.is_file():
            raw = pd.read_csv(path, dtype={"participant_id": str})
            frames.append(validate(raw, analysis, str(path), cfg))
    if not frames:
        return pd.DataFrame(columns=list(PRED_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].map(
        {"cnn_head": "cnn_head_simple", "cnn_head_fused": "cnn_head_simple_fused"}
    )
    return df


def _crops(source: str, analysis: str, cfg=config) -> pd.DataFrame | None:
    path = cfg.paths.results_dir / source / analysis / "crop_predictions.csv"
    if not path.is_file():
        return None
    crops = pd.read_csv(path, dtype={"participant_id": str})
    crops["in_middle_band"] = crops["in_middle_band"].astype(str) == "True"
    return crops


def aggregate_with(
    crops: pd.DataFrame, method: str, model: str, fused: str, families
) -> pd.DataFrame:
    """Participant predictions of one crop model under `method`, per family
    and fused over `families` with equal weights (shared schema)."""
    per_family = [
        aggregate_crops(part, method).assign(model=model, feature_set=family)
        for (family, _), part in crops[crops["model"] == model].groupby(
            ["task_family", "fold"], sort=False
        )
    ]
    if not per_family:
        return pd.DataFrame(columns=list(PRED_COLUMNS))
    preds = pd.concat(per_family, ignore_index=True)
    fused_rows = []
    for _, part in preds.groupby("fold"):
        part = part[part["feature_set"].isin(families)]
        if set(part["feature_set"]) != set(families):
            continue
        long = part[
            [
                "participant_id",
                "feature_set",
                "prob_sad",
                "fold",
                "label",
                "in_middle_band",
                "analysis",
            ]
        ].rename(columns={"feature_set": "task_family"})
        fused_rows.append(
            fuse_task_families(long.assign(model=fused)).assign(feature_set="fused")
        )
    out = pd.concat([preds] + fused_rows, ignore_index=True)
    return out[list(PRED_COLUMNS)]


# ─── metric rows ──────────────────────────────────────────────────────────────


def _row(block, analysis, frame, key, cfg=config, **extra) -> dict:
    n_boot, seed = cfg.evaluation.n_bootstrap, cfg.training.seed
    m = metrics(frame)
    lo, hi = bootstrap_ci(frame, "macro_f1", n_boot, seed)
    return {
        "block": block,
        "analysis": analysis,
        "model": key[0],
        "feature_set": key[1],
        "variant": extra.pop("variant", ""),
        "exploratory": key[0] in EXPLORATORY_MODELS,
        "n_participants": len(frame),
        "macro_f1": m["macro_f1"],
        "macro_f1_ci_low": lo,
        "macro_f1_ci_high": hi,
        "accuracy": m["accuracy"],
        "majority_baseline": extra.pop(
            "majority", float(majority_hits(frame, cfg).mean())
        ),
        **extra,
    }


def _paired_row(block, analysis, frame, key, ref, ref_key, cfg=config, **extra):
    row = _row(block, analysis, frame, key, cfg, **extra)
    row["reference"] = "/".join(ref_key) + (
        f" ({extra['ref_variant']})" if "ref_variant" in extra else ""
    )
    row.pop("ref_variant", None)
    try:
        d, lo, hi = paired_diff_ci(
            frame,
            ref,
            "macro_f1",
            cfg.evaluation.n_bootstrap,
            cfg.training.seed,
        )
    except ValueError as exc:
        row["note"] = str(exc)
        d = lo = hi = math.nan
    row.update(delta_macro_f1=d, delta_ci_low=lo, delta_ci_high=hi)
    return row


def _groups(df: pd.DataFrame, analysis: str, cfg=config) -> dict:
    frame = analysis_frame(df, analysis)
    return {
        k: g
        for k, g in frame.groupby(["model", "feature_set"], sort=False)
        if _complete(g, cfg)
    }


def family_rows(df: pd.DataFrame, cfg=config) -> list:
    rows = []
    for analysis in [a for a in cfg.evaluation.analyses if a in set(df["analysis"])]:
        groups = _groups(df, analysis, cfg)
        for label, cells in FAMILY_TABLE.items():
            for family, key in cells.items():
                if key in groups:
                    rows.append(
                        _row("family", analysis, groups[key], key, cfg, variant=family)
                    )
                    rows[-1]["row_label"] = label
    return rows


def middle_band_rows(df: pd.DataFrame, cfg=config) -> list:
    """Middle-band participants (primary run, --predict-middle)."""
    rows = []
    prim = df[df["analysis"] == "primary"]
    thr = cfg.aggregate.threshold
    for key, g in prim.groupby(["model", "feature_set"], sort=False):
        mid, test = g[g["in_middle_band"]], g[~g["in_middle_band"]]
        if mid.empty or not _complete(test, cfg):
            continue
        hits = []
        for fold, part in mid.groupby("fold"):
            share = float((test.loc[test["fold"] != fold, "label"] == SAD).mean())
            hits.append(part["label"] == (SAD if share >= thr else "HAPPY"))
        majority = float(pd.concat(hits).mean())
        rows.append(_row("middle_band", "primary", mid, key, cfg, majority=majority))
    return rows


def ablation_rows(df: pd.DataFrame, cfg=config) -> list:
    rows = []
    for analysis in [a for a in cfg.evaluation.analyses if a in set(df["analysis"])]:
        groups = _groups(df, analysis, cfg)
        for block, ref_key, key, _ in ABLATIONS:
            if key in groups and ref_key in groups:
                rows.append(
                    _paired_row(
                        block, analysis, groups[key], key, groups[ref_key], ref_key, cfg
                    )
                )
        rows += aggregation_rows(analysis, cfg)
    return rows


def aggregation_rows(analysis: str, cfg=config) -> list:
    """mean_logit / majority_vote vs mean_prob from the same crop predictions."""
    rows = []
    for source, model, fused, families in CROP_MODELS:
        crops = _crops(source, analysis, cfg)
        if crops is None:
            continue
        by_method = {}
        for method in METHODS:
            preds = aggregate_with(crops, method, model, fused, families)
            by_method[method] = _groups(preds.assign(analysis=analysis), analysis, cfg)
        for key, ref in by_method[PRIMARY_METHOD].items():
            for method in METHODS:
                if method == PRIMARY_METHOD or key not in by_method[method]:
                    continue
                rows.append(
                    _paired_row(
                        "aggregation",
                        analysis,
                        by_method[method][key],
                        key,
                        ref,
                        key,
                        cfg,
                        variant=method,
                        ref_variant=PRIMARY_METHOD,
                    )
                )
    return rows


def top_features(cfg=config) -> pd.DataFrame | None:
    path = cfg.paths.results_dir / "baselines" / "primary" / "lr_coefficients.csv"
    if not path.is_file():
        return None
    coefs = pd.read_csv(path)
    coefs = coefs[
        (coefs["model"] == "lr_handcrafted") & (coefs["feature_set"] == "all")
    ]
    stats = coefs.groupby("feature")["coef_std"].agg(
        mean="mean", std="std", min="min", max="max", n_folds="count"
    )
    order = stats["mean"].abs().sort_values(ascending=False, kind="mergesort")
    return stats.loc[order.index[: cfg.evaluation.top_features]].reset_index()


def qc_balance(cfg=config) -> dict | None:
    """Label x QC status (qc_passed vs anything else) among labeled people."""
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    participants = pd.read_csv(
        cfg.paths.metadata_dir / "participants.csv", dtype={"participant_id": str}
    )
    merged = labels.merge(participants[["participant_id", "status"]], how="left")
    merged["status"] = merged["status"].fillna("missing")
    merged["excluded"] = merged["status"] != "qc_passed"
    if not merged["excluded"].any():
        return None
    table = pd.crosstab(merged["excluded"], merged[cfg.labeling.label_column])
    chi2, p, dof, expected = chi2_contingency(table.to_numpy(), correction=True)
    return {
        "table": table.rename(index={False: "qc_passed", True: "excluded"}),
        "statuses": merged.loc[merged["excluded"], "status"].value_counts().to_dict(),
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "min_expected": float(expected.min()),
    }


# ─── figures ──────────────────────────────────────────────────────────────────


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)
    ax.tick_params(colors=_INK_2, labelsize=7)
    ax.grid(axis="x", color=_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig, path: Path, title: str, cfg=config) -> None:
    fig.suptitle(title, color=_INK, fontsize=10)
    fig.patch.set_facecolor(_SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.evaluation.dpi, metadata=_PNG_META)
    plt.close(fig)


def _interval_panel(ax, rows: pd.DataFrame, value, lo, hi, labels) -> None:
    y = np.arange(len(rows))[::-1]
    mid, low, high = (rows[c].to_numpy(dtype=float) for c in (value, lo, hi))
    ax.errorbar(
        mid,
        y,
        xerr=np.vstack([mid - low, high - mid]),
        fmt="o",
        color=_SERIES,
        ecolor=_SERIES,
        elinewidth=1.2,
        markersize=4,
        capsize=0,
    )
    ax.set_yticks(y, labels, fontsize=7)
    _style(ax)


def plot_family(rows: pd.DataFrame, analysis: str, path: Path, cfg=config) -> None:
    rows = rows[(rows["block"] == "family") & (rows["analysis"] == analysis)]
    panels = [f for f in list(FAMILIES) + [COMBINED] if f in set(rows["variant"])]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.0 * len(panels), 3.2), sharey=True, squeeze=False
    )
    labels = [k for k in FAMILY_TABLE if k in set(rows["row_label"])]
    for ax, family in zip(axes[0], panels):
        part = (
            rows[rows["variant"] == family]
            .set_index("row_label")
            .reindex(labels)
            .reset_index()
        )
        _interval_panel(
            ax, part, "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", labels
        )
        ax.set(xlim=(0, 1), xlabel="Macro-F1 [95% CI]")
        ax.set_title(family, fontsize=8, color=_INK)
    _save(
        fig,
        path,
        f"{SECONDARY}: macro-F1 per task family ({analysis} set)",
        cfg,
    )


def plot_ablations(rows: pd.DataFrame, analysis: str, path: Path, cfg=config):
    rows = rows[
        (rows["block"].isin(["backbone", "aggregation", "ensemble"]))
        & (rows["analysis"] == analysis)
        & rows["delta_macro_f1"].notna()
    ]
    if rows.empty:
        return
    labels = [
        f"{r.block}: {r.model}/{r.feature_set}"
        + (f" {r.variant}" if r.variant else "")
        + f" - {r.reference}"
        for r in rows.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(rows) + 1.4))
    ax.axvline(0, color=_MUTED, linewidth=0.8, linestyle="--")
    _interval_panel(ax, rows, "delta_macro_f1", "delta_ci_low", "delta_ci_high", labels)
    ax.set(xlabel="Macro-F1 difference vs reference [95% CI]")
    _save(fig, path, f"{SECONDARY}: ablations ({analysis} set)", cfg)


def plot_top_features(top: pd.DataFrame, path: Path, cfg=config) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(top) + 1.4))
    y = np.arange(len(top))[::-1]
    mean = top["mean"].to_numpy(dtype=float)
    spread = top["std"].fillna(0).to_numpy(dtype=float)
    ax.barh(y, mean, color=_SERIES, height=0.7)
    ax.errorbar(mean, y, xerr=spread, fmt="none", ecolor=_INK, lw=0.8)
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_yticks(y, top["feature"], fontsize=7)
    ax.set(xlabel="Mean standardized coefficient (+/- fold std); positive = SAD")
    _style(ax)
    _save(
        fig,
        path,
        f"{SECONDARY}: top {len(top)} handcrafted features (lr_handcrafted/all)",
        cfg,
    )


# ─── text ─────────────────────────────────────────────────────────────────────


def _fmt(r) -> str:
    return (
        f"{r['macro_f1']:.3f} [{r['macro_f1_ci_low']:.3f}, "
        f"{r['macro_f1_ci_high']:.3f}]"
    )


def _delta(r) -> str:
    if math.isnan(r["delta_macro_f1"]):
        return f"not paired ({r.get('note', '')})"
    return (
        f"{r['delta_macro_f1']:+.3f} [{r['delta_ci_low']:+.3f}, "
        f"{r['delta_ci_high']:+.3f}] vs {r['reference']}"
    )


def text_report(rows: pd.DataFrame, comparison, top, qc, cfg=config) -> str:
    ev = cfg.evaluation
    out = [
        "EMHA secondary analyses (EVALUATION_PROTOCOL.md section 6)",
        "Every result below is SECONDARY (or EXPLORATORY where marked) and is "
        "reported without multiplicity correction. The primary claim is in "
        "model_comparison.txt.",
        f"CIs: {ev.n_bootstrap} participant bootstrap resamples, seed "
        f"{cfg.training.seed}, {round(ev.ci_level * 100)}% percentile.",
        "",
        "1. Macro-F1 per task family [95% CI]",
    ]
    fam = rows[rows["block"] == "family"]
    for analysis in [a for a in ev.analyses if a in set(fam["analysis"])]:
        out.append(f"  -- {analysis} --")
        part = fam[fam["analysis"] == analysis]
        for label in [k for k in FAMILY_TABLE if k in set(part["row_label"])]:
            cells = part[part["row_label"] == label].set_index("variant")
            text = "  ".join(
                f"{f} {_fmt(cells.loc[f])}"
                for f in list(FAMILIES) + [COMBINED]
                if f in cells.index
            )
            out.append(f"  {label:22s} {text}")
    out += ["", "2a. Primary vs full sample (macro-F1 [CI]; accuracy / majority)"]
    if comparison is not None and not comparison.empty:
        keys = comparison[["model", "feature_set"]].drop_duplicates()
        for model, fs in keys.itertuples(index=False):
            cells = []
            for analysis in ev.analyses:
                c = comparison[
                    (comparison["model"] == model)
                    & (comparison["feature_set"] == fs)
                    & (comparison["analysis"] == analysis)
                ]
                if c.empty:
                    cells.append(f"{analysis}: not run")
                    continue
                r = c.iloc[0]
                cells.append(
                    f"{analysis} (n={r['n_participants']}): {_fmt(r)}; "
                    f"acc {r['accuracy']:.3f} / {r['majority_baseline']:.3f}"
                )
            out.append(f"  {model + '/' + fs:28s} " + "   ".join(cells))
    mid = rows[rows["block"] == "middle_band"]
    out += ["", "2b. Middle band (set aside; --predict-middle runs only)"]
    if mid.empty:
        out.append("  no model was run with --predict-middle")
    for _, r in mid.iterrows():
        out.append(
            f"  {r['model'] + '/' + r['feature_set']:28s} n={r['n_participants']}  "
            f"macro-F1 {_fmt(r)}  accuracy {r['accuracy']:.3f}  majority "
            f"{r['majority_baseline']:.3f}"
        )
    out += ["", "3. Ablations (paired macro-F1 difference [95% CI])"]
    abl = rows[rows["block"].isin(["backbone", "aggregation", "ensemble"])]
    for _, r in abl.iterrows():
        variant = f" [{r['variant']}]" if r["variant"] else ""
        out.append(
            f"  {r['analysis']:8s} {r['block']:12s}"
            f"{r['model'] + '/' + r['feature_set'] + variant:40s}"
            f"{_fmt(r)}  d {_delta(r)}"
        )
    out += ["", "3e. EXPLORATORY: handwriting-encoder (TrOCR) embeddings"]
    exp = rows[rows["block"] == "embeddings"]
    if exp.empty:
        out.append("  not run (no handwriting embeddings)")
    for _, r in exp.iterrows():
        out.append(
            f"  {r['analysis']:8s} {r['model'] + '/' + r['feature_set']:28s}"
            f"{_fmt(r)}  d {_delta(r)}  -- exploratory, never confirmatory"
        )
    out += ["", f"4. Top {ev.top_features} handcrafted features, lr_handcrafted/all"]
    if top is None:
        out.append("  lr_coefficients.csv not found")
    else:
        out.append(
            "  feature                              mean     std     min     max"
        )
        for _, r in top.iterrows():
            out.append(
                f"  {r['feature']:34s} {r['mean']:+.3f}  {r['std']:.3f}  "
                f"{r['min']:+.3f}  {r['max']:+.3f}  ({r['n_folds']} folds)"
            )
    out += ["", "5. QC exclusions and label balance"]
    if qc is None:
        out.append("  no labeled participant was excluded at QC; test not needed")
    else:
        out.append(f"  excluded statuses: {qc['statuses']}")
        for line in qc["table"].to_string().splitlines():
            out.append(f"    {line}")
        out.append(
            f"  chi-square (Yates) = {qc['chi2']:.3f}, dof {qc['dof']}, "
            f"p = {qc['p']:.4f}; smallest expected count {qc['min_expected']:.1f}"
            + (" (< 5: interpret with care)" if qc["min_expected"] < 5 else "")
        )
    return "\n".join(out) + "\n"


# ─── run ──────────────────────────────────────────────────────────────────────


def run() -> Path:
    """Write secondary.txt, secondary_metrics.csv and PNGs; return the txt path."""
    require_frozen_or_synthetic(config, "secondary")
    preds = load_predictions()
    simple = load_simple_cnn()
    df = pd.concat([preds, simple], ignore_index=True) if len(simple) else preds
    comparison = evaluate(preds).comparison
    rows = pd.DataFrame(family_rows(df) + middle_band_rows(df) + ablation_rows(df))
    # every column present even when a block (e.g. paired ablations) is empty
    rows = rows.reindex(columns=list(dict.fromkeys([*ROW_COLUMNS, *rows.columns])))
    top = top_features()
    qc = qc_balance()

    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "secondary_metrics.csv", index=False, float_format="%.6f")
    path = out / "secondary.txt"
    path.write_text(text_report(rows, comparison, top, qc), encoding="utf-8")
    if not rows.empty:
        for analysis in sorted(set(rows["analysis"])):
            if ((rows["block"] == "family") & (rows["analysis"] == analysis)).any():
                plot_family(rows, analysis, out / f"secondary_family_{analysis}.png")
            plot_ablations(rows, analysis, out / f"secondary_ablations_{analysis}.png")
    if top is not None and not top.empty:
        plot_top_features(top, out / "secondary_top_features.png")
    print(f"Written: {path}")
    return path


def main() -> int:
    argparse.ArgumentParser(description="Secondary analyses (Stage H3).").parse_args()
    try:
        run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
