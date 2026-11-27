"""
Signal boost v5 -- EXPLORATORY, post hoc (EVALUATION_PROTOCOL.md section 9,
Deviation 8). Never part of the primary claim; the primary model stays
CNN-HMM v1 (run_hybrid.py). v1-v4 outputs are only read.

Idea: v1-v4 showed that only frozen features + a linear model rose above
chance. v5 keeps that model family and gives it more information:

  C1  richer frozen, label-free features (src/features/embeddings_v5.py):
      style (ResNet18 layer1-3 channel mean/std) and dinov2 (DINOv2 CLS +
      mean patch token), next to the v1 resnet18 cache. Each feature set is
      the per-family participant mean, concatenated over the three families
      (as lr_embeddings/concat).
  C2  ridge_v5 predicts total_score instead of the binary label.
  C3  ridge_v5 trains on every outer-train participant, middle band
      included. Middle-band participants are never in a headline metric.

Per outer fold (folds.csv; train/test from fold_ids; assert_no_leakage):
  lr_v5     run_baselines.fit_lr(pca=True) on the primary outer-train
            participants, binary label.
  ridge_v5  StandardScaler -> PCA (fit_lr's component rule) -> Ridge on all
            outer-train participants. alpha by inner StratifiedKFold (on the
            TRUE labels, also in permutation draws), mean squared error.
            The chosen alpha's inner out-of-fold scores train a balanced 1-D
            logistic calibrator (score -> P(SAD)); the final ridge is refit
            on all outer-train participants. Scaler and PCA never see a
            label, so their per-split fits are cached and reused by every
            permutation draw (identical numbers, much faster).
  Both predict every outer-test participant; middle-band rows carry
  in_middle_band = True. A participant is SAD when P(SAD) >= 0.5.

Headlines, fixed in Deviation 8 before any v5 result, both ridge_v5/all_v5
on the same out-of-fold predictions:
  H1  the primary set (as v1-v4);
  H2  the strict group: total_score at or below the strict_quantile and at
      or above 1 - strict_quantile of all labeled participants (labels.csv
      only); evaluated as feature_set "<fs>@strict".
Permutation test: n_permutations draws, (total_score, label) pairs shuffled
together among outer-train participants with permutation_seed(seed, k,
fold), test labels true; statistic macro-F1 (accuracy p reported too); Holm
over H1 and H2. The extremity curve (boundary_distance >= d) is descriptive.

Outputs (v5 paths only):

    results/signal_v5/primary/run_config.json   a rerun refuses a mismatch
    results/signal_v5/primary/predictions.csv, metrics.csv, per_fold.csv,
        confusion.csv, paired.csv, extremity_curve.csv, permutation.json
    results/final/permutation_runs/primary_v5/seed_<s>/perm_<k>.csv
    results/final/EXPLORATORY_v5.txt, fig_v5_extremity.png,
        fig_v5_permutation_null_h1.png, fig_v5_permutation_null_h2.png

    python -m src.training.run_signal_v5 --step preflight
    python -m src.training.run_signal_v5 --step fit
    python -m src.training.run_signal_v5 --step evaluate
    python -m src.training.run_signal_v5 --step curve
    python -m src.training.run_signal_v5 --step permutation [--n-jobs -1]
    python -m src.training.run_signal_v5 --step report

All steps are CPU only (the feature caches come from embeddings_v5).
Nothing is appended to results/RUN_LOG.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.analysis.permutation import _plot, p_value, permutation_seed
from src.data.dataloader import labels_csv_path
from src.features import embeddings_v5
from src.features.embeddings import FAMILIES
from src.training import evaluator
from src.training import run_per_task_v3 as v3
from src.training.run_baselines import (
    PRED_COLUMNS,
    _participant_means,
    _prob_sad,
    _rows,
    fit_lr,
)
from src.training.splits import analysis_ids, assert_no_leakage, fold_ids, load_folds
from src.utils.config import config
from src.utils.protocol_guard import is_synthetic_root, require_frozen_or_synthetic

STEPS = ("preflight", "fit", "evaluate", "curve", "permutation", "report")
SCORE_COLUMN = "total_score"
DISTANCE_COLUMN = "boundary_distance"
PERMUTATION_KEYS = ("n_permutations",)
NO_REFERENCE = "reference not found"
SAD, HAPPY = "SAD", "HAPPY"
REPORT_HEADER = (
    "EXPLORATORY, POST HOC (Deviation 8). Not part of the primary claim; the "
    "primary model remains CNN-HMM v1 (cnn_hmm_fused/fused)."
)
LIMITATIONS = (
    "Limitations. v5 is post hoc (Deviation 8), decided after v1-v4 results "
    "were known; it is the fifth model version on the same participants. "
    "About 256 primary training participants per fold (about 320 with the "
    "middle band, which ridge_v5 uses for training only). H2 keeps only the "
    "lowest and highest quarter of self-reported scores, so it shows how "
    "well clearly happy and clearly sad respondents are separated, not how "
    "the model does across the whole range. The extremity curve is "
    "descriptive and its subsets shrink as d grows; no point of it is a "
    "headline. The label is a concurrent self-report, not a diagnosis."
)
CURVE_COLORS = ("#2a78d6", "#eb6834")
CURVE_MARKERS = ("o", "s")


# ---------------------------------------------------------------- paths


def output_dir(cfg=config) -> Path:
    v5 = cfg.signal_v5
    return cfg.paths.results_dir / v5.output_subdir / v5.analysis


def runs_dir(seed: int, cfg=config) -> Path:
    out = evaluator.output_dir(None, cfg) / cfg.permutation.runs_subdir
    return out / cfg.signal_v5.runs_name / f"seed_{seed}"


def draw_path(perm: int, seed: int, cfg=config) -> Path:
    return runs_dir(seed, cfg) / f"perm_{perm:04d}.csv"


def report_path(cfg=config) -> Path:
    return evaluator.output_dir(None, cfg) / cfg.signal_v5.report_name


def figure_path(name: str, cfg=config) -> Path:
    return evaluator.output_dir(None, cfg) / f"{cfg.signal_v5.figure_prefix}{name}"


def block_path(block: str, family: str, cfg=config) -> Path:
    if block == "resnet18":
        out = cfg.paths.results_dir / cfg.embedding.output_subdir
        return out / f"resnet18_{family}.npz"
    return embeddings_v5.output_path(block, family, cfg)


def strict_set(fs: str, cfg=config) -> str:
    return f"{fs}@{cfg.signal_v5.strict_name}"


def curve_set(fs: str, d: float) -> str:
    return f"{fs}@d{d:g}"


def headlines(cfg=config) -> dict:
    model, fs = cfg.signal_v5.headline
    return {"H1": (model, fs), "H2": (model, strict_set(fs, cfg))}


def v5_models(cfg=config) -> tuple:
    return tuple(cfg.signal_v5.models)


# ---------------------------------------------------------------- inputs


def check_config(cfg=config) -> None:
    v5 = cfg.signal_v5
    names = [name for name, _ in v5.feature_sets]
    if len(set(names)) != len(names):
        raise ValueError("SignalV5Config.feature_sets names must be unique")
    for name, blocks in v5.feature_sets:
        unknown = set(blocks) - set(v5.blocks)
        if unknown:
            raise ValueError(f"feature set {name}: unknown blocks {sorted(unknown)}")
    model, fs = v5.headline
    if model not in v5.models or fs not in names:
        raise ValueError(f"headline {model}/{fs} is not a v5 row")
    for model, fs in v5.curve_rows:
        if model not in v5.models or fs not in names:
            raise ValueError(f"curve row {model}/{fs} is not a v5 row")
    if not 0 < v5.strict_quantile < 0.5:
        raise ValueError("strict_quantile must be in (0, 0.5)")
    ds = list(v5.extremity_distances)
    if sorted(ds) != ds or ds[0] < 0:
        raise ValueError("extremity_distances must be increasing and >= 0")


@dataclass
class Context:
    labels: pd.DataFrame
    primary: list
    full: list
    folds: pd.DataFrame

    def column(self, name: str, ids) -> np.ndarray:
        return self.labels.loc[list(ids), name].to_numpy()

    def is_sad(self, ids, cfg=config) -> np.ndarray:
        return self.column(cfg.labeling.label_column, ids) == SAD

    def middle(self) -> set:
        return set(self.full) - set(self.primary)


def context(cfg=config) -> Context:
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    participants = pd.read_csv(
        cfg.paths.metadata_dir / "participants.csv", dtype={"participant_id": str}
    )
    primary = analysis_ids(labels, participants, "primary")
    full = analysis_ids(labels, participants, "full")
    return Context(labels.set_index("participant_id"), primary, full, load_folds(cfg))


def strict_groups(ctx: Context, cfg=config) -> tuple:
    """(low ids, high ids) of H2, from labels.csv total_score only."""
    q = cfg.signal_v5.strict_quantile
    scores = pd.Series(ctx.column(SCORE_COLUMN, ctx.full), index=ctx.full)
    lo = np.quantile(scores.to_numpy(), q, method="lower")
    hi = np.quantile(scores.to_numpy(), 1 - q, method="higher")
    low = sorted(scores.index[scores <= lo])
    high = sorted(scores.index[scores >= hi])
    label = ctx.labels[cfg.labeling.label_column]
    if (label.loc[low] != SAD).any() or (label.loc[high] != HAPPY).any():
        raise ValueError(
            "strict group crosses the label cutoff: every low participant must "
            "be SAD and every high participant HAPPY"
        )
    if set(low + high) & ctx.middle():
        raise ValueError("strict group contains middle-band participants")
    return low, high


def feature_table(fs: str, ctx: Context, cfg=config) -> pd.DataFrame:
    """Participants x features of one feature set (per-family means)."""
    blocks = dict(cfg.signal_v5.feature_sets)[fs]
    parts = []
    for block in blocks:
        for family in FAMILIES:
            path = block_path(block, family, cfg)
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found; run python -m src.features.embeddings_v5"
                    if block != "resnet18"
                    else f"{path} not found; run python -m src.features.embeddings"
                )
            parts.append(_participant_means(path).add_prefix(f"{block}__{family}__"))
    table = pd.concat(parts, axis=1, join="inner")
    missing = sorted(set(ctx.full) - set(table.index))
    if missing:
        raise ValueError(
            f"{fs}: no features for {len(missing)} participants, e.g. {missing[:5]}"
        )
    table = table.loc[ctx.full]
    if table.isna().any().any():
        raise ValueError(f"{fs}: NaN features")
    return table


def _crop_counts(path: Path) -> pd.Series:
    with np.load(path, allow_pickle=False) as z:
        pids = pd.Series(z["participant_id"].astype(str))
        if not np.isfinite(z["embeddings"]).all():
            raise ValueError(f"{path}: non-finite values")
    return pids.value_counts()


def settings(cfg=config) -> dict:
    v5 = asdict(cfg.signal_v5)
    for key in PERMUTATION_KEYS:
        v5.pop(key)
    return json.loads(
        json.dumps(
            {
                "signal_v5": v5,
                "baselines": asdict(cfg.baselines),
                "seed": cfg.training.seed,
                "n_splits": cfg.cv.n_splits,
                "threshold": cfg.aggregate.threshold,
                "folds_sha256": v3._sha256(v3.folds_path(cfg)),
                "labels_sha256": v3._sha256(labels_csv_path(cfg)),
                "features_sha256": {
                    f"{b}_{f}": v3._sha256(block_path(b, f, cfg))
                    for b in cfg.signal_v5.blocks
                    for f in FAMILIES
                },
            },
            default=str,
        )
    )


def check_settings(path: Path, want: dict) -> None:
    """Write want to path, or refuse if path already records something else."""
    path = Path(path)
    if path.is_file():
        have = json.loads(path.read_text(encoding="utf-8"))
        if have != want:
            changed = sorted(
                k for k in set(have) | set(want) if have.get(k) != want.get(k)
            )
            raise RuntimeError(
                f"{path} records different settings ({changed}); v5 runs once "
                "(Deviation 8). Log an addendum under Deviation 8 before any "
                "rerun; never delete the existing v5 outputs."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(want, indent=2), encoding="utf-8")


def preflight(cfg=config) -> dict:
    """Check every v5 input before anything is fit; print and return a summary."""
    require_frozen_or_synthetic(cfg, "run_signal_v5")
    check_config(cfg)
    hashes = v3._check_hashes(cfg)
    ctx = context(cfg)
    coverage = {}
    for block in cfg.signal_v5.blocks:
        for family in FAMILIES:
            path = block_path(block, family, cfg)
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
            counts = _crop_counts(path)
            missing = sorted(set(ctx.full) - set(counts.index))
            if missing:
                raise ValueError(
                    f"{path.name}: no crops for {len(missing)} participants, "
                    f"e.g. {missing[:5]}"
                )
            per = counts.loc[ctx.full]
            if per.nunique() != 1:
                raise ValueError(f"{path.name}: crop count differs between people")
            coverage[path.name] = f"{len(ctx.full)} x {int(per.iloc[0])} crops"
    for fs, _ in cfg.signal_v5.feature_sets:
        feature_table(fs, ctx, cfg)
    low, high = strict_groups(ctx, cfg)
    sad = ctx.is_sad(ctx.primary, cfg)
    summary = {
        "analysis": cfg.signal_v5.analysis,
        "synthetic_root": is_synthetic_root(cfg.paths.data_root),
        **hashes,
        "n_primary": len(ctx.primary),
        "n_happy": int((~sad).sum()),
        "n_sad": int(sad.sum()),
        "n_middle_band": len(ctx.middle()),
        "n_full": len(ctx.full),
        "strict_low_sad": len(low),
        "strict_high_happy": len(high),
        "strict_score_ranges": {
            "low": [
                float(ctx.column(SCORE_COLUMN, low).min()),
                float(ctx.column(SCORE_COLUMN, low).max()),
            ],
            "high": [
                float(ctx.column(SCORE_COLUMN, high).min()),
                float(ctx.column(SCORE_COLUMN, high).max()),
            ],
        },
        "features": coverage,
        "headlines": {k: "/".join(v) for k, v in headlines(cfg).items()},
        "n_permutations": cfg.signal_v5.n_permutations,
    }
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- folds


def fold_sets(fold: int, ctx: Context) -> dict:
    """Outer-fold participant sets; train and test never overlap."""
    train_p, test_p = fold_ids(ctx.folds, fold, ctx.primary)
    train_f, test_f = fold_ids(ctx.folds, fold, ctx.full)
    assert_no_leakage(train_f, test_f)
    assert_no_leakage(train_p, test_f)
    if not set(train_p) <= set(train_f) or not set(test_p) <= set(test_f):
        raise ValueError(f"fold {fold}: primary ids are not inside the full set")
    return {
        "train_primary": train_p,
        "train_full": train_f,
        "test_primary": test_p,
        "test_full": test_f,
    }


def n_components(n_rows: int, n_inner: int, n_features: int, cfg=config) -> int:
    """fit_lr's PCA rule: fits every inner-train split."""
    smallest_inner_train = n_rows - math.ceil(n_rows / n_inner) - 1
    return min(cfg.baselines.pca_components, smallest_inner_train - 1, n_features)


def _project(X_fit: np.ndarray, X_apply: list, n_comp: int, seed: int, cfg=config):
    scaler = StandardScaler().fit(X_fit)
    pca = PCA(
        n_components=n_comp, svd_solver=cfg.baselines.pca_svd_solver, random_state=seed
    ).fit(scaler.transform(X_fit))
    return [pca.transform(scaler.transform(X)) for X in (X_fit, *X_apply)]


@dataclass
class RidgeFold:
    """Label-free projections of one outer fold (reused by every draw)."""

    train_ids: list
    test_ids: list
    splits: list
    inner: list
    Z_train: np.ndarray
    Z_test: np.ndarray


def ridge_fold(X: pd.DataFrame, sets: dict, ctx: Context, fold: int, cfg=config):
    """Inner splits stratified on the TRUE labels; scaler/PCA per split."""
    train, test = sets["train_full"], sets["test_full"]
    y = ctx.is_sad(train, cfg).astype(int)
    seed = cfg.training.seed + fold
    n_inner = min(cfg.signal_v5.inner_cv, int(np.bincount(y, minlength=2).min()))
    if n_inner < 2:
        raise ValueError("each class needs >= 2 training participants for inner CV")
    Xt = X.loc[train].to_numpy(dtype=float)
    n_comp = n_components(len(train), n_inner, Xt.shape[1], cfg)
    cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    splits = list(cv.split(Xt, y))
    inner = []
    for tr, va in splits:
        z_tr, z_va = _project(Xt[tr], [Xt[va]], n_comp, seed, cfg)
        inner.append((z_tr, z_va))
    z_train, z_test = _project(
        Xt, [X.loc[test].to_numpy(dtype=float)], n_comp, seed, cfg
    )
    return RidgeFold(list(train), list(test), splits, inner, z_train, z_test)


def fit_ridge(rf: RidgeFold, score: np.ndarray, sad: np.ndarray, cfg=config) -> dict:
    """alpha by inner MSE, calibrator on inner OOF scores, final refit."""
    v5 = cfg.signal_v5
    best = None
    for alpha in v5.ridge_alphas:
        oof = np.empty(len(score), dtype=float)
        for (tr, va), (z_tr, z_va) in zip(rf.splits, rf.inner):
            oof[va] = Ridge(alpha=alpha).fit(z_tr, score[tr]).predict(z_va)
        mse = float(np.mean((oof - score) ** 2))
        if best is None or mse < best[1]:
            best = (alpha, mse, oof)
    alpha, mse, oof = best
    center, scale = float(oof.mean()), float(oof.std()) or 1.0
    calibrator = LogisticRegression(
        C=v5.calibrator_C, class_weight="balanced", max_iter=cfg.baselines.max_iter
    ).fit(((oof - center) / scale)[:, None], sad.astype(int))
    ridge = Ridge(alpha=alpha).fit(rf.Z_train, score)
    test_score = ridge.predict(rf.Z_test)
    col = list(calibrator.classes_).index(1)
    prob = calibrator.predict_proba(((test_score - center) / scale)[:, None])[:, col]
    return {"alpha": alpha, "inner_mse": mse, "test_score": test_score, "prob": prob}


def _pred_rows(model, fs, fold, ids, prob, ctx: Context, cfg=config) -> pd.DataFrame:
    labels = ctx.labels[cfg.labeling.label_column]
    return _rows(
        cfg.signal_v5.analysis, model, fs, fold, list(ids), labels, prob, ctx.middle()
    )[list(PRED_COLUMNS)]


def lr_fold(X: pd.DataFrame, sets: dict, ctx: Context, fold: int, cfg=config):
    train, test = sets["train_primary"], sets["test_full"]
    y = ctx.is_sad(train, cfg).astype(int)
    pipe = fit_lr(X.loc[train], y, cfg, pca=True, seed=cfg.training.seed + fold)
    return _prob_sad(pipe, X.loc[test])


# ---------------------------------------------------------------- steps


def _start(cfg=config) -> Path:
    require_frozen_or_synthetic(cfg, "run_signal_v5")
    check_config(cfg)
    out = output_dir(cfg)
    check_settings(out / "run_config.json", settings(cfg))
    return out


def _sorted_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["model", "feature_set", "fold", "participant_id"], kind="mergesort"
    ).reset_index(drop=True)


def step_fit(cfg=config) -> Path:
    """predictions.csv for every (model, feature set, fold); no metric printed."""
    out = _start(cfg)
    ctx = context(cfg)
    frames, chosen = [], []
    for fs, _ in cfg.signal_v5.feature_sets:
        X = feature_table(fs, ctx, cfg)
        for fold in range(cfg.cv.n_splits):
            began = time.perf_counter()
            sets = fold_sets(fold, ctx)
            test = sets["test_full"]
            if "lr_v5" in cfg.signal_v5.models:
                prob = lr_fold(X, sets, ctx, fold, cfg)
                frames.append(_pred_rows("lr_v5", fs, fold, test, prob, ctx, cfg))
            if "ridge_v5" in cfg.signal_v5.models:
                rf = ridge_fold(X, sets, ctx, fold, cfg)
                score = ctx.column(SCORE_COLUMN, rf.train_ids).astype(float)
                fit = fit_ridge(rf, score, ctx.is_sad(rf.train_ids, cfg), cfg)
                frames.append(
                    _pred_rows("ridge_v5", fs, fold, test, fit["prob"], ctx, cfg)
                )
                chosen.append({"feature_set": fs, "fold": fold, "alpha": fit["alpha"]})
            print(
                f"fit {fs} fold {fold}: {time.perf_counter() - began:.1f} s",
                flush=True,
            )
    preds = _sorted_rows(pd.concat(frames, ignore_index=True))
    path = out / "predictions.csv"
    v3.write_part(preds, path)
    v3.write_part(pd.DataFrame(chosen), out / "ridge_alpha.csv")
    evaluator.validate(
        pd.read_csv(path, dtype={"participant_id": str}),
        cfg.signal_v5.analysis,
        str(path),
        cfg,
    )
    counts = preds.groupby(["model", "feature_set"], sort=False).size()
    print(f"fit: {len(counts)} groups, {len(preds)} rows -> {path}")
    return path


def _read_preds(cfg=config) -> pd.DataFrame:
    path = output_dir(cfg) / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run --step fit first")
    return evaluator.validate(
        pd.read_csv(path, dtype={"participant_id": str}),
        cfg.signal_v5.analysis,
        str(path),
        cfg,
    )


def reference_frames(cfg=config) -> dict:
    """(model, feature_set) -> v1 primary rows (read only)."""
    out, cache = {}, {}
    analysis = cfg.signal_v5.analysis
    for model, fs in cfg.signal_v5.paired_references:
        source = v3.REFERENCE_SOURCES[model]
        path = cfg.paths.results_dir / source / analysis / "predictions.csv"
        if source not in cache:
            cache[source] = (
                evaluator.analysis_frame(
                    evaluator.validate(
                        pd.read_csv(path, dtype={"participant_id": str}),
                        analysis,
                        str(path),
                        cfg,
                    ),
                    analysis,
                )
                if path.is_file()
                else None
            )
        df = cache[source]
        if df is None:
            continue
        rows = df[(df["model"] == model) & (df["feature_set"] == fs)]
        if not rows.empty:
            out[(model, fs)] = rows
    return out


def _strict_frames(frame: pd.DataFrame, ids: set, cfg=config) -> pd.DataFrame:
    part = frame[frame["participant_id"].isin(ids)].copy()
    part["feature_set"] = part["feature_set"].map(lambda fs: strict_set(fs, cfg))
    return part


def evaluation_frame(preds: pd.DataFrame, refs: dict, ctx: Context, cfg=config):
    """v5 + reference rows on the primary set and on the strict group."""
    low, high = strict_groups(ctx, cfg)
    strict = set(low) | set(high)
    primary = evaluator.analysis_frame(preds, cfg.signal_v5.analysis)
    ref_rows = list(refs.values())
    frames = [primary, _strict_frames(primary, strict, cfg)]
    frames += ref_rows + [_strict_frames(r, strict, cfg) for r in ref_rows]
    return pd.concat(frames, ignore_index=True)


def _paired(result, label: str, key_a: tuple, key_b: tuple, cfg=config) -> dict:
    analysis = cfg.signal_v5.analysis
    row = {
        "comparison": label,
        "model": "/".join(key_a),
        "reference": "/".join(key_b),
        "delta_macro_f1": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "note": "",
    }
    a = result.groups.get((analysis, *key_a))
    b = result.groups.get((analysis, *key_b))
    if a is None:
        raise ValueError(f"v5 group {row['model']} was not evaluated")
    if b is None:
        row["note"] = NO_REFERENCE
        return row
    row["delta_macro_f1"], row["ci_low"], row["ci_high"] = evaluator.paired_diff_ci(
        a, b, "macro_f1", cfg.evaluation.n_bootstrap, cfg.training.seed
    )
    return row


def step_evaluate(cfg=config) -> Path:
    """metrics, per_fold, confusion and paired (primary + strict); no metric printed."""
    out = _start(cfg)
    ctx = context(cfg)
    refs = reference_frames(cfg)
    result = evaluator.evaluate(evaluation_frame(_read_preds(cfg), refs, ctx, cfg), cfg)
    models = set(v5_models(cfg))
    written = []
    for name, table in (
        ("metrics.csv", result.comparison),
        ("per_fold.csv", result.per_fold),
        ("confusion.csv", result.confusion),
    ):
        path = out / name
        table[table["model"].isin(models)].to_csv(path, index=False)
        written.append(path)
    paired = []
    for tag, (model, fs) in headlines(cfg).items():
        suffix = "" if tag == "H1" else f"@{cfg.signal_v5.strict_name}"
        for ref_model, ref_fs in cfg.signal_v5.paired_references:
            paired.append(
                _paired(result, tag, (model, fs), (ref_model, ref_fs + suffix), cfg)
            )
    base_model, base_fs = cfg.signal_v5.paired_references[0]
    for model, fs in (("lr_v5", "all_v5"), ("ridge_v5", "resnet18")):
        if model in models:
            paired.append(
                _paired(result, "ablation", (model, fs), (base_model, base_fs), cfg)
            )
    path = out / "paired.csv"
    pd.DataFrame(paired).to_csv(path, index=False)
    written.append(path)
    for path in written:
        print(f"Written: {path}")
    return out


def curve_frame(preds: pd.DataFrame, ctx: Context, cfg=config) -> pd.DataFrame:
    """Rows of the curve models restricted to boundary_distance >= d."""
    distance = pd.Series(ctx.column(DISTANCE_COLUMN, ctx.full), index=ctx.full)
    frames = []
    for model, fs in cfg.signal_v5.curve_rows:
        rows = preds[(preds["model"] == model) & (preds["feature_set"] == fs)]
        for d in cfg.signal_v5.extremity_distances:
            keep = set(distance.index[distance >= d])
            part = rows[rows["participant_id"].isin(keep)].copy()
            part["feature_set"] = curve_set(fs, d)
            part["analysis"] = "full"
            part["in_middle_band"] = False
            frames.append(part)
    return pd.concat(frames, ignore_index=True)


def step_curve(cfg=config) -> Path:
    """extremity_curve.csv and fig_v5_extremity.png (descriptive)."""
    out = _start(cfg)
    ctx = context(cfg)
    result = evaluator.evaluate(curve_frame(_read_preds(cfg), ctx, cfg), cfg)
    comp = result.comparison.set_index(["model", "feature_set"])
    rows = []
    for model, fs in cfg.signal_v5.curve_rows:
        for d in cfg.signal_v5.extremity_distances:
            key = (model, curve_set(fs, d))
            if key not in comp.index:
                rows.append(
                    {"model": model, "feature_set": fs, "d": d, "note": "skipped"}
                )
                continue
            r = comp.loc[key]
            row = {"model": model, "feature_set": fs, "d": d}
            for col in ("n_participants", "n_sad", "n_happy", "majority_baseline"):
                row[col] = r[col]
            for m in v3.TABLE_METRICS:
                for suffix in ("", "_ci_low", "_ci_high"):
                    row[f"{m}{suffix}"] = r[f"{m}{suffix}"]
            row["note"] = ""
            rows.append(row)
    table = pd.DataFrame(rows)
    path = out / "extremity_curve.csv"
    table.to_csv(path, index=False)
    figure = figure_path("extremity.png", cfg)
    figure.parent.mkdir(parents=True, exist_ok=True)
    _plot_curve(table, figure, cfg)
    for p in (path, figure):
        print(f"Written: {p}")
    return path


def _plot_curve(table: pd.DataFrame, path: Path, cfg=config) -> None:
    table = table[table["note"] == ""] if "note" in table else table
    ds = list(cfg.signal_v5.extremity_distances)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    offsets = np.linspace(-0.25, 0.25, len(cfg.signal_v5.curve_rows))
    for i, (model, fs) in enumerate(cfg.signal_v5.curve_rows):
        part = table[(table["model"] == model) & (table["feature_set"] == fs)]
        x = np.asarray(part["d"], dtype=float) + offsets[i]
        y = part["accuracy"].to_numpy(dtype=float)
        err = np.vstack([y - part["accuracy_ci_low"], part["accuracy_ci_high"] - y])
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=CURVE_COLORS[i % len(CURVE_COLORS)],
            marker=CURVE_MARKERS[i % len(CURVE_MARKERS)],
            markersize=7,
            linewidth=2,
            capsize=3,
            label=f"{model}/{fs}",
        )
        if i == 0:
            ax.plot(
                part["d"],
                part["majority_baseline"],
                color="#888888",
                linestyle=":",
                linewidth=1.2,
                label="majority baseline",
            )
            for _, r in part.iterrows():
                ax.annotate(
                    f"n={int(r['n_participants'])}",
                    xy=(r["d"], 0.02),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    fontsize=7,
                    color="#555555",
                )
    ax.set_xticks(ds, [f">= {d:g}" for d in ds])
    ax.set_xlabel("distance of the self-reported score from the cutoff")
    ax.set_ylabel("participant accuracy [95% CI]")
    ax.set_title("v5 extremity curve (exploratory, descriptive)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    evaluator._save(fig, path, cfg)


# ---------------------------------------------------------------- permutation


def headline_folds(ctx: Context, cfg=config) -> list:
    """Label-free RidgeFold of the headline feature set for every fold."""
    _, fs = cfg.signal_v5.headline
    X = feature_table(fs, ctx, cfg)
    return [
        ridge_fold(X, fold_sets(fold, ctx), ctx, fold, cfg)
        for fold in range(cfg.cv.n_splits)
    ]


def shuffled_targets(rf: RidgeFold, ctx: Context, perm: int, fold: int, cfg=config):
    """(score, sad) of the outer-train participants, shuffled together."""
    score = ctx.column(SCORE_COLUMN, rf.train_ids).astype(float)
    sad = ctx.is_sad(rf.train_ids, cfg)
    rng = np.random.default_rng(permutation_seed(cfg.training.seed, perm, fold))
    order = rng.permutation(len(score))
    return score[order], sad[order]


def draw(perm: int, folds: list, ctx: Context, cfg=config) -> pd.DataFrame:
    """One null draw of the headline model: test rows, true test labels."""
    model, fs = cfg.signal_v5.headline
    frames = []
    for fold, rf in enumerate(folds):
        score, sad = shuffled_targets(rf, ctx, perm, fold, cfg)
        prob = fit_ridge(rf, score, sad, cfg)["prob"]
        frames.append(_pred_rows(model, fs, fold, rf.test_ids, prob, ctx, cfg))
    return pd.concat(frames, ignore_index=True).assign(perm=perm)


def _draw_one(perm, folds, ctx, cfg) -> int:
    path = draw_path(perm, cfg.training.seed, cfg)
    if not path.is_file():
        v3.write_part(draw(perm, folds, ctx, cfg), path)
    return perm


def statistics(rows: pd.DataFrame, strict: set, cfg=config) -> dict:
    """H1 and H2 macro-F1 and accuracy of one prediction frame."""
    primary = evaluator.analysis_frame(rows, cfg.signal_v5.analysis)
    out = {}
    for tag, part in (
        ("H1", primary),
        ("H2", primary[primary["participant_id"].isin(strict)]),
    ):
        m = evaluator.metrics(part)
        out[tag] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
    return out


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def step_permutation(cfg=config, n_jobs: int = 1) -> Path:
    """Null draws (cached, resumable) and p-values once all exist."""
    out = _start(cfg)
    ctx = context(cfg)
    v5 = cfg.signal_v5
    seed = cfg.training.seed
    n = v5.n_permutations
    todo = [k for k in range(n) if not draw_path(k, seed, cfg).is_file()]
    if todo:
        began = time.perf_counter()
        folds = headline_folds(ctx, cfg)
        print(f"projections: {time.perf_counter() - began:.1f} s", flush=True)
        began = time.perf_counter()
        first, rest = todo[:3], todo[3:]
        for k in first:
            _draw_one(k, folds, ctx, cfg)
        each = (time.perf_counter() - began) / len(first)
        print(
            f"draws: {len(first)} in {each * len(first):.1f} s; projected "
            f"{each * len(rest) / 60:.1f} min for {len(rest)} more (1 job)",
            flush=True,
        )
        Parallel(n_jobs=n_jobs)(delayed(_draw_one)(k, folds, ctx, cfg) for k in rest)
    paths = [draw_path(k, seed, cfg) for k in range(n)]
    done = sum(p.is_file() for p in paths)
    if n == 0 or done < n:
        raise RuntimeError(f"draws incomplete ({done}/{n})")
    low, high = strict_groups(ctx, cfg)
    strict = set(low) | set(high)
    model, fs = v5.headline
    preds = _read_preds(cfg)
    observed = statistics(
        preds[(preds["model"] == model) & (preds["feature_set"] == fs)], strict, cfg
    )
    nulls = {
        "H1": {"macro_f1": [], "accuracy": []},
        "H2": {"macro_f1": [], "accuracy": []},
    }
    for p in paths:
        stats = statistics(
            evaluator.validate(
                pd.read_csv(p, dtype={"participant_id": str}).drop(columns="perm"),
                v5.analysis,
                str(p),
                cfg,
            ),
            strict,
            cfg,
        )
        for tag in nulls:
            for metric in nulls[tag]:
                nulls[tag][metric].append(stats[tag][metric])
    result = {"n": n, "seed": seed, "statistic": "macro_f1", "alpha": v5.alpha}
    raw = {}
    for tag, key in headlines(cfg).items():
        entry = {"group": "/".join(key)}
        for metric in ("macro_f1", "accuracy"):
            null = np.asarray(nulls[tag][metric])
            entry[f"observed_{metric}"] = observed[tag][metric]
            entry[f"p_{metric}"] = p_value(null, observed[tag][metric])
            entry[f"null_mean_{metric}"] = float(null.mean())
            entry[f"null_{metric}"] = null.tolist()
        raw[tag] = entry["p_macro_f1"]
        result[tag] = entry
    adjusted = holm(raw)
    for tag in raw:
        result[tag]["p_macro_f1_holm"] = adjusted[tag]
        result[tag]["significant_holm"] = bool(adjusted[tag] < v5.alpha)
    path = out / "permutation.json"
    v3._write_json(result, path)
    for tag in raw:
        figure = figure_path(f"permutation_null_{tag.lower()}.png", cfg)
        figure.parent.mkdir(parents=True, exist_ok=True)
        _plot(
            np.asarray(result[tag]["null_macro_f1"]),
            result[tag]["observed_macro_f1"],
            result[tag]["p_macro_f1"],
            f"v5 {tag} {result[tag]['group']}: null (N = {n}, exploratory)",
            figure,
            cfg,
        )
        print(f"Written: {figure}")
    print(f"Written: {path}")
    return path


# ---------------------------------------------------------------- report


def _metric_line(row) -> str:
    parts = [f"{v3.METRIC_NAMES[m]} {v3._ci(row, m)}" for m in v3.TABLE_METRICS]
    return "; ".join(parts) + f"; majority {row['majority_baseline']:.3f}"


def _permutation_lines(tag: str, cfg=config) -> list:
    path = output_dir(cfg) / "permutation.json"
    if not path.is_file():
        done = len(list(runs_dir(cfg.training.seed, cfg).glob("perm_*.csv")))
        return [
            f"  permutation test: draws incomplete ({done}/"
            f"{cfg.signal_v5.n_permutations})"
        ]
    perm = json.loads(path.read_text(encoding="utf-8"))
    e = perm[tag]
    return [
        f"  permutation test (N = {perm['n']}, seed {perm['seed']}):",
        f"    macro-F1  p = {e['p_macro_f1']:.4f}, Holm-adjusted p = "
        f"{e['p_macro_f1_holm']:.4f} (null mean {e['null_mean_macro_f1']:.3f})",
        f"    accuracy  p = {e['p_accuracy']:.4f} (null mean "
        f"{e['null_mean_accuracy']:.3f})",
    ]


def _paired_text(r) -> str:
    if isinstance(r["note"], str) and r["note"]:
        return r["note"]
    return f"{r['delta_macro_f1']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"


def step_report(cfg=config) -> Path:
    """EXPLORATORY_v5.txt (needs evaluate and curve; permutation if run)."""
    _start(cfg)
    v5 = cfg.signal_v5
    out = output_dir(cfg)
    ctx = context(cfg)
    metrics = pd.read_csv(out / "metrics.csv")
    paired = pd.read_csv(out / "paired.csv")
    curve = pd.read_csv(out / "extremity_curve.csv")
    alphas = pd.read_csv(out / "ridge_alpha.csv")
    by_key = metrics.set_index(["model", "feature_set"])
    low, high = strict_groups(ctx, cfg)
    sad = ctx.is_sad(ctx.primary, cfg)
    lines = [
        "EMHA v5 -- signal boost",
        REPORT_HEADER,
        "",
        "1. Data",
        f"  primary set {len(ctx.primary)} (HAPPY {int((~sad).sum())}, SAD "
        f"{int(sad.sum())}); middle band {len(ctx.middle())} (training side of "
        "ridge_v5 only)",
        f"  strict group (H2): lowest {v5.strict_quantile:.0%} of total_score "
        f"({len(low)} SAD, scores {ctx.column(SCORE_COLUMN, low).min():g}-"
        f"{ctx.column(SCORE_COLUMN, low).max():g}) and highest "
        f"{v5.strict_quantile:.0%} ({len(high)} HAPPY, scores "
        f"{ctx.column(SCORE_COLUMN, high).min():g}-"
        f"{ctx.column(SCORE_COLUMN, high).max():g})",
        f"  ridge alpha chosen per fold ({'/'.join(v5.headline)}): "
        + ", ".join(
            f"{a:g}" for a in alphas[alphas["feature_set"] == v5.headline[1]]["alpha"]
        ),
        "",
    ]
    for i, (tag, key) in enumerate(headlines(cfg).items(), start=2):
        row = by_key.loc[key]
        name = "primary set" if tag == "H1" else "strict group"
        lines += [
            f"{i}. Pre-declared headline {tag} ({name}): {'/'.join(key)}",
            f"  n = {int(row['n_participants'])} (SAD {int(row['n_sad'])}, HAPPY "
            f"{int(row['n_happy'])})",
            f"  {_metric_line(row)}",
            *_permutation_lines(tag, cfg),
            "  paired macro-F1 difference, headline - reference [CI]:",
        ]
        for _, r in paired[paired["comparison"] == tag].iterrows():
            lines.append(f"    vs {r['reference']:<28} {_paired_text(r)}")
        lines.append("")
    lines += ["4. Every v5 row, primary set (descriptive except H1)"]
    for model in v5.models:
        for fs, _ in v5.feature_sets:
            lines.append(f"  {model}/{fs}: {_metric_line(by_key.loc[(model, fs)])}")
    lines += ["", "5. Every v5 row, strict group (descriptive except H2)"]
    for model in v5.models:
        for fs, _ in v5.feature_sets:
            key = (model, strict_set(fs, cfg))
            lines.append(f"  {model}/{key[1]}: {_metric_line(by_key.loc[key])}")
    lines += ["", "6. Ablations vs v1 lr_embeddings/concat (macro-F1 [CI])"]
    for _, r in paired[paired["comparison"] == "ablation"].iterrows():
        lines.append(f"  {r['model']:<22} {_paired_text(r)}")
    lines += [
        "",
        "7. Extremity curve (descriptive): test participants with "
        "boundary_distance >= d",
        "  model/feature_set     d     n  SAD HAPPY  majority  accuracy [CI]"
        "            balanced acc.   ROC-AUC",
    ]
    for _, r in curve.iterrows():
        if isinstance(r.get("note"), str) and r["note"]:
            lines.append(
                f"  {r['model']}/{r['feature_set']} d>={r['d']:g}: {r['note']}"
            )
            continue
        lines.append(
            f"  {r['model'] + '/' + r['feature_set']:<20} {r['d']:>3g} "
            f"{int(r['n_participants']):>5} {int(r['n_sad']):>4} "
            f"{int(r['n_happy']):>5}  {r['majority_baseline']:.3f}     "
            f"{r['accuracy']:.3f} [{r['accuracy_ci_low']:.3f}, "
            f"{r['accuracy_ci_high']:.3f}]  {r['balanced_accuracy']:.3f}  "
            f"{r['roc_auc']:.3f}"
        )
    lines += [
        "",
        "  Reading: if accuracy rises with d, the model tracks how clearly",
        "  happy or sad people report being. A flat curve means it does not.",
        "",
        "8. " + LIMITATIONS,
    ]
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {path}")
    return path


# ---------------------------------------------------------------- CLI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--n-jobs", type=int, default=1, help="speed only")
    args = parser.parse_args()
    if args.step == "preflight":
        preflight(config)
    elif args.step == "fit":
        step_fit(config)
    elif args.step == "evaluate":
        step_evaluate(config)
    elif args.step == "curve":
        step_curve(config)
    elif args.step == "permutation":
        step_permutation(config, n_jobs=args.n_jobs)
    elif args.step == "report":
        step_report(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
