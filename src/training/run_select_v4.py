"""
Exercise selection v4 -- EXPLORATORY, post hoc (EVALUATION_PROTOCOL.md
section 9, Deviation 7). Never part of the primary claim; the primary model
stays CNN-HMM v1 (run_hybrid.py). v1, v2 and v3 outputs are only read.

Idea: test every exercise on its own, drop the weak ones and keep the
strong ones. The catch is that "strong" must be decided without looking at
the participants a model is scored on, otherwise the kept exercises look
better than they are (selection on the test set). So the whole selection is
repeated inside every outer fold, on that fold's training participants only:

  1. train, test = fold_ids(folds.csv, fold, primary IDs); assert_no_leakage
  2. rank: for each of the 24 exercises and each LR kind in
     SelectV4Config.rank_kinds, rank_inner_folds-fold out-of-fold ROC-AUC on
     the training participants (fit_lr as in run_baselines, features as in
     v3). The exercise's score is the mean over the kinds; rank 1 is the
     strongest (ties in ALL_CELLS order).
  3. select: for k in SelectV4Config.k_grid keep the top k exercises.
     top24 keeps everything and is the no-removal control.
  4. fit and predict, test participants only:
       lr_handcrafted_v4 / lr_embeddings_v4: one LR on the kept exercises'
           features side by side (fit_lr; PCA for embeddings)
       cnn_head_v4 / cnn_hmm_v4: plain mean of v3's per-exercise test
           prob_sad over the kept exercises. v3 trained those models on the
           same fold's training participants only, so nothing is retrained.
     cnn_hmm_v4/top24 equals v3's cnn_hmm_v3_fused/all24 (a built-in check).

The headline, fixed before any v4 result, is cnn_hmm_v4/top<headline_k>.
Everything else is descriptive. selection_frequency.csv counts in how many
of the 5 folds each exercise made the top headline_k: an exercise kept in
every fold is consistently informative; one kept once is likely noise.

Outputs (v4 paths only):

    results/select_v4/primary/run_config.json   a rerun refuses a mismatch
    results/select_v4/primary/parts/rank_fold<k>.csv
    results/select_v4/primary/ranking.csv, selected_cells.csv,
        selection_frequency.csv, predictions.csv, metrics.csv, per_fold.csv,
        confusion.csv, paired.csv, permutation.json
    results/final/permutation_runs/primary_v4/seed_<s>/perm_<k>.csv
    results/final/EXPLORATORY_v4.txt, fig_v4_k_curve.png,
        fig_v4_selection.png, fig_v4_permutation_null.png

    python -m src.training.run_select_v4 --step preflight
    python -m src.training.run_select_v4 --step rank [--n-jobs -1]
    python -m src.training.run_select_v4 --step combine
    python -m src.training.run_select_v4 --step evaluate
    python -m src.training.run_select_v4 --step draws [--draws 0-19] [--n-jobs -1]
    python -m src.training.run_select_v4 --step permutation
    python -m src.training.run_select_v4 --step report

All steps are CPU only. Inputs: results/features/handcrafted_crop.csv,
results/embeddings/resnet18_<family>.npz, v3's predictions.csv and, for the
CNN rows of the null, v3's cached draws (permutation_runs/primary_v3). In
draw k, fold f, the training labels are shuffled with
permutation_seed(seed, k, f), the same shuffle v3 used for that draw, so the
ranking, the LR refits and v3's CNN draw all see the same null labels.

Nothing is appended to results/RUN_LOG.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.analysis.permutation import _plot, p_value, parse_draws, permutation_seed
from src.data.crop_manifest import ALL_CELLS
from src.training import evaluator
from src.training import run_per_task_v3 as v3
from src.training.run_baselines import PRED_COLUMNS, _prob_sad, fit_lr
from src.training.run_cnn import shuffled_labels
from src.training.splits import assert_no_leakage, fold_ids, folds_path
from src.utils.config import config
from src.utils.protocol_guard import is_synthetic_root, require_frozen_or_synthetic

STEPS = (
    "preflight",
    "rank",
    "combine",
    "evaluate",
    "draws",
    "permutation",
    "report",
)
PERMUTATION_KEYS = ("n_permutations",)
DRAWS_KIND = "v4_draws"
NO_REFERENCE = "reference not found"
REPORT_HEADER = (
    "EXPLORATORY, POST HOC (Deviation 7). Not part of the primary claim; the "
    "primary model remains CNN-HMM v1 (cnn_hmm_fused/fused)."
)
LIMITATIONS = (
    "Limitations. The ranking uses about 256 training participants per fold, "
    "so an exercise's score has sampling noise of its own; the stability "
    "table shows how much the kept set moves between folds. Only the "
    "headline row is a pre-declared test; the k curve and the other models "
    "are descriptive, and picking the best-looking k after the fact would "
    "re-introduce the selection bias this design avoids. The ranking uses the "
    "LR models only (cheap enough to repeat inside every fold); the CNN rows "
    "reuse v3's per-exercise models, which were trained on the same training "
    "participants. v4 is post hoc (Deviation 7), decided after v1, v2 and v3 "
    "results were known."
)
MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MODEL_MARKERS = ("o", "s", "^", "D")


# ---------------------------------------------------------------- paths


def output_dir(cfg=config) -> Path:
    v4 = cfg.select_v4
    return cfg.paths.results_dir / v4.output_subdir / v4.analysis


def rank_part_path(fold: int, cfg=config) -> Path:
    return output_dir(cfg) / "parts" / f"rank_fold{fold}.csv"


def runs_dir(seed: int, cfg=config) -> Path:
    out = evaluator.output_dir(None, cfg) / cfg.permutation.runs_subdir
    return out / cfg.select_v4.runs_name / f"seed_{seed}"


def draw_path(perm: int, seed: int, cfg=config) -> Path:
    return runs_dir(seed, cfg) / f"perm_{perm:04d}.csv"


def report_path(cfg=config) -> Path:
    return evaluator.output_dir(None, cfg) / cfg.select_v4.report_name


def figure_path(name: str, cfg=config) -> Path:
    return evaluator.output_dir(None, cfg) / f"{cfg.select_v4.figure_prefix}{name}"


def v3_predictions_path(cfg=config) -> Path:
    return v3.output_dir(cfg) / "predictions.csv"


def top_name(k: int) -> str:
    return f"top{k}"


def v4_models(cfg=config) -> tuple:
    v4 = cfg.select_v4
    return tuple(m for m, _ in v4.lr_models) + tuple(m for m, _ in v4.fused_models)


def headline(cfg=config) -> tuple:
    v4 = cfg.select_v4
    return v4.headline_model, top_name(v4.headline_k)


# ---------------------------------------------------------------- inputs


def check_config(cfg=config) -> None:
    """Refuse settings that would make v4 incoherent with v3."""
    v4 = cfg.select_v4
    cells = tuple(cfg.per_task_v3.cells)
    if v4.analysis != cfg.per_task_v3.analysis:
        raise ValueError("SelectV4Config.analysis must equal PerTaskV3Config.analysis")
    if not is_synthetic_root(cfg.paths.data_root) and cells != ALL_CELLS:
        raise ValueError("PerTaskV3Config.cells must equal crop_manifest.ALL_CELLS")
    ks = tuple(v4.k_grid)
    if sorted(set(ks)) != list(ks) or ks[0] < 1 or ks[-1] != len(cells):
        raise ValueError(
            f"k_grid {ks} must be increasing, start at >= 1 and end at "
            f"{len(cells)} (the no-removal control)"
        )
    if v4.headline_k not in ks or v4.headline_k == len(cells):
        raise ValueError("headline_k must be in k_grid and below the cell count")
    if v4.headline_model not in v4_models(cfg):
        raise ValueError(f"headline_model {v4.headline_model!r} is not a v4 model")
    kinds = set(cfg.per_task_v3.lr_kinds)
    used = set(v4.rank_kinds) | {kind for _, kind in v4.lr_models}
    if not used <= kinds:
        raise ValueError(f"LR kinds {sorted(used - kinds)} are not v3 LR kinds")


def load_tables(cfg=config) -> dict:
    """LR kind -> cell -> that cell's feature table, indexed by participant_id."""
    kinds = list(cfg.select_v4.rank_kinds)
    kinds += [k for _, k in cfg.select_v4.lr_models if k not in kinds]
    return {
        kind: {
            cell: v3._lr_inputs(kind, cell, cfg)[0] for cell in cfg.per_task_v3.cells
        }
        for kind in kinds
    }


def _label_index(labels: pd.DataFrame, pids, cfg=config) -> np.ndarray:
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    return lab.loc[list(pids)].map(cfg.data.label_to_index).to_numpy()


def load_v3_cells(cfg=config) -> pd.DataFrame:
    """v3's per-exercise test rows of the models v4 fuses."""
    path = v3_predictions_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; v4 needs v3's predictions.csv (run_per_task_v3 "
            "--step combine)"
        )
    df = pd.read_csv(path, dtype={"participant_id": str})
    sources = [src for _, src in cfg.select_v4.fused_models]
    cells = list(cfg.per_task_v3.cells)
    return df[df["model"].isin(sources) & df["feature_set"].isin(cells)]


def _sha256(path: Path) -> str:
    return v3._sha256(path)


def settings(cfg=config) -> dict:
    """Everything a rerun must match (run_config.json)."""
    v4 = asdict(cfg.select_v4)
    for key in PERMUTATION_KEYS:
        v4.pop(key)
    return json.loads(
        json.dumps(
            {
                "select_v4": v4,
                "cells": list(cfg.per_task_v3.cells),
                "baselines": asdict(cfg.baselines),
                "seed": cfg.training.seed,
                "n_splits": cfg.cv.n_splits,
                "threshold": cfg.aggregate.threshold,
                "folds_sha256": _sha256(folds_path(cfg)),
                "labels_sha256": _sha256(v3.labels_csv_path(cfg)),
                "handcrafted_crop_sha256": _sha256(v3.handcrafted_path(cfg)),
                "embeddings_sha256": {
                    f: _sha256(v3.embedding_path(f, cfg)) for f in v3.FAMILIES
                },
                "v3_predictions_sha256": _sha256(v3_predictions_path(cfg)),
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
                f"{path} records different settings ({changed}); v4 runs once "
                "(Deviation 7). Log an addendum under Deviation 7 before any "
                "rerun; never delete the existing v4 outputs."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(want, indent=2), encoding="utf-8")


def preflight(cfg=config) -> dict:
    """Check every v4 input before anything is fit; print and return a summary."""
    require_frozen_or_synthetic(cfg, "run_select_v4")
    check_config(cfg)
    hashes = v3._check_hashes(cfg)
    labels, ids, folds = v3.context(cfg)
    tables = load_tables(cfg)
    for kind, per_cell in tables.items():
        for cell, X in per_cell.items():
            v3._require_rows(kind, cell, X.index, ids)
            if X.loc[ids].isna().any().any():
                raise ValueError(f"{kind} {cell}: NaN values for primary participants")
    per_cell = load_v3_cells(cfg)
    gaps = []
    for _, src in cfg.select_v4.fused_models:
        for cell in cfg.per_task_v3.cells:
            for fold in range(cfg.cv.n_splits):
                _, test = fold_ids(folds, fold, ids)
                got = per_cell[
                    (per_cell["model"] == src)
                    & (per_cell["feature_set"] == cell)
                    & (per_cell["fold"] == fold)
                ]["participant_id"]
                if set(got) != set(test):
                    gaps.append(f"{src}/{cell}/fold{fold}")
    if gaps:
        raise ValueError(
            f"v3 predictions.csv does not hold exactly the test participants for "
            f"{len(gaps)} (model, cell, fold), e.g. {gaps[:5]}"
        )
    v3_draws = sum(
        v3.draw_path(k, cfg.training.seed, cfg).is_file()
        for k in range(max(cfg.select_v4.n_permutations, 0))
    )
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    counts = lab.loc[ids].value_counts()
    summary = {
        "analysis": cfg.select_v4.analysis,
        "synthetic_root": is_synthetic_root(cfg.paths.data_root),
        **hashes,
        "n_primary": len(ids),
        "n_happy": int(counts.get("HAPPY", 0)),
        "n_sad": int(counts.get("SAD", 0)),
        "n_cells": len(cfg.per_task_v3.cells),
        "k_grid": list(cfg.select_v4.k_grid),
        "headline": "/".join(headline(cfg)),
        "rank_kinds": list(cfg.select_v4.rank_kinds),
        "v3_draws_present": f"{v3_draws}/{cfg.select_v4.n_permutations}",
    }
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- ranking


def oof_auc(
    X: pd.DataFrame, train: list, y: np.ndarray, use_pca: bool, seed: int, cfg=config
) -> float:
    """Out-of-fold ROC-AUC of fit_lr over the training participants only."""
    y = np.asarray(y)
    n = min(cfg.select_v4.rank_inner_folds, int(np.bincount(y, minlength=2).min()))
    if n < 2:
        raise ValueError("each class needs >= 2 training participants to rank")
    Xt = X.loc[train].to_numpy(dtype=float)
    prob = np.empty(len(train), dtype=float)
    split = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    for tr, va in split.split(Xt, y):
        pipe = fit_lr(Xt[tr], y[tr], cfg, pca=use_pca, seed=seed)
        prob[va] = _prob_sad(pipe, Xt[va])
    sad = cfg.data.label_to_index["SAD"]
    return float(roc_auc_score(y == sad, prob))


def rank_fold(
    fold: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    tables: dict,
    cfg=config,
    shuffle_seed: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """One fold's exercise ranking, from its training participants only.

    shuffle_seed permutes the training labels (null draws); the test
    participants are never read, so their features and labels cannot move
    the ranking.
    """
    v4 = cfg.select_v4
    train, test = fold_ids(folds, fold, ids)
    assert_no_leakage(train, test)
    fit_labels = (
        labels if shuffle_seed is None else shuffled_labels(labels, train, shuffle_seed)
    )
    y = _label_index(fit_labels, train, cfg)
    cells = list(cfg.per_task_v3.cells)
    jobs = [(cell, kind) for cell in cells for kind in v4.rank_kinds]
    seed = cfg.training.seed + fold
    aucs = Parallel(n_jobs=n_jobs)(
        delayed(oof_auc)(
            tables[kind][cell], train, y, v3._lr_use_pca(kind, cfg), seed, cfg
        )
        for cell, kind in jobs
    )
    score = dict(zip(jobs, aucs))
    rows = []
    for cell in cells:
        row = {"fold": fold, "cell": cell, "family": v3.family_of(cell)}
        for kind in v4.rank_kinds:
            row[f"auc_{kind}"] = score[(cell, kind)]
        row["score"] = float(np.mean([score[(cell, k)] for k in v4.rank_kinds]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["order"] = range(len(frame))
    frame = frame.sort_values(["score", "order"], ascending=[False, True])
    frame["rank"] = range(1, len(frame) + 1)
    return frame.drop(columns="order").reset_index(drop=True)


def selection(ranking: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """selected_cells.csv: per (k, fold) the kept cells, best first."""
    rows = []
    for k in cfg.select_v4.k_grid:
        for fold, part in ranking.groupby("fold"):
            kept = part.sort_values("rank").head(k)
            for _, r in kept.iterrows():
                rows.append(
                    {
                        "k": k,
                        "fold": int(fold),
                        "rank": int(r["rank"]),
                        "cell": r["cell"],
                        "score": float(r["score"]),
                    }
                )
    return pd.DataFrame(rows)


def kept_cells(selected: pd.DataFrame, k: int, fold: int) -> list:
    part = selected[(selected["k"] == k) & (selected["fold"] == fold)]
    return list(part.sort_values("rank")["cell"])


def selection_frequency(ranking: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """Per cell: folds in the top headline_k, mean rank, mean score."""
    k = cfg.select_v4.headline_k
    n_folds = ranking["fold"].nunique()
    rows = []
    for cell in cfg.per_task_v3.cells:
        part = ranking[ranking["cell"] == cell]
        family, task, style = v3._task_fields(cell)
        rows.append(
            {
                "cell": cell,
                "family": family,
                "task": task,
                "style": style,
                f"folds_in_top{k}": int((part["rank"] <= k).sum()),
                "n_folds": n_folds,
                "mean_rank": float(part["rank"].mean()),
                "mean_score": float(part["score"].mean()),
                "min_score": float(part["score"].min()),
                "max_score": float(part["score"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out["order"] = range(len(out))
    out = out.sort_values(
        [f"folds_in_top{k}", "mean_rank", "order"], ascending=[False, True, True]
    )
    return out.drop(columns="order").reset_index(drop=True)


# ---------------------------------------------------------------- models


def lr_rows(
    model: str,
    kind: str,
    fold: int,
    cells: list,
    k: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    tables: dict,
    cfg=config,
    fit_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One LR on the kept cells side by side; test rows with the TRUE labels."""
    train, test = fold_ids(folds, fold, ids)
    assert_no_leakage(train, test)
    X = pd.concat([tables[kind][c].add_prefix(f"{c}__") for c in cells], axis=1)
    y = _label_index(labels if fit_labels is None else fit_labels, train, cfg)
    pipe = fit_lr(
        X.loc[train],
        y,
        cfg,
        pca=v3._lr_use_pca(kind, cfg),
        seed=cfg.training.seed + fold,
    )
    prob = _prob_sad(pipe, X.loc[test])
    return v3._participant_rows(model, top_name(k), fold, test, prob, labels, cfg)


def fuse_rows(
    per_cell: pd.DataFrame, src: str, out_model: str, selected: pd.DataFrame, cfg=config
) -> pd.DataFrame:
    """Per (k, fold): plain mean of the kept cells' per-exercise prob_sad."""
    rows = per_cell[per_cell["model"] == src]
    frames = []
    for k in cfg.select_v4.k_grid:
        for fold in range(cfg.cv.n_splits):
            cells = kept_cells(selected, k, fold)
            part = rows[(rows["fold"] == fold) & rows["feature_set"].isin(cells)]
            wide = part.pivot(
                index="participant_id", columns="feature_set", values="prob_sad"
            ).reindex(columns=cells)
            if wide.empty or wide.isna().any().any():
                raise ValueError(
                    f"{src} top{k} fold {fold}: per-exercise rows missing for "
                    f"some kept cells {cells}"
                )
            lab = part.drop_duplicates("participant_id").set_index("participant_id")
            prob = wide.mean(axis=1).to_numpy()
            frames.append(
                pd.DataFrame(
                    {
                        "analysis": cfg.select_v4.analysis,
                        "model": out_model,
                        "feature_set": top_name(k),
                        "fold": fold,
                        "participant_id": wide.index.to_numpy(),
                        "label": lab.loc[wide.index, "label"].to_numpy(),
                        "prob_sad": prob,
                        "pred": np.where(
                            prob >= cfg.aggregate.threshold, "SAD", "HAPPY"
                        ),
                        "in_middle_band": False,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)[list(PRED_COLUMNS)]


def predict(
    selected: pd.DataFrame,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    tables: dict,
    per_cell: pd.DataFrame | None,
    cfg=config,
    fit_labels_of=None,
) -> pd.DataFrame:
    """All v4 rows for one selection. fit_labels_of(fold) gives null labels."""
    frames = []
    for model, kind in cfg.select_v4.lr_models:
        for k in cfg.select_v4.k_grid:
            for fold in range(cfg.cv.n_splits):
                fit = None if fit_labels_of is None else fit_labels_of(fold)
                frames.append(
                    lr_rows(
                        model,
                        kind,
                        fold,
                        kept_cells(selected, k, fold),
                        k,
                        ids,
                        labels,
                        folds,
                        tables,
                        cfg,
                        fit,
                    )
                )
    if per_cell is not None:
        for out_model, src in cfg.select_v4.fused_models:
            frames.append(fuse_rows(per_cell, src, out_model, selected, cfg))
    return v3._sorted_rows(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------- steps


def _start(cfg=config) -> Path:
    require_frozen_or_synthetic(cfg, "run_select_v4")
    check_config(cfg)
    out = output_dir(cfg)
    check_settings(out / "run_config.json", settings(cfg))
    return out


def step_rank(cfg=config, n_jobs: int = 1) -> Path:
    """ranking.csv; resumable per fold."""
    out = _start(cfg)
    labels, ids, folds = v3.context(cfg)
    tables = load_tables(cfg)
    parts = []
    for fold in range(cfg.cv.n_splits):
        path = rank_part_path(fold, cfg)
        if path.is_file():
            print(f"resume: rank fold {fold}")
        else:
            began = time.perf_counter()
            frame = rank_fold(fold, ids, labels, folds, tables, cfg, n_jobs=n_jobs)
            v3.write_part(frame, path)
            print(f"rank fold {fold}: {time.perf_counter() - began:.1f} s", flush=True)
        parts.append(pd.read_csv(path))
    path = out / "ranking.csv"
    v3.write_part(pd.concat(parts, ignore_index=True), path)
    print(f"Written: {path}")
    return path


def _read_ranking(cfg=config) -> pd.DataFrame:
    path = output_dir(cfg) / "ranking.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run --step rank first")
    return pd.read_csv(path)


def step_combine(cfg=config) -> Path:
    """selected_cells.csv, selection_frequency.csv and predictions.csv."""
    out = _start(cfg)
    ranking = _read_ranking(cfg)
    labels, ids, folds = v3.context(cfg)
    selected = selection(ranking, cfg)
    preds = predict(
        selected, ids, labels, folds, load_tables(cfg), load_v3_cells(cfg), cfg
    )
    v3.write_part(selected, out / "selected_cells.csv")
    v3.write_part(selection_frequency(ranking, cfg), out / "selection_frequency.csv")
    path = out / "predictions.csv"
    v3.write_part(preds, path)
    evaluator.validate(
        pd.read_csv(path, dtype={"participant_id": str}),
        cfg.select_v4.analysis,
        str(path),
        cfg,
    )
    counts = preds.groupby(["model", "feature_set"], sort=False).size()
    print(f"combine: {len(counts)} groups")
    for (model, fs), n in counts.items():
        print(f"  {model}/{fs}: {n} rows")
    return path


def _read_frame(path: Path, cfg=config) -> pd.DataFrame:
    analysis = cfg.select_v4.analysis
    df = evaluator.validate(
        pd.read_csv(path, dtype={"participant_id": str}), analysis, str(path), cfg
    )
    return evaluator.analysis_frame(df, analysis)


def reference_frames(cfg=config) -> dict:
    """(model, feature_set) -> v1 primary rows, for the paired references."""
    out, cache = {}, {}
    for model, fs in cfg.select_v4.paired_references:
        source = v3.REFERENCE_SOURCES[model]
        path = (
            cfg.paths.results_dir / source / cfg.select_v4.analysis / "predictions.csv"
        )
        if source not in cache:
            cache[source] = _read_frame(path, cfg) if path.is_file() else None
        df = cache[source]
        if df is None:
            continue
        rows = df[(df["model"] == model) & (df["feature_set"] == fs)]
        if not rows.empty:
            out[(model, fs)] = rows
    return out


def _paired(result, label: str, key_a: tuple, key_b: tuple, cfg=config) -> dict:
    analysis = cfg.select_v4.analysis
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
        raise ValueError(f"v4 group {row['model']} was not evaluated")
    if b is None:
        row["note"] = NO_REFERENCE
        return row
    row["delta_macro_f1"], row["ci_low"], row["ci_high"] = evaluator.paired_diff_ci(
        a, b, "macro_f1", cfg.evaluation.n_bootstrap, cfg.training.seed
    )
    return row


def step_evaluate(cfg=config) -> Path:
    """metrics, per_fold, confusion and paired; no metric printed."""
    _start(cfg)
    out = output_dir(cfg)
    mine = _read_frame(out / "predictions.csv", cfg)
    refs = reference_frames(cfg)
    result = evaluator.evaluate(pd.concat([mine, *refs.values()]), cfg)
    models = set(v4_models(cfg))
    written = []
    for name, table in (
        ("metrics.csv", result.comparison),
        ("per_fold.csv", result.per_fold),
        ("confusion.csv", result.confusion),
    ):
        path = out / name
        table[table["model"].isin(models)].to_csv(path, index=False)
        written.append(path)
    full = top_name(len(cfg.per_task_v3.cells))
    keep = top_name(cfg.select_v4.headline_k)
    paired = [
        _paired(result, "removal", (m, keep), (m, full), cfg) for m in v4_models(cfg)
    ]
    paired += [
        _paired(result, "headline_vs_v1", headline(cfg), tuple(ref), cfg)
        for ref in cfg.select_v4.paired_references
    ]
    path = out / "paired.csv"
    pd.DataFrame(paired).to_csv(path, index=False)
    written.append(path)
    for path in written:
        print(f"Written: {path}")
    return out


def draw(
    perm: int,
    seed: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    tables: dict,
    cfg=config,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """One null draw of the whole v4 procedure: rank, select, refit, fuse.

    Fold f shuffles its training labels with permutation_seed(seed, perm, f),
    v3's shuffle for the same draw; the CNN rows fuse v3's cached draw perm
    (missing -> LR rows only).
    """

    def fit_labels_of(fold):
        train, _ = fold_ids(folds, fold, ids)
        return shuffled_labels(labels, train, permutation_seed(seed, perm, fold))

    ranking = pd.concat(
        [
            rank_fold(
                fold,
                ids,
                labels,
                folds,
                tables,
                cfg,
                shuffle_seed=permutation_seed(seed, perm, fold),
                n_jobs=n_jobs,
            )
            for fold in range(cfg.cv.n_splits)
        ],
        ignore_index=True,
    )
    v3_draw = v3.draw_path(perm, seed, cfg)
    per_cell = None
    if v3_draw.is_file():
        frame = pd.read_csv(v3_draw, dtype={"participant_id": str})
        sources = [src for _, src in cfg.select_v4.fused_models]
        per_cell = frame[
            frame["model"].isin(sources)
            & frame["feature_set"].isin(list(cfg.per_task_v3.cells))
        ]
    preds = predict(
        selection(ranking, cfg),
        ids,
        labels,
        folds,
        tables,
        per_cell,
        cfg,
        fit_labels_of,
    )
    return preds.assign(perm=perm)


def draw_settings(cfg=config) -> dict:
    return {**settings(cfg), "kind": DRAWS_KIND}


def step_draws(cfg=config, draws: str | None = None, n_jobs: int = 1) -> Path | None:
    """Cache null draws (one perm_<k>.csv each); resumable."""
    _start(cfg)
    n = cfg.select_v4.n_permutations
    if n == 0:
        print("n_permutations = 0: no permutation test")
        return None
    seed = cfg.training.seed
    out = runs_dir(seed, cfg)
    check_settings(out / "run_config.json", draw_settings(cfg))
    labels, ids, folds = v3.context(cfg)
    tables = load_tables(cfg)
    for perm in parse_draws(draws if draws else f"0-{n - 1}", n):
        path = draw_path(perm, seed, cfg)
        if path.is_file():
            print(f"resume: draw {perm}")
            continue
        if not v3.draw_path(perm, seed, cfg).is_file():
            print(f"draw {perm}: v3 draw missing, LR rows only", flush=True)
        began = time.perf_counter()
        frame = draw(perm, seed, ids, labels, folds, tables, cfg, n_jobs)
        v3.write_part(frame, path)
        print(f"draw {perm}: {time.perf_counter() - began:.1f} s", flush=True)
    done = len(list(out.glob("perm_*.csv")))
    print(f"draws: {done}/{n} present in {out}")
    return out


def _macro_f1(frame: pd.DataFrame, model: str, fs: str, cfg=config) -> float:
    return v3._observed(frame, model, fs, cfg)


def step_permutation(cfg=config) -> Path:
    """p-values over all cached draws; only the headline is a test."""
    _start(cfg)
    n = cfg.select_v4.n_permutations
    seed = cfg.training.seed
    paths = [draw_path(k, seed, cfg) for k in range(n)]
    done = sum(p.is_file() for p in paths)
    if n == 0 or done < n:
        raise RuntimeError(f"draws incomplete ({done}/{n})")
    preds = pd.read_csv(
        output_dir(cfg) / "predictions.csv", dtype={"participant_id": str}
    )
    draws = [pd.read_csv(p, dtype={"participant_id": str}) for p in paths]
    groups = preds[["model", "feature_set"]].drop_duplicates().itertuples(index=False)
    per_group = {}
    for model, fs in groups:
        if not all(
            ((d["model"] == model) & (d["feature_set"] == fs)).any() for d in draws
        ):
            per_group[f"{model}/{fs}"] = {"note": "not in every draw"}
            continue
        null = np.array([_macro_f1(d, model, fs, cfg) for d in draws])
        observed = _macro_f1(preds, model, fs, cfg)
        per_group[f"{model}/{fs}"] = {
            "observed_macro_f1": observed,
            "p_value": p_value(null, observed),
            "null_mean": float(null.mean()),
            "null": null.tolist(),
        }
    key = "/".join(headline(cfg))
    head = per_group[key]
    result = {
        "headline": key,
        "analysis": cfg.select_v4.analysis,
        "n": n,
        "seed": seed,
        **{k: v for k, v in head.items() if k != "null"},
        "runs_dir": str(runs_dir(seed, cfg)),
        "descriptive": {k: v for k, v in per_group.items() if k != key},
    }
    out = output_dir(cfg) / "permutation.json"
    v3._write_json(result, out)
    if "null" in head:
        figure = figure_path("permutation_null.png", cfg)
        figure.parent.mkdir(parents=True, exist_ok=True)
        _plot(
            np.asarray(head["null"]),
            head["observed_macro_f1"],
            head["p_value"],
            f"v4 {key}: permutation null (N = {n}, exploratory)",
            figure,
            cfg,
        )
        print(f"Written: {figure}")
    print(f"Written: {out}")
    return out


# ---------------------------------------------------------------- report


def _ci(row, m: str) -> str:
    return f"{row[m]:.3f} [{row[f'{m}_ci_low']:.3f}, {row[f'{m}_ci_high']:.3f}]"


def _metric_line(row) -> str:
    parts = [f"{v3.METRIC_NAMES[m]} {_ci(row, m)}" for m in v3.TABLE_METRICS]
    return "; ".join(parts) + f"; majority {row['majority_baseline']:.3f}"


def _permutation_text(cfg=config) -> str:
    path = output_dir(cfg) / "permutation.json"
    if path.is_file():
        perm = json.loads(path.read_text(encoding="utf-8"))
        if "p_value" not in perm:
            return "not available (headline not in every draw)"
        return (
            f"p = {perm['p_value']:.4f} (N = {perm['n']}, seed {perm['seed']}, "
            f"null mean {perm['null_mean']:.3f})"
        )
    if cfg.select_v4.n_permutations == 0:
        return "not run (n_permutations = 0)"
    done = len(list(runs_dir(cfg.training.seed, cfg).glob("perm_*.csv")))
    return f"draws incomplete ({done}/{cfg.select_v4.n_permutations})"


def _plot_k_curve(metrics: pd.DataFrame, path: Path, cfg=config) -> None:
    ks = list(cfg.select_v4.k_grid)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    offsets = np.linspace(-0.12, 0.12, len(v4_models(cfg)))
    pos = np.arange(len(ks))
    for i, model in enumerate(v4_models(cfg)):
        mine = metrics[metrics["model"] == model].set_index("feature_set")
        rows = mine.loc[[top_name(k) for k in ks]]
        y = rows["macro_f1"].to_numpy()
        err = np.vstack([y - rows["macro_f1_ci_low"], rows["macro_f1_ci_high"] - y])
        ax.errorbar(
            pos + offsets[i],
            y,
            yerr=err,
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
            markersize=7,
            linewidth=2,
            capsize=3,
            label=model,
        )
    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax.text(pos[-1] + 0.3, 0.5, "0.5", va="center", fontsize=8, color="#555555")
    ax.axvline(
        ks.index(cfg.select_v4.headline_k), color="#bbbbbb", linewidth=1, zorder=0
    )
    ax.set_xticks(pos, [f"top {k}" for k in ks])
    ax.set_xlabel("exercises kept per fold (chosen on training participants only)")
    ax.set_ylabel("participant macro-F1 [95% CI]")
    ax.set_title("v4 exercise selection (exploratory): keeping fewer exercises")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluator._save(fig, path, cfg)


def _plot_selection(ranking: pd.DataFrame, path: Path, cfg=config) -> None:
    cells = list(cfg.per_task_v3.cells)
    folds = sorted(ranking["fold"].unique())
    grid = ranking.pivot(index="fold", columns="cell", values="rank")
    grid = grid.reindex(index=folds, columns=cells)
    k = cfg.select_v4.headline_k
    fig, ax = plt.subplots(figsize=(11.0, 2.8))
    ax.imshow(grid.to_numpy(), cmap="Blues_r", aspect="auto", vmin=1, vmax=len(cells))
    for (i, j), r in np.ndenumerate(grid.to_numpy()):
        kept = r <= k
        ax.text(
            j,
            i,
            f"{int(r)}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold" if kept else "normal",
            color="white" if r <= len(cells) * 0.4 else "#333333",
        )
        if kept:
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=1.2
                )
            )
    ax.set_xticks(range(len(cells)), cells, rotation=90, fontsize=8)
    ax.set_yticks(range(len(folds)), [f"fold {f}" for f in folds], fontsize=8)
    ax.set_title(
        f"Exercise rank per fold (1 = strongest; boxed = kept in top {k})", fontsize=9
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluator._save(fig, path, cfg)


def step_report(cfg=config) -> Path:
    """EXPLORATORY_v4.txt and the two figures."""
    _start(cfg)
    v4 = cfg.select_v4
    out = output_dir(cfg)
    metrics = pd.read_csv(out / "metrics.csv")
    paired = pd.read_csv(out / "paired.csv")
    ranking = _read_ranking(cfg)
    freq = pd.read_csv(out / "selection_frequency.csv")
    selected = pd.read_csv(out / "selected_cells.csv")
    by_key = metrics.set_index(["model", "feature_set"])
    model, fs = headline(cfg)
    k = v4.headline_k
    lines = [
        "EMHA v4 -- exercise selection",
        REPORT_HEADER,
        "",
        "1. Design",
        f"  Per outer fold, each of the {len(cfg.per_task_v3.cells)} exercises is "
        "scored on that fold's training participants only",
        f"  ({v4.rank_inner_folds}-fold out-of-fold ROC-AUC, mean of "
        f"{', '.join(v4.rank_kinds)}); the top k are kept.",
        f"  k grid: {', '.join(map(str, v4.k_grid))} (top"
        f"{len(cfg.per_task_v3.cells)} = no removal). Headline fixed in advance: "
        f"{model}/{fs}.",
        "",
        f"2. Pre-declared v4 test: {model}/{fs}",
        f"  {_metric_line(by_key.loc[(model, fs)])}",
        f"  permutation test: {_permutation_text(cfg)}",
        "  paired macro-F1 difference [CI]:",
    ]
    for _, r in paired.iterrows():
        if r["comparison"] == "headline_vs_v1" or r["model"] == f"{model}/{fs}":
            text = (
                r["note"]
                if isinstance(r["note"], str) and r["note"]
                else f"{r['delta_macro_f1']:+.3f} [{r['ci_low']:+.3f}, "
                f"{r['ci_high']:+.3f}]"
            )
            lines.append(f"    vs {r['reference']:<24} {text}")
    lines += ["", "3. Every model and k (descriptive)"]
    for m in v4_models(cfg):
        for kk in v4.k_grid:
            row = by_key.loc[(m, top_name(kk))]
            lines.append(f"  {m}/{top_name(kk)}: {_metric_line(row)}")
    lines += [
        "",
        f"4. Did removing exercises help? top{k} minus top"
        f"{len(cfg.per_task_v3.cells)}, same model (macro-F1 [CI])",
    ]
    for _, r in paired[paired["comparison"] == "removal"].iterrows():
        lines.append(
            f"  {r['model']:<28} {r['delta_macro_f1']:+.3f} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        )
    lines += [
        "",
        f"5. Which exercises were kept (top {k}), across "
        f"{ranking['fold'].nunique()} folds",
        f"  cell    task                      folds_in_top{k}  mean_rank  "
        "mean_score [min, max]",
    ]
    for _, r in freq.iterrows():
        style = r["style"] if isinstance(r["style"], str) else ""
        task = f"{r['task']} ({style})" if style else r["task"]
        lines.append(
            f"  {r['cell']:<7} {task:<25} {int(r[f'folds_in_top{k}']):>2}/"
            f"{int(r['n_folds'])}           {r['mean_rank']:>5.1f}      "
            f"{r['mean_score']:.3f} [{r['min_score']:.3f}, {r['max_score']:.3f}]"
        )
    lines += ["", f"  kept per fold (top {k}, best first):"]
    for fold in sorted(selected["fold"].unique()):
        lines.append(f"    fold {fold}: {', '.join(kept_cells(selected, k, fold))}")
    lines += [
        "",
        "  Reading: an exercise score of 0.5 is chance. An exercise kept in every",
        "  fold is consistently informative on training data; one kept in one",
        "  or two folds was probably picked by noise.",
        "",
        "6. " + LIMITATIONS,
    ]
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_k_curve(metrics, figure_path("k_curve.png", cfg), cfg)
    _plot_selection(ranking, figure_path("selection.png", cfg), cfg)
    for p in (path, figure_path("k_curve.png", cfg), figure_path("selection.png", cfg)):
        print(f"Written: {p}")
    return path


# ---------------------------------------------------------------- CLI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--draws", default=None, help="e.g. 0-9 or 3,5")
    parser.add_argument("--n-jobs", type=int, default=1, help="speed only")
    args = parser.parse_args()
    if args.step == "preflight":
        preflight(config)
    elif args.step == "rank":
        step_rank(config, n_jobs=args.n_jobs)
    elif args.step == "combine":
        step_combine(config)
    elif args.step == "evaluate":
        step_evaluate(config)
    elif args.step == "draws":
        step_draws(config, args.draws, n_jobs=args.n_jobs)
    elif args.step == "permutation":
        step_permutation(config)
    elif args.step == "report":
        step_report(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
