"""
CNN-HMM v2 (task order) — EXPLORATORY, post hoc (EVALUATION_PROTOCOL.md
section 9, Deviation 5). Never part of the primary claim, the ensemble, or
any v1 output; the primary model stays CNN-HMM v1 (run_hybrid.py).

Each primary-set participant is one sequence of 24 frozen ImageNet ResNet18
embeddings (results/embeddings/resnet18_<family>.npz, 512 values per task
image), in FINALE form order (src.data.crop_manifest.ALL_CELLS). A missing
crop stops the run; nothing is imputed. Per outer fold (folds.csv):

  1. Inner split of the outer-train participants exactly as run_hybrid
     (inner_split, seed TrainingConfig.seed + fold).
  2. Select the HMM setting (HybridV2Config grids) by inner-val participant
     macro-F1 of the uncalibrated sigmoid of the decision score; each
     candidate is fit on inner-train with HybridConfig.selection_restarts
     restarts (run_hybrid's rule; ties keep grid order).
  3. Refit the winner on inner-train with HMMConfig.n_restarts, Platt-
     calibrate it on inner-val, and score the outer test participants:
     SAD when P(SAD) >= AggregateConfig.threshold.

Scaler and PCA live inside HMMClassifier and are fit on inner-train frames
only; outer-test participants are only scored. run_hybrid.py, hmm.py,
HybridConfig and HMMConfig are imported, never changed: select_hmm and
fit_final are copied here because v2 has its own grid and its permutation
draws fit serially.

Outputs (v2 paths only):

    models/hmm_v2/primary/fold_<k>.joblib
    results/hybrid_v2/primary/parts/fold_<k>.{csv,json}  per-fold cache (resume)
    results/hybrid_v2/primary/predictions.csv       cnn_hmm_v2 / task_order
    results/hybrid_v2/primary/selected_settings.csv chosen setting per fold
    results/hybrid_v2/primary/selection.json        every grid row per fold
    results/hybrid_v2/primary/run_config.json       a rerun refuses a mismatch
    results/hybrid_v2/primary/metrics.csv, per_fold.csv, confusion.csv,
        paired.csv                                   src.training.evaluator
    results/hybrid_v2/primary/permutation.json, permutation_null_v2.png
    results/final/permutation_runs/primary_v2/seed_<s>/perm_<k>.json
    results/final/EXPLORATORY_v2.txt

Permutation: HybridV2Config.n_permutations draws; in each, the outer-train
labels of every fold are shuffled among themselves (run_cnn.shuffled_labels,
permutation_seed(seed, draw, fold) as in src/analysis/permutation.py) and v2
is refit in full; test participants keep their true labels. Draws run in
parallel processes, one BLAS/OpenMP thread each (Deviation 2), and each is
cached, so a stopped run resumes. p = (1 + #{null >= observed}) / (1 + N).

    python -m src.training.run_hybrid_v2 [--n-jobs N]                 fit
    python -m src.training.run_hybrid_v2 --permutation [--n-jobs N]   null

Nothing is appended to results/RUN_LOG.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from src.analysis.permutation import SOURCES, _plot, p_value, permutation_seed
from src.data.crop_manifest import ALL_CELLS
from src.data.dataloader import labels_csv_path
from src.models.hmm import (
    PARALLEL_BACKEND,
    HMMClassifier,
    candidate_scores,
    replay,
    run_isolated,
)
from src.models.hmm import logger as hmm_logger
from src.training import evaluator
from src.training.run_baselines import PRED_COLUMNS
from src.training.run_cnn import shuffled_labels
from src.training.run_hybrid import SequenceSet, _participant_macro_f1
from src.training.splits import (
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    folds_path,
    inner_split,
    load_folds,
)
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic
from src.utils.seed import set_seed

MODEL = "cnn_hmm_v2"
FEATURE_SET = "task_order"
LABEL = "CNN-HMM v2 (task order) — exploratory"
V1_LABEL = "CNN-HMM v1 (column sequence) — primary"
REPORT_HEADER = "EXPLORATORY, POST HOC (Deviation 5). Not part of the primary claim."
ANALYSIS = "primary"
FAMILIES = ("drawing", "word", "cursive")
METRIC_NAMES = {
    "macro_f1": "macro-F1",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced accuracy",
    "roc_auc": "ROC-AUC",
}


def output_dir(cfg=config) -> Path:
    return cfg.paths.results_dir / cfg.hybrid_v2.output_subdir / ANALYSIS


def model_path(fold: int, cfg=config) -> Path:
    return (
        cfg.paths.models_dir
        / cfg.hybrid_v2.model_subdir
        / ANALYSIS
        / (f"fold_{fold}.joblib")
    )


def report_path(cfg=config) -> Path:
    return evaluator.output_dir(None, cfg) / cfg.hybrid_v2.report_name


def runs_dir(seed: int, cfg=config) -> Path:
    out = evaluator.output_dir(None, cfg) / cfg.permutation.runs_subdir
    return out / cfg.hybrid_v2.runs_name / f"seed_{seed}"


def embedding_path(family: str, cfg=config) -> Path:
    prefix = cfg.hybrid_v2.embedding_prefix
    return cfg.paths.results_dir / "embeddings" / f"{prefix}_{family}.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sequences(ids: list, cfg=config) -> tuple:
    """(ids, array (n, 24, 512)): each row is one participant in ALL_CELLS order.

    Raises if any participant lacks a cell or a crop appears twice.
    """
    rows = {}
    for family in FAMILIES:
        path = embedding_path(family, cfg)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run src.features.embeddings")
        with np.load(path, allow_pickle=False) as z:
            emb = z["embeddings"]
            keys = zip(z["participant_id"].astype(str), z["cell"].astype(str))
            for i, key in enumerate(keys):
                if key in rows:
                    raise ValueError(f"{path}: crop {key} appears twice")
                rows[key] = emb[i]
    ids = [str(p) for p in ids]
    missing = [(p, c) for p in ids for c in ALL_CELLS if (p, c) not in rows]
    if missing:
        raise ValueError(
            f"{len(missing)} crops missing from the embedding caches (not "
            f"imputed), e.g. {missing[:5]}"
        )
    array = np.stack([np.stack([rows[(p, c)] for c in ALL_CELLS]) for p in ids])
    return ids, array.astype(float)


def _context(cfg=config) -> tuple:
    """(labels.csv, primary-set IDs, folds.csv)."""
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    participants = pd.read_csv(
        cfg.paths.metadata_dir / "participants.csv", dtype={"participant_id": str}
    )
    return labels, analysis_ids(labels, participants, ANALYSIS), load_folds(cfg)


def _set(ids: list, labels: pd.DataFrame, seqs: dict, cfg=config) -> SequenceSet:
    """One sequence per participant; meta in run_hybrid's SequenceSet form."""
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    meta = pd.DataFrame(
        {
            "participant_id": [str(p) for p in ids],
            "task_family": FEATURE_SET,
            "label": lab.loc[[str(p) for p in ids]].to_numpy(),
        }
    )
    meta["label_index"] = meta["label"].map(cfg.data.label_to_index)
    return SequenceSet(meta, [seqs[str(p)] for p in ids])


def _grid(cfg=config) -> list:
    h = cfg.hybrid_v2
    return [
        {"n_states": n, "pca_components": p, "topology": t}
        for n, p, t in itertools.product(h.n_states_grid, h.pca_grid, h.topology_grid)
    ]


def select_hmm(
    train: SequenceSet, val: SequenceSet, seed: int, n_jobs: int | None = None
) -> tuple:
    """run_hybrid.select_hmm on the v2 grid: (winner, table of all candidates)."""
    n_jobs = config.runner.hmm_n_jobs if n_jobs is None else n_jobs
    threads = config.runner.hmm_threads_per_fit
    grid = _grid()
    level = hmm_logger.getEffectiveLevel()
    outcomes = Parallel(n_jobs=n_jobs, backend=PARALLEL_BACKEND)(
        delayed(run_isolated)(
            candidate_scores,
            (
                replace(
                    config.hmm, n_restarts=config.hybrid.selection_restarts, **setting
                ),
                seed,
                train.seqs,
                train.labels,
                val.seqs,
                threads,
            ),
            threads,
            level,
        )
        for setting in grid
    )
    table, best = [], None
    for setting, (scores, records, caught) in zip(grid, outcomes):
        replay(records, caught)
        try:
            if isinstance(scores, Exception):
                raise scores
            f1 = _participant_macro_f1(val.meta, 1.0 / (1.0 + np.exp(-scores)))
            table.append({**setting, "inner_val_macro_f1": f1})
        except (ValueError, RuntimeError) as exc:
            table.append({**setting, "inner_val_macro_f1": None, "error": str(exc)})
            continue
        if best is None or f1 > best["inner_val_macro_f1"]:
            best = {**setting, "inner_val_macro_f1": f1}
    if best is None:
        errors = sorted({row["error"] for row in table if "error" in row})
        raise RuntimeError(f"no HMM setting could be fit: {errors[:3]}")
    return best, table


def fit_final(
    train: SequenceSet,
    val: SequenceSet,
    setting: dict,
    seed: int,
    n_jobs: int | None = None,
) -> HMMClassifier:
    """run_hybrid.fit_final with n_jobs: refit on train, Platt on val."""
    n_jobs = config.runner.hmm_n_jobs if n_jobs is None else n_jobs
    params = {k: setting[k] for k in ("n_states", "pca_components", "topology")}
    clf = HMMClassifier(replace(config.hmm, **params), seed, n_jobs=n_jobs)
    clf.fit(train.seqs, train.labels)
    clf.fit_calibrator(clf.decision_scores(val.seqs), val.labels)
    return clf


def run_fold(
    fold: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    seqs: dict,
    shuffle_seed: int | None = None,
    n_jobs: int | None = None,
) -> tuple:
    """(test predictions, selection record, fitted classifier) for one fold.

    With shuffle_seed the outer-train labels are permuted among themselves
    before fitting (the inner split is drawn on the true labels, as in
    run_hybrid); test participants always keep their true labels.
    """
    train, test = fold_ids(folds, fold, ids)
    assert_no_leakage(train, test)
    seed = config.training.seed + fold
    inner_train, inner_val = inner_split(
        train, labels, config.cv.inner_val_fraction, seed
    )
    assert_no_leakage(inner_train, inner_val, test)
    fit_labels = (
        labels if shuffle_seed is None else shuffled_labels(labels, train, shuffle_seed)
    )
    set_seed(seed)
    tr = _set(inner_train, fit_labels, seqs)
    va = _set(inner_val, fit_labels, seqs)
    best, table = select_hmm(tr, va, seed, n_jobs)
    clf = fit_final(tr, va, best, seed, n_jobs)

    te = _set(test, labels, seqs)
    sad = config.data.label_to_index["SAD"]
    prob = clf.predict_proba(te.seqs)[:, list(clf.classes_).index(sad)]
    preds = te.meta[["participant_id", "label"]].assign(
        analysis=ANALYSIS,
        model=MODEL,
        feature_set=FEATURE_SET,
        fold=fold,
        prob_sad=prob,
        pred=np.where(prob >= config.aggregate.threshold, "SAD", "HAPPY"),
        in_middle_band=False,
    )[list(PRED_COLUMNS)]
    selection = {
        "fold": fold,
        "chosen": best,
        "grid": table,
        "refit_restarts": config.hmm.n_restarts,
        "restart_log": clf.restart_log,
        "n_inner_train": len(inner_train),
        "n_inner_val": len(inner_val),
        "n_test": len(test),
    }
    return preds, selection, clf


def settings(cfg=config) -> dict:
    """Everything a rerun must match (run_config.json)."""
    return json.loads(
        json.dumps(
            {
                "model": MODEL,
                "feature_set": FEATURE_SET,
                "analysis": ANALYSIS,
                "cells": list(ALL_CELLS),
                "hybrid_v2": asdict(cfg.hybrid_v2),
                "hmm": asdict(cfg.hmm),
                "selection_restarts": cfg.hybrid.selection_restarts,
                "seed": cfg.training.seed,
                "inner_val_fraction": cfg.cv.inner_val_fraction,
                "threshold": cfg.aggregate.threshold,
                "folds_sha256": _sha256(folds_path(cfg)),
                "labels_sha256": _sha256(labels_csv_path(cfg)),
                "embeddings_sha256": {
                    f: _sha256(embedding_path(f, cfg)) for f in FAMILIES
                },
            },
            default=str,
        )
    )


def _check_settings(path: Path, want: dict) -> bool:
    """True if path holds want; False if absent; raises on a mismatch."""
    if not path.is_file():
        return False
    if json.loads(path.read_text(encoding="utf-8")) != want:
        raise RuntimeError(
            f"{path} records different settings; v2 runs once (Deviation 5). "
            "A bug fix is logged under Deviation 5 and the old outputs are "
            "moved aside deliberately before rerunning."
        )
    return True


def run(n_jobs: int | None = None, cfg=config) -> Path:
    """Fit v2 on every outer fold, then evaluate and write the report."""
    require_frozen_or_synthetic(cfg, "run_hybrid_v2")
    labels, ids, folds = _context(cfg)
    order, array = load_sequences(ids, cfg)
    seqs = dict(zip(order, array))
    out = output_dir(cfg)
    pred_path = out / "predictions.csv"
    want = settings(cfg)
    if _check_settings(out / "run_config.json", want) and pred_path.is_file():
        print(f"v2 already fitted with these settings: {pred_path}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps(want, indent=2))
        parts = out / "parts"
        parts.mkdir(exist_ok=True)
        frames, selections = [], []
        for fold in range(cfg.cv.n_splits):
            part, record = parts / f"fold_{fold}.csv", parts / f"fold_{fold}.json"
            if part.is_file() and record.is_file() and model_path(fold, cfg).is_file():
                frames.append(
                    pd.read_csv(
                        part,
                        dtype={"participant_id": str},
                        float_precision="round_trip",
                    )
                )
                selections.append(json.loads(record.read_text()))
                print(f"fold {fold}: cached ({part.name})", flush=True)
                continue
            t0 = time.perf_counter()
            preds, selection, clf = run_fold(
                fold, ids, labels, folds, seqs, None, n_jobs
            )
            clf.save(model_path(fold, cfg))
            preds.to_csv(part, index=False)
            record.write_text(json.dumps(selection, indent=2, default=str))
            frames.append(preds)
            selections.append(selection)
            took = time.perf_counter() - t0
            print(
                f"fold {fold}: {took:.1f} s, chosen {selection['chosen']}", flush=True
            )
            if fold == 0:
                fit = took * cfg.cv.n_splits
                null = fit * cfg.hybrid_v2.n_permutations
                print(
                    f"fold 0 elapsed {took:.1f} s; estimated (a) main v2 run "
                    f"~{fit / 60:.1f} min, (b) permutation N = "
                    f"{cfg.hybrid_v2.n_permutations} ~{null / 3600:.1f} h, "
                    f"(a) + (b) ~{(fit + null) / 3600:.1f} h",
                    flush=True,
                )
        preds = pd.concat(frames, ignore_index=True)
        evaluator.validate(preds, ANALYSIS, pred_path)
        preds.to_csv(pred_path, index=False)
        pd.DataFrame([{"fold": s["fold"], **s["chosen"]} for s in selections]).to_csv(
            out / "selected_settings.csv", index=False
        )
        (out / "selection.json").write_text(
            json.dumps(selections, indent=2, default=str)
        )
    evaluate(cfg)
    return write_report(cfg)


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    df = evaluator.validate(
        pd.read_csv(path, dtype={"participant_id": str}), ANALYSIS, path
    )
    return evaluator.analysis_frame(df, ANALYSIS)


def v2_frame(cfg=config) -> pd.DataFrame:
    return _read_predictions(output_dir(cfg) / "predictions.csv")


def reference_frames(cfg=config) -> dict:
    """(model, feature_set) -> primary-set predictions of each paired reference."""
    out = {}
    for model, fs in cfg.hybrid_v2.paired_references:
        path = cfg.paths.results_dir / SOURCES[model] / ANALYSIS / "predictions.csv"
        df = _read_predictions(path)
        out[(model, fs)] = df[(df["model"] == model) & (df["feature_set"] == fs)]
    return out


def evaluate(cfg=config) -> evaluator.Evaluation:
    """Pooled metrics, bootstrap CIs, per fold, confusion, paired differences."""
    refs = reference_frames(cfg)
    result = evaluator.evaluate(pd.concat([v2_frame(cfg), *refs.values()]), cfg)
    key = (ANALYSIS, MODEL, FEATURE_SET)
    if key not in result.groups:
        raise ValueError(f"v2 predictions incomplete: {result.skipped}")
    out = output_dir(cfg)
    mine = result.comparison["model"] == MODEL
    result.comparison[mine].to_csv(out / "metrics.csv", index=False)
    result.per_fold[result.per_fold["model"] == MODEL].to_csv(
        out / "per_fold.csv", index=False
    )
    result.confusion[result.confusion["model"] == MODEL].to_csv(
        out / "confusion.csv", index=False
    )
    ev = cfg.evaluation
    paired = []
    for model, fs in refs:
        diff, lo, hi = evaluator.paired_diff_ci(
            result.groups[key],
            result.groups[(ANALYSIS, model, fs)],
            "macro_f1",
            ev.n_bootstrap,
            cfg.training.seed,
        )
        paired.append(
            {
                "reference": f"{model}/{fs}",
                "delta_macro_f1": diff,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    return result


def _draw(perm: int, seed: int, ids, labels, folds, order, array) -> dict:
    """Null macro-F1 of one draw: v2 refit in full on shuffled outer-train labels."""
    seqs = dict(zip(order, array))
    with threadpool_limits(limits=config.runner.hmm_threads_per_fit):
        frames = [
            run_fold(
                fold,
                ids,
                labels,
                folds,
                seqs,
                shuffle_seed=permutation_seed(seed, perm, fold),
                n_jobs=1,
            )[0]
            for fold in range(config.cv.n_splits)
        ]
    return {"perm": perm, "macro_f1": evaluator.metrics(pd.concat(frames))["macro_f1"]}


def permutation(
    n: int | None = None,
    seed: int | None = None,
    n_jobs: int | None = None,
    cfg=config,
) -> dict:
    """N full-refit draws (cached per draw, resumable), p-value, null plot."""
    require_frozen_or_synthetic(cfg, "run_hybrid_v2 permutation")
    n = cfg.hybrid_v2.n_permutations if n is None else n
    seed = cfg.training.seed if seed is None else seed
    n_jobs = cfg.runner.hmm_n_jobs if n_jobs is None else n_jobs
    observed = evaluator.metrics(v2_frame(cfg))["macro_f1"]
    labels, ids, folds = _context(cfg)
    order, array = load_sequences(ids, cfg)
    runs = runs_dir(seed, cfg)
    if not _check_settings(runs / "settings.json", settings(cfg)):
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "settings.json").write_text(json.dumps(settings(cfg), indent=2))
    todo = [k for k in range(n) if not (runs / f"perm_{k}.json").is_file()]
    print(f"permutation: {n - len(todo)} of {n} draws cached, {len(todo)} to run")
    start = time.perf_counter()
    draws = Parallel(
        n_jobs=n_jobs, backend=PARALLEL_BACKEND, return_as="generator_unordered"
    )(delayed(_draw)(k, seed, ids, labels, folds, order, array) for k in todo)
    for done, draw in enumerate(draws, 1):
        (runs / f"perm_{draw['perm']}.json").write_text(json.dumps(draw))
        if done == 1 or done % 10 == 0 or done == len(todo):
            elapsed = time.perf_counter() - start
            left = elapsed / done * (len(todo) - done)
            print(
                f"  {done}/{len(todo)} draws, {elapsed / 60:.1f} min elapsed, "
                f"~{left / 60:.1f} min left",
                flush=True,
            )
    null = np.array(
        [
            json.loads((runs / f"perm_{k}.json").read_text())["macro_f1"]
            for k in range(n)
        ]
    )
    p = p_value(null, observed)
    result = {
        "model": MODEL,
        "feature_set": FEATURE_SET,
        "analysis": ANALYSIS,
        "observed_macro_f1": observed,
        "p_value": p,
        "n": n,
        "seed": seed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if n > 1 else 0.0,
        "runs_dir": str(runs),
    }
    out = output_dir(cfg)
    (out / "permutation.json").write_text(json.dumps(result, indent=2))
    _plot(
        null,
        observed,
        p,
        f"{LABEL}: permutation null (N = {n})",
        out / "permutation_null_v2.png",
        cfg,
    )
    write_report(cfg)
    return result


def _ci(row, m: str) -> str:
    return f"{row[m]:.3f} [{row[f'{m}_ci_low']:.3f}, {row[f'{m}_ci_high']:.3f}]"


def write_report(cfg=config) -> Path:
    """results/final/EXPLORATORY_v2.txt from the v2 output files only."""
    out = output_dir(cfg)
    ev = cfg.evaluation
    row = pd.read_csv(out / "metrics.csv").iloc[0]
    per_fold = pd.read_csv(out / "per_fold.csv")
    conf = pd.read_csv(out / "confusion.csv").iloc[0]
    paired = pd.read_csv(out / "paired.csv")
    chosen = pd.read_csv(out / "selected_settings.csv", dtype=str)
    lines = [
        REPORT_HEADER,
        LABEL,
        f"Model {MODEL} / {FEATURE_SET}, {ANALYSIS} set, pooled out-of-fold, "
        f"n = {int(row['n_participants'])} ({int(row['n_sad'])} SAD, "
        f"{int(row['n_happy'])} HAPPY); SAD when P(SAD) >= "
        f"{cfg.aggregate.threshold}.",
        f"The primary model remains {V1_LABEL} (cnn_hmm_fused/fused; "
        "RESULTS.txt, EVALUATION_PROTOCOL.md section 9, Deviation 4).",
        f"CIs: {ev.n_bootstrap} participant bootstrap resamples, seed "
        f"{cfg.training.seed}, {round(ev.ci_level * 100)}% percentile.",
        "",
        "1. Pooled metrics [CI]",
    ]
    lines += [f"  {name:18s} {_ci(row, m)}" for m, name in METRIC_NAMES.items()]
    lines += [f"  {'majority baseline':18s} {row['majority_baseline']:.3f}", ""]
    lines += [
        "2. Per fold",
        "  fold    n  macro-F1  accuracy  balanced acc.  ROC-AUC  majority",
    ]
    for _, f in per_fold.iterrows():
        lines.append(
            f"  {int(f['fold']):>4d}  {int(f['n_participants']):>3d}  "
            f"{f['macro_f1']:.3f}     {f['accuracy']:.3f}     "
            f"{f['balanced_accuracy']:.3f}          {f['roc_auc']:.3f}    "
            f"{f['majority_baseline']:.3f}"
        )
    lines += [
        "",
        "3. Confusion matrix (rows = true label, columns = predicted)",
        "               pred HAPPY  pred SAD",
        f"  true HAPPY   {int(conf['tn']):>10d}  {int(conf['fp']):>8d}",
        f"  true SAD     {int(conf['fn']):>10d}  {int(conf['tp']):>8d}",
        "",
        "4. Paired macro-F1 difference, v2 - reference [CI] (same resamples)",
    ]
    for _, r in paired.iterrows():
        lines.append(
            f"  {r['reference']:22s} {r['delta_macro_f1']:+.3f} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        )
    lines += ["", "5. Permutation test (full v2 refit per draw)"]
    perm_path = out / "permutation.json"
    if perm_path.is_file():
        perm = json.loads(perm_path.read_text())
        lines.append(
            f"  observed macro-F1 {perm['observed_macro_f1']:.3f}, "
            f"p = {perm['p_value']:.4f} (N = {perm['n']}, seed {perm['seed']}, "
            f"null mean {perm['null_mean']:.3f})"
        )
    else:
        lines.append("  not run yet")
    lines += ["", "6. HMM setting chosen per fold (inner-validation macro-F1)"]
    for _, c in chosen.iterrows():
        lines.append(
            f"  fold {c['fold']}: n_states {c['n_states']}, PCA "
            f"{c['pca_components']}, {c['topology']} "
            f"({float(c['inner_val_macro_f1']):.3f})"
        )
    rel = out.relative_to(cfg.paths.results_dir.parent).as_posix()
    lines += ["", f"Sources: {rel}/ (metrics, per_fold, confusion, paired, "]
    lines[-1] += "selected_settings, permutation)."
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CNN-HMM v2 (task order), exploratory (Deviation 5)."
    )
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.permutation:
        permutation(n_jobs=args.n_jobs)
    else:
        run(n_jobs=args.n_jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
