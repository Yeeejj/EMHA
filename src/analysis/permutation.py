"""
Participant-level permutation tests — Stage H2 (EVALUATION_PROTOCOL.md section 4).

Null: no association between handwriting and label. In every permutation and
outer fold, the labels of that fold's outer-train participants are shuffled
among them (src.training.run_cnn.shuffled_labels, seed
permutation_seed(seed, perm, fold)); the whole pipeline is refit on the
shuffled labels, and the fold's test participants are scored against their
TRUE labels. p = (1 + #{null macro-F1 >= observed}) / (1 + n).

LR models (PermutationConfig.lr_models: lr_handcrafted/all,
lr_embeddings/concat; n_lr = 1000; primary or full): run_baselines.fit_lr
(scaler [-> PCA] -> LR, C by inner CV on the outer-train participants) is
refit per fold exactly as in the observed run, in parallel with joblib. The
permuted label vectors are drawn in the parent process, so the workers only
fit.

CNN models (PermutationConfig.cnn_models: cnn_head_fused, cnn_hmm_fused, and
the ensemble; n_cnn = 100; primary only): FULL retrains, no proxy. One draw
retrains the fine-tuned CNN for every family and fold (run_cnn._fit_fold,
same epochs and early stopping on the shuffled inner-val labels), refits the
CNN-HMM on the word and cursive checkpoints (run_hybrid._fold: selection and
Platt calibration on the shuffled inner-val labels), refits the two ensemble
LRs on the same shuffled labels, and combines them (run_ensemble.combine).
One draw therefore gives the null value of all three CNN-path models. Its
participant predictions are cached as
results/final/permutation_runs/<analysis>/seed_<seed>/perm_<k>.csv, the
resume unit on Kaggle; the draw's CNN checkpoints live in a temporary
directory removed when the draw finishes.

The observed macro-F1 is read from the model's predictions.csv (the run H1
evaluates), pooled over the out-of-fold participants outside the middle band.

Outputs, results/final/:

    permutation_<model>_<analysis>_null.npy   null macro-F1 values
    permutation_<model>_<analysis>.png        histogram + observed value
    permutation_<model>_<analysis>.json       observed, p_value, n, seed, ...
    model_comparison.txt                      one line appended per test

    python -m src.analysis.permutation --model lr_handcrafted
        [--analysis primary] [--n 1000] [--n-jobs -1] [--seed 42]
        [--device auto]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from src.data.dataloader import labels_csv_path, load_tables
from src.features.embeddings import resolve_device
from src.training import run_cnn, run_ensemble, run_hybrid
from src.training.evaluator import analysis_frame, metrics, output_dir, validate
from src.training.run_baselines import PRED_COLUMNS, _prob_sad, feature_sets, fit_lr
from src.training.splits import (
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    folds_path,
    load_folds,
)
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic

plt.switch_backend("Agg")

SAD, HAPPY = "SAD", "HAPPY"
# Runner output folder of each tested model (observed predictions).
SOURCES = {
    "lr_handcrafted": "baselines",
    "lr_embeddings": "baselines",
    "cnn_head_fused": "cnn",
    "cnn_hmm_fused": "hybrid",
    "ensemble": "ensemble",
}
_SURFACE, _INK, _INK_2, _MUTED, _SERIES = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#2a78d6",
)
_PNG_META = {"Software": None}


def permutation_seed(seed: int, perm: int, fold: int) -> int:
    """Shuffle seed of one (permutation, fold); shared by the LR and CNN paths."""
    return int(np.random.SeedSequence([seed, perm, fold]).generate_state(1)[0])


def p_value(null, observed: float) -> float:
    """(1 + #{null >= observed}) / (1 + n)."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


# ─── context ──────────────────────────────────────────────────────────────────


@dataclass
class _Context:
    analysis: str
    labels: pd.DataFrame  # labels.csv, true labels
    ids: list
    folds: pd.DataFrame
    splits: list  # [(train_ids, test_ids)] per outer fold

    def true_index(self, ids, cfg=config) -> np.ndarray:
        return _label_index(self.labels, ids, cfg)


def _label_index(labels: pd.DataFrame, ids, cfg=config) -> np.ndarray:
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    return lab.loc[list(ids)].map(cfg.data.label_to_index).to_numpy()


def _context(analysis: str, cfg=config) -> _Context:
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    participants = pd.read_csv(
        cfg.paths.metadata_dir / "participants.csv", dtype={"participant_id": str}
    )
    folds = load_folds(cfg)
    ids = analysis_ids(labels, participants, analysis)
    if not ids:
        raise ValueError(f"no participants in analysis {analysis!r}")
    splits = []
    for fold in range(cfg.cv.n_splits):
        train, test = fold_ids(folds, fold, ids)
        assert_no_leakage(train, test)
        splits.append((train, test))
    return _Context(analysis, labels, ids, folds, splits)


def permuted_train_labels(ctx: _Context, perm: int, seed: int, cfg=config) -> list:
    """Per fold: label indices of the outer-train participants, shuffled."""
    return [
        _label_index(
            run_cnn.shuffled_labels(
                ctx.labels, train, permutation_seed(seed, perm, fold)
            ),
            train,
            cfg,
        )
        for fold, (train, _) in enumerate(ctx.splits)
    ]


# ─── observed ─────────────────────────────────────────────────────────────────


def observed_macro_f1(model: str, fs: str, analysis: str, cfg=config) -> float:
    """Pooled macro-F1 of the observed run, as the H1 evaluator computes it."""
    path = cfg.paths.results_dir / SOURCES[model] / analysis / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the {SOURCES[model]} runner")
    preds = validate(pd.read_csv(path, dtype={"participant_id": str}), analysis, path)
    rows = analysis_frame(preds, analysis)
    rows = rows[(rows["model"] == model) & (rows["feature_set"] == fs)]
    if set(rows["fold"]) != set(range(cfg.cv.n_splits)):
        raise ValueError(f"{path}: {model}/{fs} does not cover all folds")
    return metrics(rows)["macro_f1"]


# ─── LR path ──────────────────────────────────────────────────────────────────


def _lr_features(model: str, fs: str, ids: list, cfg=config) -> tuple:
    for m, name, X, use_pca in feature_sets(cfg):
        if (m, name) == (model, fs):
            missing = set(ids) - set(X.index)
            if missing:
                raise ValueError(f"{model}/{fs}: no features for {sorted(missing)[:5]}")
            return X, use_pca
    raise ValueError(f"no feature set {model}/{fs}")


def _lr_probs(X, plan, y_trains, use_pca, cfg) -> list:
    """prob_sad of every fold's test rows, LR refit as in run_baselines."""
    out = []
    for fold, (train_rows, test_rows) in enumerate(plan):
        pipe = fit_lr(
            X[train_rows],
            y_trains[fold],
            cfg,
            pca=use_pca,
            seed=cfg.training.seed + fold,
        )
        out.append(_prob_sad(pipe, X[test_rows]))
    return out


def _lr_null_one(X, plan, y_trains, y_test_sad, use_pca, cfg) -> float:
    prob = np.concatenate(_lr_probs(X, plan, y_trains, use_pca, cfg))
    pred = prob >= cfg.aggregate.threshold
    return float(
        f1_score(
            y_test_sad, pred, labels=[False, True], average="macro", zero_division=0
        )
    )


def _lr_plan(X: pd.DataFrame, ctx: _Context) -> list:
    pos = {pid: i for i, pid in enumerate(X.index.astype(str))}
    return [
        (np.array([pos[p] for p in train]), np.array([pos[p] for p in test]))
        for train, test in ctx.splits
    ]


def lr_null(
    model: str,
    fs: str,
    analysis: str,
    n: int,
    n_jobs: int,
    seed: int,
    cfg=config,
) -> np.ndarray:
    """Null macro-F1 of an LR model over permutations 0..n-1."""
    ctx = _context(analysis, cfg)
    X_df, use_pca = _lr_features(model, fs, ctx.ids, cfg)
    X = X_df.to_numpy(dtype=float)
    plan = _lr_plan(X_df, ctx)
    sad = cfg.data.label_to_index[SAD]
    y_test_sad = np.concatenate([ctx.true_index(t, cfg) for _, t in ctx.splits]) == sad
    jobs = (
        delayed(_lr_null_one)(
            X,
            plan,
            permuted_train_labels(ctx, perm, seed, cfg),
            y_test_sad,
            use_pca,
            cfg,
        )
        for perm in range(n)
    )
    return np.asarray(Parallel(n_jobs=n_jobs)(jobs), dtype=float)


def lr_predictions(
    model: str, fs: str, ctx: _Context, y_trains: list | None, cfg=config
) -> pd.DataFrame:
    """Out-of-fold predictions (shared schema); y_trains None = true labels."""
    X_df, use_pca = _lr_features(model, fs, ctx.ids, cfg)
    plan = _lr_plan(X_df, ctx)
    if y_trains is None:
        y_trains = [ctx.true_index(train, cfg) for train, _ in ctx.splits]
    probs = _lr_probs(X_df.to_numpy(dtype=float), plan, y_trains, use_pca, cfg)
    frames = []
    col = cfg.labeling.label_column
    label_of = ctx.labels.set_index("participant_id")[col]
    for fold, ((_, test), prob) in enumerate(zip(ctx.splits, probs)):
        frames.append(
            pd.DataFrame(
                {
                    "analysis": ctx.analysis,
                    "model": model,
                    "feature_set": fs,
                    "fold": fold,
                    "participant_id": test,
                    "label": label_of.loc[test].to_numpy(),
                    "prob_sad": prob,
                    "pred": np.where(prob >= cfg.aggregate.threshold, SAD, HAPPY),
                    "in_middle_band": False,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[list(PRED_COLUMNS)]


# ─── CNN path ─────────────────────────────────────────────────────────────────


def runs_dir(analysis: str, seed: int, cfg=config) -> Path:
    return (
        output_dir(None, cfg) / cfg.permutation.runs_subdir / analysis / f"seed_{seed}"
    )


def _settings(analysis: str, seed: int, cfg=config) -> dict:
    return json.loads(
        json.dumps(
            {
                "analysis": analysis,
                "seed": seed,
                "folds_sha256": hashlib.sha256(
                    folds_path(cfg).read_bytes()
                ).hexdigest(),
                "cnn": asdict(cfg.cnn),
                "training": asdict(cfg.training),
                "hmm": asdict(cfg.hmm),
                "hybrid": asdict(cfg.hybrid),
                "baselines": asdict(cfg.baselines),
                "aggregate": asdict(cfg.aggregate),
            },
            default=str,
        )
    )


def cnn_null_run(ctx: _Context, perm: int, seed: int, device: str, cfg=config):
    """One full-retrain draw: participant predictions of every CNN-path model."""
    manifest, _, dropped = load_tables(cfg)
    lr_y = permuted_train_labels(ctx, perm, seed, cfg)
    head, hmm = [], []
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cfg.paths.models_dir) as tmp:
        for fold in range(cfg.cv.n_splits):
            shuffle_seed = permutation_seed(seed, perm, fold)
            for family in run_cnn.FAMILIES:
                ckpt = Path(tmp) / family / f"fold_{fold}.pth"
                crops = run_cnn._fit_fold(
                    family,
                    fold,
                    ctx.ids,
                    [],
                    ctx.folds,
                    ctx.labels,
                    manifest,
                    dropped,
                    cfg.cnn,
                    shuffle_seed,
                    device,
                    ckpt,
                )
                head.append(
                    crops.assign(analysis=ctx.analysis, model=run_cnn.MODEL)[
                        list(run_cnn.CROP_COLUMNS)
                    ]
                )
                if family in cfg.hybrid.families:
                    crops, _ = run_hybrid._fold(
                        family,
                        fold,
                        ctx.ids,
                        [],
                        ctx.folds,
                        ctx.labels,
                        manifest,
                        dropped,
                        f"perm_{perm}",
                        device,
                        ckpt=ckpt,
                        shuffle_seed=shuffle_seed,
                        save=False,
                    )
                    hmm.append(
                        crops.assign(analysis=ctx.analysis, model=run_hybrid.MODEL)
                    )
    head = run_cnn._participant_predictions(pd.concat(head), ctx.analysis)
    hmm = run_hybrid._predictions(pd.concat(hmm), ctx.analysis)
    components = {
        (m, fs): lr_predictions(m, fs, ctx, lr_y, cfg)
        for m, fs in cfg.permutation.lr_models
    }
    for frame in (head, hmm):
        for (m, fs), part in frame.groupby(["model", "feature_set"]):
            components[(m, fs)] = part
    needed = [(m, fs) for m, fs, _ in run_ensemble.COMPONENTS]
    ensemble = run_ensemble.combine([components[k] for k in needed])
    keep = list(cfg.permutation.lr_models) + list(cfg.permutation.cnn_models)
    frames = [components[k] for k in keep if k in components] + [ensemble]
    return pd.concat(frames, ignore_index=True)[list(PRED_COLUMNS)].assign(perm=perm)


def cnn_null(
    model: str, fs: str, analysis: str, n: int, seed: int, device: str, cfg=config
) -> np.ndarray:
    """Null macro-F1 of a CNN-path model; draws are cached and resumed."""
    ctx = _context(analysis, cfg)
    out = runs_dir(analysis, seed, cfg)
    run_cnn.check_resume(out, _settings(analysis, seed, cfg))
    null = []
    for perm in range(n):
        path = out / f"perm_{perm:04d}.csv"
        if path.is_file():
            print(f"resume: permutation {perm} already done")
        else:
            print(f"permutation {perm + 1}/{n}: full retrain ({device})")
            frame = cnn_null_run(ctx, perm, seed, device, cfg)
            tmp = path.with_suffix(".tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(path)
        frame = pd.read_csv(path, dtype={"participant_id": str})
        rows = frame[(frame["model"] == model) & (frame["feature_set"] == fs)]
        null.append(metrics(rows)["macro_f1"])
    return np.asarray(null, dtype=float)


# ─── outputs ──────────────────────────────────────────────────────────────────


def _plot(null, observed, p, title: str, path: Path, cfg=config) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.hist(null, bins=cfg.permutation.hist_bins, color=_SERIES, rwidth=0.9)
    ax.axvline(observed, color=_INK, linewidth=1.5)
    ax.annotate(
        f"observed {observed:.3f}\np = {p:.4f}",
        xy=(observed, ax.get_ylim()[1] * 0.95),
        xytext=(4, 0),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color=_INK,
    )
    ax.set(xlabel="Null macro-F1 (shuffled training labels)", ylabel="Permutations")
    ax.set_title(title, fontsize=9, color=_INK)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)
    ax.tick_params(colors=_INK_2, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.evaluation.dpi, metadata=_PNG_META)
    plt.close(fig)


def permutation_test(
    model: str,
    analysis: str = "primary",
    n: int | None = None,
    n_jobs: int = -1,
    seed: int | None = None,
    device: str = "auto",
    cfg=config,
) -> dict:
    """Permutation p-value of the model's pooled macro-F1; writes the outputs."""
    require_frozen_or_synthetic(cfg, "permutation")
    pcfg = cfg.permutation
    seed = cfg.training.seed if seed is None else seed
    lr, cnn = dict(pcfg.lr_models), dict(pcfg.cnn_models)
    if model in lr:
        fs, n = lr[model], n or pcfg.n_lr
        method = "LR: nested-CV pipeline refit on shuffled outer-train labels"
        print(f"Permutation ({method}), N = {n}, n_jobs = {n_jobs}")
        null = lr_null(model, fs, analysis, n, n_jobs, seed, cfg)
    elif model in cnn:
        if analysis not in pcfg.cnn_analyses:
            raise ValueError(
                f"CNN-path permutation tests run on {pcfg.cnn_analyses} only "
                "(EVALUATION_PROTOCOL.md section 4)"
            )
        fs, n = cnn[model], n or pcfg.n_cnn
        method = (
            "CNN: full retrains (EVALUATION_PROTOCOL.md section 4; no epoch-capped "
            "proxy), CNN + CNN-HMM + ensemble LRs refit on shuffled labels"
        )
        print(f"Permutation ({method}), N = {n}; n_jobs unused, draws run in turn")
        null = cnn_null(model, fs, analysis, n, seed, resolve_device(device), cfg)
    else:
        tested = sorted(lr) + sorted(cnn)
        raise ValueError(f"no permutation test for {model!r}; protocol tests {tested}")

    observed = observed_macro_f1(model, fs, analysis, cfg)
    p = p_value(null, observed)
    out = output_dir(None, cfg)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"permutation_{model}_{analysis}"
    null_path = out / f"{stem}_null.npy"
    np.save(null_path, null)
    _plot(
        null,
        observed,
        p,
        f"Permutation test, {model} / {fs} ({analysis} set, N = {n})",
        out / f"{stem}.png",
        cfg,
    )
    result = {
        "model": model,
        "feature_set": fs,
        "analysis": analysis,
        "n": int(n),
        "seed": int(seed),
        "method": method,
        "observed": float(observed),
        "p_value": p,
        "null_mean": float(np.mean(null)),
        "significant": bool(p <= pcfg.alpha),
        "null_path": str(null_path),
    }
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
    line = (
        f"Permutation test, {analysis} {model}/{fs}: observed macro-F1 "
        f"{observed:.3f}, p = {p:.4f} (N = {n}, seed {seed}, null mean "
        f"{result['null_mean']:.3f}; {method})\n"
    )
    with (out / "model_comparison.txt").open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Permutation test (Stage H2).")
    tested = [
        m for m, _ in config.permutation.lr_models + config.permutation.cnn_models
    ]
    parser.add_argument("--model", required=True, choices=tested)
    parser.add_argument("--analysis", choices=("primary", "full"), default="primary")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        permutation_test(
            args.model, args.analysis, args.n, args.n_jobs, args.seed, args.device
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
