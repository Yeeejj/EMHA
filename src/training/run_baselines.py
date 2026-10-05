"""
Majority and logistic-regression baselines inside the frozen outer folds — Stage F.

Models (model / feature_set):

    majority        all       training-fold SAD share as prob_sad
    lr_handcrafted  all, drawing, word, cursive
                              results/features/handcrafted_participant.csv
    lr_embeddings   concat, drawing, word, cursive
                              participant-mean frozen ResNet18 embedding per
                              family (results/embeddings/resnet18_*.npz);
                              concat = the three family means side by side
    lr_handwriting  concat    EXPLORATORY: TrOCR word + cursive embeddings,
                              only if handwriting_*.npz exist; never part of
                              any confirmatory claim or the ensemble

Every LR is Pipeline(StandardScaler -> [PCA, embeddings only] ->
LogisticRegression(class_weight, max_iter)). C is chosen from
BaselineConfig.C_grid by stratified inner CV (macro-F1) on the outer-train
participants ONLY, then refit on all of them (fit_lr). Scaler and PCA are
fit inside every inner and outer training split by the Pipeline, never on a
test or middle-band participant.

Outer folds come from folds.csv filtered by splits.analysis_ids
(primary | full), optionally restricted to the pilot codes (--subset pilot);
assert_no_leakage(train, test, middle) runs in every fold. With
--predict-middle (primary analysis only), each fold's model also predicts
the middle-band participants assigned to that fold; they are never used for
fitting and never enter the test-fold metrics.

Refuses to run on real data before the protocol-frozen tag
(src.utils.protocol_guard).

Outputs, results/baselines/<analysis>[_<subset>]/:

    predictions.csv     analysis, model, feature_set, fold, participant_id,
                        label, prob_sad, pred, in_middle_band
    summary.json        per model/feature_set: per-fold metrics and chosen C,
                        pooled out-of-fold metrics, middle-band metrics
    lr_coefficients.csv standardized coefficients per fold (non-PCA models)

Run from the project root:

    python -m src.training.run_baselines [--analysis primary|full]
        [--subset pilot] [--predict-middle]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataloader import labels_csv_path
from src.features.handcrafted import FAMILIES
from src.training.splits import (
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    load_folds,
)
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic

PRED_COLUMNS = (
    "analysis",
    "model",
    "feature_set",
    "fold",
    "participant_id",
    "label",
    "prob_sad",
    "pred",
    "in_middle_band",
)
SUBSETS = ("pilot",)
EXPLORATORY_MODELS = ("lr_handwriting",)
HANDWRITING_FAMILIES = ("word", "cursive")


# ── features ─────────────────────────────────────────────────────────────────


def _handcrafted_sets(cfg) -> dict:
    path = cfg.paths.results_dir / "features" / "handcrafted_participant.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run src.features.handcrafted")
    table = pd.read_csv(path, dtype={"participant_id": str}).set_index("participant_id")
    sets = {"all": table}
    for family in FAMILIES:
        sets[family] = table[[c for c in table.columns if c.startswith(f"{family}__")]]
    return sets


def _participant_means(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as z:
        emb = pd.DataFrame(z["embeddings"], index=z["participant_id"].astype(str))
    return emb.groupby(level=0).mean()


def _embedding_sets(cfg, prefix: str, families) -> dict | None:
    emb_dir = cfg.paths.results_dir / "embeddings"
    paths = {f: emb_dir / f"{prefix}_{f}.npz" for f in families}
    if not all(p.is_file() for p in paths.values()):
        return None
    means = {f: _participant_means(p).add_prefix(f"{f}__") for f, p in paths.items()}
    sets = {"concat": pd.concat(list(means.values()), axis=1, join="inner")}
    sets.update(means)
    return sets


def feature_sets(cfg) -> list:
    """[(model, feature_set, X, use_pca)] for every LR to evaluate."""
    out = [
        ("lr_handcrafted", name, X, False) for name, X in _handcrafted_sets(cfg).items()
    ]
    emb = _embedding_sets(cfg, "resnet18", FAMILIES)
    if emb is None:
        raise FileNotFoundError("resnet18_*.npz missing; run src.features.embeddings")
    out += [("lr_embeddings", name, X, True) for name, X in emb.items()]
    hw = _embedding_sets(cfg, "handwriting", HANDWRITING_FAMILIES)
    if hw is not None:
        out.append(("lr_handwriting", "concat", hw["concat"], True))
    return out


# ── model ────────────────────────────────────────────────────────────────────


def fit_lr(X_train, y_train, cfg, pca: bool = False, seed: int | None = None):
    """Best Pipeline(scaler [-> PCA] -> LR) after inner-CV choice of C.

    The inner CV is stratified with min(inner_cv, smallest class count)
    folds over the given (outer-train) participants only; the winning C is
    refit on all of them. PCA keeps min(pca_components, smallest inner
    training size - 1, n_features) components.
    """
    bcfg = cfg.baselines
    y = np.asarray(y_train)
    n_inner = min(bcfg.inner_cv, int(np.bincount(y, minlength=2).min()))
    if n_inner < 2:
        raise ValueError("each class needs >= 2 training participants for inner CV")

    steps = [("scaler", StandardScaler())]
    if pca:
        smallest_inner_train = len(y) - math.ceil(len(y) / n_inner) - 1
        n_comp = min(bcfg.pca_components, smallest_inner_train - 1, X_train.shape[1])
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_comp,
                    svd_solver=bcfg.pca_svd_solver,
                    random_state=seed,
                ),
            )
        )
    steps.append(
        (
            "lr",
            LogisticRegression(class_weight=bcfg.class_weight, max_iter=bcfg.max_iter),
        )
    )
    pipe = Pipeline(steps)
    search = GridSearchCV(
        pipe,
        {"lr__C": list(bcfg.C_grid)},
        scoring=bcfg.scoring,
        cv=StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed),
        refit=True,
    )
    # Plain float arrays: sklearn's per-column DataFrame checks dominated
    # runtime on 512-1536-column embedding sets.
    search.fit(np.asarray(X_train, dtype=float), y)
    return search.best_estimator_


def _prob_sad(pipe, X) -> np.ndarray:
    sad = config.data.label_to_index["SAD"]
    col = list(pipe.classes_).index(sad)
    return pipe.predict_proba(np.asarray(X, dtype=float))[:, col]


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    y = (df["label"] == "SAD").to_numpy()
    p = (df["pred"] == "SAD").to_numpy()
    out = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "roc_auc": None,
    }
    if len(set(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, df["prob_sad"]))
    return out


# ── run ──────────────────────────────────────────────────────────────────────


def _restrict(ids: list, subset: str | None) -> list:
    if subset is None:
        return ids
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    lo, hi = config.cv.pilot_id_range
    return [p for p in ids if lo <= int(p) <= hi]


def _rows(analysis, model, fs, fold, ids, labels, prob, middle) -> pd.DataFrame:
    thr = config.aggregate.threshold
    prob = np.asarray(prob, dtype=float)
    return pd.DataFrame(
        {
            "analysis": analysis,
            "model": model,
            "feature_set": fs,
            "fold": fold,
            "participant_id": ids,
            "label": labels.loc[ids].to_numpy(),
            "prob_sad": prob,
            "pred": np.where(prob >= thr, "SAD", "HAPPY"),
            "in_middle_band": [p in middle for p in ids],
        }
    )


def run(
    analysis: str = "primary", subset: str | None = None, predict_middle: bool = False
) -> Path:
    """Evaluate all baselines over the outer folds; return predictions.csv path."""
    require_frozen_or_synthetic(config, "run_baselines")
    if predict_middle and analysis != "primary":
        raise ValueError("--predict-middle applies to the primary analysis only")

    meta = config.paths.metadata_dir
    labels = pd.read_csv(labels_csv_path(config), dtype={"participant_id": str})
    participants = pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
    folds = load_folds(config)
    ids = _restrict(analysis_ids(labels, participants, analysis), subset)
    if not ids:
        raise ValueError(f"no participants in analysis {analysis!r} / {subset}")
    middle_ids = []
    if predict_middle:
        full = set(analysis_ids(labels, participants, "full"))
        band = labels.loc[labels["in_middle_band"].astype(bool), "participant_id"]
        middle_ids = _restrict(sorted(full & set(band.astype(str))), subset)

    label_of = labels.assign(participant_id=labels["participant_id"].astype(str))
    label_of = label_of.set_index("participant_id")[config.labeling.label_column]
    sad_index = config.data.label_to_index["SAD"]

    sets = feature_sets(config)
    for model, fs, X, _ in sets:
        missing = set(ids + middle_ids) - set(X.index)
        if missing:
            raise ValueError(f"{model}/{fs}: no features for {sorted(missing)[:10]}")

    preds, per_fold, coefs = [], [], []
    middle = set(middle_ids)
    for fold in range(config.cv.n_splits):
        train, test = fold_ids(folds, fold, ids)
        mid = fold_ids(folds, fold, middle_ids)[1] if middle_ids else []
        assert_no_leakage(train, test, mid)
        eval_ids = test + mid
        y_train = label_of.loc[train].map(config.data.label_to_index).to_numpy()
        seed = config.training.seed + fold

        share = float(np.mean(y_train == sad_index))
        preds.append(
            _rows(
                analysis,
                "majority",
                "all",
                fold,
                eval_ids,
                label_of,
                np.full(len(eval_ids), share),
                middle,
            )
        )
        per_fold.append(
            {
                "model": "majority",
                "feature_set": "all",
                "fold": fold,
                "C": None,
                "n_pca": None,
            }
        )

        for model, fs, X, use_pca in sets:
            pipe = fit_lr(X.loc[train], y_train, config, pca=use_pca, seed=seed)
            prob = _prob_sad(pipe, X.loc[eval_ids])
            preds.append(
                _rows(analysis, model, fs, fold, eval_ids, label_of, prob, middle)
            )
            C = float(pipe["lr"].C)
            n_pca = int(pipe["pca"].n_components_) if use_pca else None
            per_fold.append(
                {
                    "model": model,
                    "feature_set": fs,
                    "fold": fold,
                    "C": C,
                    "n_pca": n_pca,
                }
            )
            if not use_pca:
                coefs.append(
                    pd.DataFrame(
                        {
                            "analysis": analysis,
                            "model": model,
                            "feature_set": fs,
                            "fold": fold,
                            "feature": X.columns,
                            "coef_std": pipe["lr"].coef_[0],
                            "C": C,
                        }
                    )
                )

    predictions = pd.concat(preds, ignore_index=True)[list(PRED_COLUMNS)]
    summary = _summary(analysis, subset, predict_middle, predictions, per_fold, ids)

    name = analysis if subset is None else f"{analysis}_{subset}"
    out_dir = config.paths.results_dir / config.baselines.output_subdir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.concat(coefs, ignore_index=True).to_csv(
        out_dir / "lr_coefficients.csv", index=False
    )

    for key, entry in summary["models"].items():
        m = entry["pooled_test"]
        print(
            f"{key:28s} macro-F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}"
            f"{'  (exploratory)' if entry['exploratory'] else ''}"
        )
    print(f"Written: {out_dir}")
    return pred_path


def _summary(analysis, subset, predict_middle, predictions, per_fold, ids) -> dict:
    bcfg = config.baselines
    models = {}
    fold_info = pd.DataFrame(per_fold)
    for (model, fs), part in predictions.groupby(["model", "feature_set"], sort=False):
        test = part[~part["in_middle_band"]]
        mid = part[part["in_middle_band"]]
        info = fold_info[
            (fold_info["model"] == model) & (fold_info["feature_set"] == fs)
        ]
        folds = []
        for fold, fpart in test.groupby("fold"):
            row = info[info["fold"] == fold].iloc[0]
            folds.append(
                {
                    "fold": int(fold),
                    "C": row["C"] if pd.notna(row["C"]) else None,
                    "n_pca": int(row["n_pca"]) if pd.notna(row["n_pca"]) else None,
                    **_metrics(fpart),
                }
            )
        models[f"{model}/{fs}"] = {
            "model": model,
            "feature_set": fs,
            "exploratory": model in EXPLORATORY_MODELS,
            "folds": folds,
            "pooled_test": _metrics(test),
            "middle_band": _metrics(mid) if predict_middle else None,
        }
    return {
        "analysis": analysis,
        "subset": subset,
        "predict_middle": predict_middle,
        "n_participants": len(ids),
        "n_splits": config.cv.n_splits,
        "seed": config.training.seed,
        "settings": {
            "C_grid": list(bcfg.C_grid),
            "inner_cv": bcfg.inner_cv,
            "class_weight": bcfg.class_weight,
            "max_iter": bcfg.max_iter,
            "pca_components": bcfg.pca_components,
            "pca_svd_solver": bcfg.pca_svd_solver,
            "scoring": bcfg.scoring,
        },
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Majority + LR baselines.")
    parser.add_argument("--analysis", choices=("primary", "full"), default="primary")
    parser.add_argument("--subset", choices=SUBSETS, default=None)
    parser.add_argument("--predict-middle", action="store_true")
    args = parser.parse_args()
    try:
        run(args.analysis, args.subset, args.predict_middle)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
