"""
Per-exercise training v3 — EXPLORATORY, post hoc (EVALUATION_PROTOCOL.md
section 9, Deviation 6). Never part of the primary claim, the ensemble, or
any v1 or v2 output; the primary model stays CNN-HMM v1 (run_hybrid.py).

v1 trains one model per task family on all of that family's crops. v3
trains one model per exercise: each of the 24 FINALE cells
(src.data.crop_manifest.ALL_CELLS, PerTaskV3Config.cells) is modelled on its
own, one image per participant, on the primary set and folds.csv. Per
(cell, outer fold), every model takes its participants from fold_splits:

  1. train, test = fold_ids(folds.csv, fold, primary IDs)
  2. inner_train, inner_val = inner_split(train, true labels,
     CVConfig.inner_val_fraction, TrainingConfig.seed + fold), the same
     split for all 24 cells of a fold
  3. assert_no_leakage(train, test) and
     assert_no_leakage(inner_train, inner_val, test)

Models per cell: lr_handcrafted_v3 (the cell's rows of
results/features/handcrafted_crop.csv), lr_embeddings_v3 (the cell's frozen
ResNet18 embedding), cnn_head_v3 (ResNet18 fine-tuned on the cell's crops)
and cnn_hmm_v3 (layer3 column-sequence HMM on that checkpoint), plus the
equal-weight combinations in PerTaskV3Config.fuse_sets and
cnn_hmm_v3_selected. Nothing is fit, selected, weighted or calibrated on
outer-test participants.

Outputs (v3 paths only):

    models/cnn_v3/primary/<cell>/fold_<k>.pth
    models/hmm_v3/primary/<cell>/fold_<k>.joblib
    results/per_task_v3/primary/run_config.json      a rerun refuses a mismatch
    results/per_task_v3/primary/parts/...            per (model, cell, fold)
    results/per_task_v3/primary/predictions.csv, inner_val_predictions.csv,
        selected_cell.csv, per_task_table.csv, metrics.csv, per_fold.csv,
        confusion.csv, paired.csv, permutation_lr.csv, permutation_cnn.json
    results/final/permutation_runs/primary_v3/seed_<s>/perm_<k>.csv
    results/final/EXPLORATORY_v3.txt

Preflight checks, before anything is fit: folds.csv and labels.csv match
the SHA-256 values frozen in the plan (real roots only), every primary
participant has exactly one processed crop, one handcrafted row and one
embedding for each of the 24 cells, no primary crop is dropped in
qc_log.csv, and it prints the inner split sizes per fold and the number of
crops with imputed handcrafted values per cell.

    python -m src.training.run_per_task_v3 --step preflight
    python -m src.training.run_per_task_v3 --step lr    per-exercise LR parts
    python -m src.training.run_per_task_v3 --step cnn [--cells D1,W2_RH]
        [--folds 0,1] [--device cuda]
    python -m src.training.run_per_task_v3 --step hmm [--cells ...]
        [--folds ...] [--device cuda] [--n-jobs -1]
    python -m src.training.run_per_task_v3 --step combine   predictions.csv
    python -m src.training.run_per_task_v3 --step evaluate  metrics, tables
    python -m src.training.run_per_task_v3 --step report    EXPLORATORY_v3.txt
    python -m src.training.run_per_task_v3 --step permutation-lr [--n-jobs -1]
    python -m src.training.run_per_task_v3 --step draws --draws 0-9 [--device cuda]
    python -m src.training.run_per_task_v3 --step permutation-cnn

combine averages the per-cell test prob_sad of cnn_head_v3 and cnn_hmm_v3
over each fuse set and picks, per fold, the cell with the best cnn_hmm_v3
inner-val macro-F1 (inner-val rows only). evaluate refuses while
n_cnn_permutations is undecided (-1) and prints only file paths. A missing
v1 reference group gives a paired.csv row "no v1 counterpart".

Permutation tests (exploratory). permutation-lr: per (LR kind, cell),
n_lr_permutations draws; in draw k, fold f the outer-train labels are
permuted among themselves with permutation_seed(seed, k, f) and the LR is
refit as in the observed run; BH q-values within each kind across cells.
draws: one full CNN + HMM retrain of every cell x fold per draw, cached as
permutation_runs/primary_v3/seed_<s>/perm_<kkkk>.csv (mergeable with
scripts/merge_draws.py; checkpoints only in a temporary folder).
permutation-cnn: the headline test once all n_cnn_permutations draws are
cached. draws and permutation-cnn refuse while D2 is undecided
(n_cnn_permutations = -1).

Resume unit of cnn and hmm: one (cell, fold). A cnn unit is done when its
test part, its .val.csv part and its checkpoint exist; an hmm unit when its
test part, .val.csv part, .json selection record and joblib exist. hmm
refuses a (cell, fold) without a CNN checkpoint. --n-jobs sets
RunnerConfig.hmm_n_jobs (speed only, Deviation 2), read by both
run_hybrid.select_hmm and run_hybrid.fit_final.

Nothing is appended to results/RUN_LOG.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.analysis.permutation import _plot, p_value, parse_draws, permutation_seed
from src.data.crop_manifest import (
    ALL_CELLS,
    CURSIVE_CELLS,
    DRAWING_CELLS,
    DRAWING_NAMES,
    WORD_NAMES,
)
from src.data.dataloader import CropDataset, labels_csv_path, load_tables
from src.data.transforms import get_eval_transform, get_train_transform
from src.features.embeddings import resolve_device
from src.features.handcrafted import features_for_family
from src.models.cnn import EmotionCNN
from src.models.hybrid import load_cnn
from src.training import evaluator, run_hybrid
from src.training.run_baselines import PRED_COLUMNS, _prob_sad, fit_lr
from src.training.run_cnn import check_resume, shuffled_labels
from src.training.splits import (
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    folds_path,
    inner_split,
    load_folds,
)
from src.training.trainer import Trainer
from src.utils.config import config
from src.utils.protocol_guard import is_synthetic_root, require_frozen_or_synthetic
from src.utils.seed import set_seed

FOLDS_SHA256 = "40403812420cde0e1f7d1980ce02ad0f77c531fdf59aab62b7987ca4b90751bf"
LABELS_SHA256 = "694d360091e28dc331422cf7171b8e3afcf5dda16545e7ad5b2a9291204d3006"
FAMILIES = ("drawing", "word", "cursive")
HANDCRAFTED_CROP = "handcrafted_crop.csv"
PROCESSED_MANIFEST = "processed_manifest.csv"
IMPUTED_COLUMN = "imputed"
UNDECIDED_KEYS = ("n_cnn_permutations", "n_lr_permutations")
STEPS = (
    "preflight",
    "lr",
    "cnn",
    "hmm",
    "combine",
    "evaluate",
    "report",
    "permutation-lr",
    "draws",
    "permutation-cnn",
)
PERMUTATION_FIGURE = "permutation_null_v3.png"
DRAWS_KIND = "v3_draws"
DESCRIPTIVE_ONLY = "descriptive only"
LABELS = ("HAPPY", "SAD")
NO_V1 = "no v1 counterpart"
REFERENCE_SOURCES = {
    "lr_handcrafted": "baselines",
    "lr_embeddings": "baselines",
    "cnn_head": "cnn",
    "cnn_head_fused": "cnn",
    "cnn_hmm": "hybrid",
    "cnn_hmm_fused": "hybrid",
    "cnn_hmm_v2": "hybrid_v2",
}
TABLE_METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "roc_auc")
METRIC_NAMES = {
    "macro_f1": "macro-F1",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced accuracy",
    "roc_auc": "ROC-AUC",
}
REPORT_HEADER = (
    "EXPLORATORY, POST HOC (Deviation 6). Not part of the primary claim; the "
    "primary model remains CNN-HMM v1 (cnn_hmm_fused/fused)."
)
LIMITATIONS = (
    "Limitations. Each per-exercise model trains on about 217 images (one per "
    "inner-train participant), far fewer than the v1 family models. An "
    "inner-validation set of about 39 participants is used for CNN early "
    "stopping, HMM selection, Platt calibration and cell selection, so those "
    "choices share one small set. 24 exercises x 4 models is a 24-way "
    "multiplicity per model; per-cell results are descriptive and only "
    "cnn_hmm_v3_fused/all24 is a pre-declared test. The whole v3 design is "
    "post hoc (Deviation 6), decided after the v1 and v2 results were known."
)
MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MODEL_MARKERS = ("o", "s", "^", "D")


def family_of(cell: str) -> str:
    """Task family of a cell code: D* drawing, W* word, CS* cursive."""
    if cell not in ALL_CELLS:
        raise ValueError(f"unknown cell {cell!r}; expected one of {ALL_CELLS}")
    if cell in CURSIVE_CELLS:
        return "cursive"
    if cell in DRAWING_CELLS:
        return "drawing"
    return "word"


def output_dir(cfg=config) -> Path:
    v3 = cfg.per_task_v3
    return cfg.paths.results_dir / v3.output_subdir / v3.analysis


def cnn_ckpt_path(cell: str, fold: int, cfg=config) -> Path:
    v3 = cfg.per_task_v3
    base = cfg.paths.models_dir / v3.cnn_model_subdir / v3.analysis
    return base / cell / f"fold_{fold}.pth"


def hmm_model_path(cell: str, fold: int, cfg=config) -> Path:
    v3 = cfg.per_task_v3
    base = cfg.paths.models_dir / v3.hmm_model_subdir / v3.analysis
    return base / cell / f"fold_{fold}.joblib"


def runs_dir(seed: int, cfg=config) -> Path:
    out = evaluator.output_dir(None, cfg) / cfg.permutation.runs_subdir
    return out / cfg.per_task_v3.runs_name / f"seed_{seed}"


def handcrafted_path(cfg=config) -> Path:
    return cfg.paths.results_dir / cfg.features.output_subdir / HANDCRAFTED_CROP


def embedding_path(family: str, cfg=config) -> Path:
    prefix = cfg.per_task_v3.embedding_prefix
    out = cfg.paths.results_dir / cfg.embedding.output_subdir
    return out / f"{prefix}_{family}.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context(cfg=config) -> tuple:
    """(labels.csv, analysis-set IDs, folds.csv)."""
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    participants = pd.read_csv(
        cfg.paths.metadata_dir / "participants.csv", dtype={"participant_id": str}
    )
    ids = analysis_ids(labels, participants, cfg.per_task_v3.analysis)
    return labels, ids, load_folds(cfg)


def fold_splits(
    fold: int, ids: list, labels: pd.DataFrame, folds: pd.DataFrame, cfg=config
) -> tuple:
    """(inner_train, inner_val, test) of one outer fold, shared by every model.

    The inner split is drawn on the true labels, so it is the same for all
    24 cells of a fold and in every permutation draw.
    """
    train, test = fold_ids(folds, fold, ids)
    assert_no_leakage(train, test)
    inner_train, inner_val = inner_split(
        train, labels, cfg.cv.inner_val_fraction, cfg.training.seed + fold
    )
    assert_no_leakage(inner_train, inner_val, test)
    return inner_train, inner_val, test


def cell_manifest(
    manifest: pd.DataFrame, cell: str, ids: list, dropped: set
) -> pd.DataFrame:
    """processed_manifest rows of one cell for ids; exactly one per participant."""
    wanted = {str(p) for p in ids}
    rows = manifest[manifest["cell"] == cell].copy()
    rows["participant_id"] = rows["participant_id"].astype(str)
    rows = rows[rows["participant_id"].isin(wanted)]
    keep = [(p, cell) not in dropped for p in rows["participant_id"]]
    rows = rows[np.array(keep, dtype=bool)]
    counts = rows["participant_id"].value_counts()
    missing = sorted(wanted - set(counts.index))
    repeated = sorted(counts.index[counts > 1])
    if missing or repeated:
        raise ValueError(
            f"cell {cell}: every participant needs exactly one processed crop; "
            f"missing {missing[:10]} ({len(missing)}), repeated {repeated[:10]} "
            f"({len(repeated)})"
        )
    return rows.sort_values("participant_id").reset_index(drop=True)


def cell_loader(
    manifest_cell: pd.DataFrame,
    labels: pd.DataFrame,
    ids: list,
    family: str,
    train: bool,
    cfg=config,
) -> DataLoader:
    """DataLoader over one cell's crops (build_loaders' settings, one cell)."""
    transform = (
        get_train_transform(cfg, family) if train else get_eval_transform(cfg, family)
    )
    ds = CropDataset(manifest_cell, labels, ids, [family], transform=transform)
    if not train:
        return DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )
    return DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )


def _handcrafted_table(cfg=config) -> pd.DataFrame:
    path = handcrafted_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run src.features.handcrafted")
    return pd.read_csv(path, dtype={"participant_id": str, IMPUTED_COLUMN: str})


def handcrafted_cell(cell: str, cfg=config) -> pd.DataFrame:
    """One cell's handcrafted features, indexed by participant_id."""
    table = _handcrafted_table(cfg)
    columns = list(features_for_family(family_of(cell)))
    rows = table[table["cell"] == cell].set_index("participant_id")[columns]
    repeated = sorted(set(rows.index[rows.index.duplicated()]))
    if repeated:
        raise ValueError(f"{cell}: participants with two handcrafted rows {repeated}")
    bad = rows.index[rows.isna().any(axis=1)]
    if len(bad):
        raise ValueError(f"{cell}: NaN handcrafted values for {sorted(bad)[:10]}")
    return rows


def embedding_cell(cell: str, cfg=config) -> pd.DataFrame:
    """One cell's frozen ResNet18 embeddings, indexed by participant_id."""
    path = embedding_path(family_of(cell), cfg)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run src.features.embeddings")
    with np.load(path, allow_pickle=False) as z:
        mask = z["cell"].astype(str) == cell
        emb = z["embeddings"][mask]
        pids = z["participant_id"].astype(str)[mask]
    rows = pd.DataFrame(emb, index=pd.Index(pids, name="participant_id"))
    repeated = sorted(set(rows.index[rows.index.duplicated()]))
    if repeated:
        raise ValueError(f"{path}: {cell} embedded twice for {repeated[:10]}")
    return rows


def imputed_counts(cfg=config) -> pd.DataFrame:
    """Per cell, the number of crops with at least one imputed handcrafted value."""
    table = _handcrafted_table(cfg)
    flagged = table[table[IMPUTED_COLUMN].fillna("").str.strip() != ""]
    counts = flagged["cell"].value_counts()
    return pd.DataFrame(
        {
            "cell": list(cfg.per_task_v3.cells),
            "n_crops_with_imputed_values": [
                int(counts.get(c, 0)) for c in cfg.per_task_v3.cells
            ],
        }
    )


def settings(cfg=config) -> dict:
    """Everything a rerun must match (run_config.json).

    The permutation counts are left out: n_cnn_permutations is set after the
    main run (Deviation 6, D2) and must not invalidate it.
    """
    v3 = asdict(cfg.per_task_v3)
    for key in UNDECIDED_KEYS:
        v3.pop(key)
    meta = cfg.paths.metadata_dir
    return json.loads(
        json.dumps(
            {
                "per_task_v3": v3,
                "cnn": asdict(cfg.cnn),
                "training": asdict(cfg.training),
                "hmm": asdict(cfg.hmm),
                "hybrid": asdict(cfg.hybrid),
                "baselines": asdict(cfg.baselines),
                "aggregate": asdict(cfg.aggregate),
                "seed": cfg.training.seed,
                "inner_val_fraction": cfg.cv.inner_val_fraction,
                "threshold": cfg.aggregate.threshold,
                "folds_sha256": _sha256(folds_path(cfg)),
                "labels_sha256": _sha256(labels_csv_path(cfg)),
                "processed_manifest_sha256": _sha256(meta / PROCESSED_MANIFEST),
                "embeddings_sha256": {
                    f: _sha256(embedding_path(f, cfg)) for f in FAMILIES
                },
                "handcrafted_crop_sha256": _sha256(handcrafted_path(cfg)),
            },
            default=str,
        )
    )


def check_settings(path: Path, want: dict) -> bool:
    """True if path holds want; False if absent; raises on a mismatch."""
    if not path.is_file():
        return False
    have = json.loads(path.read_text(encoding="utf-8"))
    if have != want:
        changed = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
        raise RuntimeError(
            f"{path} records different settings ({changed}); v3 runs once "
            "(Deviation 6). Log an addendum under Deviation 6 before any "
            "rerun; never delete the existing v3 outputs."
        )
    return True


def _check_hashes(cfg=config) -> dict:
    hashes = {
        "folds_sha256": _sha256(folds_path(cfg)),
        "labels_sha256": _sha256(labels_csv_path(cfg)),
    }
    if is_synthetic_root(cfg.paths.data_root):
        return hashes
    for key, want in (("folds_sha256", FOLDS_SHA256), ("labels_sha256", LABELS_SHA256)):
        if hashes[key] != want:
            raise RuntimeError(
                f"{key} is {hashes[key]}, expected {want} (plan section 3, L2)"
            )
    return hashes


def _require_rows(name: str, cell: str, have, ids: list) -> None:
    missing = sorted(set(ids) - set(have))
    if missing:
        raise ValueError(
            f"{name} {cell}: no row for {len(missing)} primary participants, "
            f"e.g. {missing[:10]}"
        )


def preflight(cfg=config) -> dict:
    """Check every v3 input before anything is fit; print and return a summary."""
    require_frozen_or_synthetic(cfg, "run_per_task_v3")
    if tuple(cfg.per_task_v3.cells) != ALL_CELLS:
        raise ValueError("PerTaskV3Config.cells must equal crop_manifest.ALL_CELLS")
    hashes = _check_hashes(cfg)
    labels, ids, folds = context(cfg)
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    counts = lab.loc[ids].value_counts()

    manifest, _, dropped = load_tables(cfg)
    primary = set(ids)
    bad = sorted((p, c) for p, c in dropped if str(p) in primary)
    if bad:
        raise ValueError(
            f"{len(bad)} dropped crops of primary participants (qc_log.csv): {bad}"
        )
    for cell in cfg.per_task_v3.cells:
        cell_manifest(manifest, cell, ids, dropped)
        hand = handcrafted_cell(cell, cfg)
        _require_rows("handcrafted", cell, hand.index, ids)
        emb = embedding_cell(cell, cfg)
        _require_rows("embedding", cell, emb.index, ids)
        mine = emb.loc[ids]
        nan = sorted(mine.index[mine.isna().any(axis=1)])
        if nan:
            raise ValueError(f"embedding {cell}: NaN values for {nan[:10]}")

    splits = []
    for fold in range(cfg.cv.n_splits):
        inner_train, inner_val, test = fold_splits(fold, ids, labels, folds, cfg)
        splits.append(
            {
                "fold": fold,
                "n_inner_train": len(inner_train),
                "n_inner_val": len(inner_val),
                "n_test": len(test),
            }
        )
    imputed = imputed_counts(cfg)
    summary = {
        "analysis": cfg.per_task_v3.analysis,
        "synthetic_root": is_synthetic_root(cfg.paths.data_root),
        **hashes,
        "n_primary": len(ids),
        "n_happy": int(counts.get("HAPPY", 0)),
        "n_sad": int(counts.get("SAD", 0)),
        "n_cells": len(cfg.per_task_v3.cells),
        "n_dropped_primary_crops": len(bad),
        "inner_splits": splits,
        "imputed_crops_per_cell": dict(
            zip(imputed["cell"], imputed["n_crops_with_imputed_values"])
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


def part_path(kind: str, cell: str, fold: int, cfg=config) -> Path:
    return output_dir(cfg) / "parts" / f"{kind}__{cell}__fold{fold}.csv"


def val_part_path(kind: str, cell: str, fold: int, cfg=config) -> Path:
    return output_dir(cfg) / "parts" / f"{kind}__{cell}__fold{fold}.val.csv"


def selection_path(cell: str, fold: int, cfg=config) -> Path:
    kind = cfg.per_task_v3.hmm_model
    return output_dir(cfg) / "parts" / f"{kind}__{cell}__fold{fold}.json"


def write_part(frame: pd.DataFrame, path: Path) -> None:
    """Write frame to path atomically (.tmp, then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _write_json(record: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(path)


def _participant_rows(
    model: str,
    cell: str,
    fold: int,
    pids,
    prob,
    labels: pd.DataFrame,
    cfg=config,
) -> pd.DataFrame:
    """PRED_COLUMNS rows, one per participant, labelled with the TRUE labels."""
    pids = [str(p) for p in pids]
    if len(set(pids)) != len(pids):
        raise ValueError(f"{model} {cell} fold {fold}: one row per participant")
    prob = np.asarray(prob, dtype=float)
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    rows = pd.DataFrame(
        {
            "analysis": cfg.per_task_v3.analysis,
            "model": model,
            "feature_set": cell,
            "fold": fold,
            "participant_id": pids,
            "label": lab.loc[pids].to_numpy(),
            "prob_sad": prob,
            "pred": np.where(prob >= cfg.aggregate.threshold, "SAD", "HAPPY"),
            "in_middle_band": False,
        }
    )[list(PRED_COLUMNS)]
    return rows.sort_values("participant_id").reset_index(drop=True)


def _lr_use_pca(kind: str, cfg=config) -> bool:
    """False for lr_handcrafted_v3, True for lr_embeddings_v3."""
    hand_kind, emb_kind = cfg.per_task_v3.lr_kinds
    if kind not in (hand_kind, emb_kind):
        raise ValueError(
            f"unknown LR kind {kind!r}; expected {cfg.per_task_v3.lr_kinds}"
        )
    return kind == emb_kind


def _lr_inputs(kind: str, cell: str, cfg=config) -> tuple:
    """(feature table indexed by participant_id, use PCA) for one LR kind."""
    use_pca = _lr_use_pca(kind, cfg)
    loader = embedding_cell if use_pca else handcrafted_cell
    return loader(cell, cfg), use_pca


def fit_lr_cell(
    kind: str,
    cell: str,
    fold: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    cfg=config,
    y_train=None,
    X=None,
) -> tuple:
    """(fitted pipeline, features, outer-train IDs, test IDs) of one LR fit.

    Fit on the outer-train participants only (C by inner CV inside fit_lr,
    as in run_baselines); y_train None means their true label indices. X is
    the cell's feature table as _lr_inputs loads it; None loads it here.
    """
    if X is None:
        X, use_pca = _lr_inputs(kind, cell, cfg)
    else:
        use_pca = _lr_use_pca(kind, cfg)
    train, test = fold_ids(folds, fold, ids)
    assert_no_leakage(train, test)
    if y_train is None:
        lab = labels.set_index("participant_id")[cfg.labeling.label_column]
        y_train = lab.loc[train].map(cfg.data.label_to_index).to_numpy()
    pipe = fit_lr(
        X.loc[train], y_train, cfg, pca=use_pca, seed=cfg.training.seed + fold
    )
    return pipe, X, train, test


def lr_cell(
    kind: str,
    cell: str,
    fold: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    cfg=config,
    y_train=None,
) -> pd.DataFrame:
    """Test-participant predictions (PRED_COLUMNS) of one (kind, cell, fold)."""
    pipe, X, _, test = fit_lr_cell(kind, cell, fold, ids, labels, folds, cfg, y_train)
    prob = _prob_sad(pipe, X.loc[test])
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    return pd.DataFrame(
        {
            "analysis": cfg.per_task_v3.analysis,
            "model": kind,
            "feature_set": cell,
            "fold": fold,
            "participant_id": test,
            "label": lab.loc[test].to_numpy(),
            "prob_sad": prob,
            "pred": np.where(prob >= cfg.aggregate.threshold, "SAD", "HAPPY"),
            "in_middle_band": False,
        }
    )[list(PRED_COLUMNS)]


def _fit_labels(labels, inner_train, inner_val, shuffle_seed) -> pd.DataFrame:
    if shuffle_seed is None:
        return labels
    return shuffled_labels(labels, sorted(inner_train + inner_val), shuffle_seed)


def fit_cnn_cell(
    cell: str,
    fold: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    manifest: pd.DataFrame,
    dropped: set,
    device: str,
    ckpt: Path,
    cfg=config,
    shuffle_seed=None,
) -> tuple:
    """(test rows, inner-val rows) of cnn_head_v3 for one (cell, fold).

    run_cnn._fit_fold on one cell: fit on inner-train, early-stop on
    inner-val, save the checkpoint with its split IDs. shuffle_seed permutes
    the outer-train labels among themselves before fitting; both prediction
    sets always use the true labels.
    """
    family = family_of(cell)
    inner_train, inner_val, test = fold_splits(fold, ids, labels, folds, cfg)
    rows = cell_manifest(manifest, cell, inner_train + inner_val + test, dropped)
    fit_labels = _fit_labels(labels, inner_train, inner_val, shuffle_seed)

    set_seed(cfg.training.seed + fold)
    train_loader = cell_loader(rows, fit_labels, inner_train, family, True, cfg)
    val_loader = cell_loader(rows, fit_labels, inner_val, family, False, cfg)
    trainer = Trainer(EmotionCNN(cfg.cnn), cfg, device)
    trainer.fit(train_loader, val_loader)
    trainer.save(
        ckpt,
        {
            "cell": cell,
            "family": family,
            "fold": fold,
            "cnn": asdict(cfg.cnn),
            "inner_train_ids": list(inner_train),
            "inner_val_ids": list(inner_val),
            "test_ids": list(test),
            "middle_ids": [],
        },
    )

    out = []
    for role_ids in (test, inner_val):
        crops = trainer.predict_crops(
            cell_loader(rows, labels, role_ids, family, False, cfg)
        )
        out.append(
            _participant_rows(
                cfg.per_task_v3.cnn_model,
                cell,
                fold,
                crops["participant_id"],
                crops["prob_sad"],
                labels,
                cfg,
            )
        )
    return out[0], out[1]


def fit_hmm_cell(
    cell: str,
    fold: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    manifest: pd.DataFrame,
    dropped: set,
    device: str,
    ckpt: Path,
    cfg=config,
    shuffle_seed=None,
    save_path=None,
) -> tuple:
    """(test rows, inner-val rows, selection record) of cnn_hmm_v3.

    run_hybrid._fold on one cell, without its models/hmm/ save: layer3
    sequences of the cell's cnn_head_v3 checkpoint; the HMM setting is
    selected on inner-val, refit on inner-train and Platt-calibrated on
    inner-val. Test participants are only predicted.
    """
    family = family_of(cell)
    inner_train, inner_val, test = fold_splits(fold, ids, labels, folds, cfg)
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found; run --step cnn first")
    cnn, state = load_cnn(ckpt)
    run_hybrid._check_split(state, inner_train, inner_val, test, ckpt)
    rows = cell_manifest(manifest, cell, inner_train + inner_val + test, dropped)
    fit_labels = _fit_labels(labels, inner_train, inner_val, shuffle_seed)

    seed = cfg.training.seed + fold
    set_seed(seed)
    sets = {
        role: run_hybrid.extract_set(
            cnn, rows, role_labels, role_ids, family, dropped, device
        )
        for role, role_ids, role_labels in (
            ("inner_train", inner_train, fit_labels),
            ("inner_val", inner_val, fit_labels),
            ("test", test, labels),
        )
    }
    best, table = run_hybrid.select_hmm(sets["inner_train"], sets["inner_val"], seed)
    clf = run_hybrid.fit_final(sets["inner_train"], sets["inner_val"], best, seed)
    if save_path:
        clf.save(save_path)

    sad_col = list(clf.classes_).index(cfg.data.label_to_index["SAD"])
    out = []
    for role in ("test", "inner_val"):
        prob = clf.predict_proba(sets[role].seqs)[:, sad_col]
        out.append(
            _participant_rows(
                cfg.per_task_v3.hmm_model,
                cell,
                fold,
                sets[role].meta["participant_id"],
                prob,
                labels,
                cfg,
            )
        )
    selection = {
        "cell": cell,
        "fold": fold,
        "chosen": best,
        "grid": table,
        "refit_restarts": cfg.hmm.n_restarts,
        "restart_log": clf.restart_log,
        "cnn_checkpoint": str(ckpt),
        "n_inner_train": len(inner_train),
        "n_inner_val": len(inner_val),
        "n_test": len(test),
    }
    return out[0], out[1], selection


def _start(cfg=config) -> Path:
    """Write run_config.json, or check it matches; return the output folder."""
    out = output_dir(cfg)
    want = settings(cfg)
    if not check_settings(out / "run_config.json", want):
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps(want, indent=2))
    return out


def step_lr(cfg=config) -> Path:
    """Fit every (LR kind, cell, fold) not yet cached; never prints a metric."""
    require_frozen_or_synthetic(cfg, "run_per_task_v3 lr")
    out = _start(cfg)
    labels, ids, folds = context(cfg)
    v3 = cfg.per_task_v3
    start = time.perf_counter()
    fitted = resumed = 0
    for kind in v3.lr_kinds:
        for cell in v3.cells:
            for fold in range(cfg.cv.n_splits):
                path = part_path(kind, cell, fold, cfg)
                if path.is_file():
                    print(f"resume: {kind} {cell} fold {fold}")
                    resumed += 1
                    continue
                write_part(lr_cell(kind, cell, fold, ids, labels, folds, cfg), path)
                fitted += 1
            took = time.perf_counter() - start
            print(f"lr: {kind} {cell} done, {took / 60:.1f} min elapsed", flush=True)
    total = len(v3.lr_kinds) * len(v3.cells) * cfg.cv.n_splits
    print(
        f"lr: {fitted} parts fitted, {resumed} already done, "
        f"{total} expected, {(time.perf_counter() - start) / 60:.1f} min"
    )
    return out


def _units(cells, folds, cfg=config) -> list:
    """(cell, fold) pairs to run, in cell then fold order; validated."""
    known = tuple(cfg.per_task_v3.cells)
    cells = known if cells is None else tuple(cells)
    unknown = sorted(set(cells) - set(known))
    if unknown:
        raise ValueError(f"unknown cells {unknown}; expected some of {known}")
    folds = range(cfg.cv.n_splits) if folds is None else list(folds)
    bad = sorted(set(folds) - set(range(cfg.cv.n_splits)))
    if bad:
        raise ValueError(f"folds {bad} out of range 0..{cfg.cv.n_splits - 1}")
    return [(c, k) for c in known if c in cells for k in sorted(set(folds))]


def _step_units(name: str, cfg, cells, folds, device, done, fit) -> Path:
    require_frozen_or_synthetic(cfg, f"run_per_task_v3 {name}")
    units = _units(cells, folds, cfg)
    out = _start(cfg)
    device = resolve_device(device)
    labels, ids, fold_table = context(cfg)
    manifest, _, dropped = load_tables(cfg)
    start = time.perf_counter()
    fitted = resumed = 0
    for cell, fold in units:
        if done(cell, fold):
            print(f"resume: {name} {cell} fold {fold}")
            resumed += 1
            continue
        began = time.perf_counter()
        fit(cell, fold, ids, labels, fold_table, manifest, dropped, device)
        fitted += 1
        print(f"{name} {cell} fold {fold}: {time.perf_counter() - began:.1f} s")
    print(
        f"{name}: {fitted} units fitted, {resumed} already done, {len(units)} "
        f"requested, {time.perf_counter() - start:.1f} s",
        flush=True,
    )
    return out


def step_cnn(cfg=config, cells=None, folds=None, device="auto") -> Path:
    """Fine-tune cnn_head_v3 for every (cell, fold) not yet done; no metric."""
    model = cfg.per_task_v3.cnn_model

    def done(cell, fold):
        return all(
            p.is_file()
            for p in (
                part_path(model, cell, fold, cfg),
                val_part_path(model, cell, fold, cfg),
                cnn_ckpt_path(cell, fold, cfg),
            )
        )

    def fit(cell, fold, ids, labels, folds_, manifest, dropped, device_):
        test_rows, val_rows = fit_cnn_cell(
            cell,
            fold,
            ids,
            labels,
            folds_,
            manifest,
            dropped,
            device_,
            cnn_ckpt_path(cell, fold, cfg),
            cfg,
        )
        write_part(val_rows, val_part_path(model, cell, fold, cfg))
        write_part(test_rows, part_path(model, cell, fold, cfg))

    return _step_units("cnn", cfg, cells, folds, device, done, fit)


def step_hmm(cfg=config, cells=None, folds=None, device="auto") -> Path:
    """Fit cnn_hmm_v3 for every (cell, fold) not yet done; no metric."""
    model = cfg.per_task_v3.hmm_model

    def done(cell, fold):
        return all(
            p.is_file()
            for p in (
                part_path(model, cell, fold, cfg),
                val_part_path(model, cell, fold, cfg),
                selection_path(cell, fold, cfg),
                hmm_model_path(cell, fold, cfg),
            )
        )

    def fit(cell, fold, ids, labels, folds_, manifest, dropped, device_):
        test_rows, val_rows, selection = fit_hmm_cell(
            cell,
            fold,
            ids,
            labels,
            folds_,
            manifest,
            dropped,
            device_,
            cnn_ckpt_path(cell, fold, cfg),
            cfg,
            save_path=hmm_model_path(cell, fold, cfg),
        )
        _write_json(selection, selection_path(cell, fold, cfg))
        write_part(val_rows, val_part_path(model, cell, fold, cfg))
        write_part(test_rows, part_path(model, cell, fold, cfg))

    return _step_units("hmm", cfg, cells, folds, device, done, fit)


def report_path(cfg=config) -> Path:
    return evaluator.output_dir(None, cfg) / cfg.per_task_v3.report_name


def figure_path(name: str, cfg=config) -> Path:
    return evaluator.output_dir(None, cfg) / f"{cfg.per_task_v3.figure_prefix}{name}"


def per_cell_models(cfg=config) -> tuple:
    v3 = cfg.per_task_v3
    return (*v3.lr_kinds, v3.cnn_model, v3.hmm_model)


def v3_models(cfg=config) -> tuple:
    v3 = cfg.per_task_v3
    return (
        *per_cell_models(cfg),
        v3.fused_cnn_model,
        v3.fused_hmm_model,
        v3.selected_model,
    )


def load_parts(cfg=config, kind: str = "test") -> pd.DataFrame:
    """Every per-cell part of one kind ("test" or "val"); raises on any gap."""
    v3 = cfg.per_task_v3
    if kind == "test":
        models, path_of = per_cell_models(cfg), part_path
    elif kind == "val":
        models, path_of = (v3.cnn_model, v3.hmm_model), val_part_path
    else:
        raise ValueError(f"kind must be 'test' or 'val', got {kind!r}")
    paths = [
        path_of(model, cell, fold, cfg)
        for model in models
        for cell in v3.cells
        for fold in range(cfg.cv.n_splits)
    ]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} {kind} parts missing under {output_dir(cfg) / 'parts'}"
            f", e.g. {missing[:10]}; run --step lr, cnn and hmm first"
        )
    return pd.concat(
        [pd.read_csv(p, dtype={"participant_id": str}) for p in paths],
        ignore_index=True,
    )


def fuse_cells(cfg=config) -> dict:
    """Fuse-set name -> the cells (form order) whose family is in that set."""
    out = {}
    for name, families in cfg.per_task_v3.fuse_sets:
        cells = tuple(c for c in cfg.per_task_v3.cells if family_of(c) in families)
        if not cells:
            raise ValueError(f"fuse set {name!r} has no cells in {families}")
        out[name] = cells
    return out


def fuse(per_cell: pd.DataFrame, model: str, out_model: str, cfg=config):
    """Per fuse set and fold: plain mean of the set's per-cell prob_sad."""
    rows = per_cell[per_cell["model"] == model]
    frames = []
    for name, cells in fuse_cells(cfg).items():
        for fold in range(cfg.cv.n_splits):
            part = rows[(rows["fold"] == fold) & rows["feature_set"].isin(cells)]
            wide = part.pivot(
                index="participant_id", columns="feature_set", values="prob_sad"
            )
            wide = wide.reindex(columns=list(cells))
            gaps = sorted(wide.index[wide.isna().any(axis=1)])
            if gaps:
                raise ValueError(
                    f"{model} {name} fold {fold}: participants without every "
                    f"cell of the set, e.g. {gaps[:10]}"
                )
            labels = part.drop_duplicates("participant_id").set_index("participant_id")[
                "label"
            ]
            prob = wide.mean(axis=1).to_numpy()
            frames.append(
                pd.DataFrame(
                    {
                        "analysis": cfg.per_task_v3.analysis,
                        "model": out_model,
                        "feature_set": name,
                        "fold": fold,
                        "participant_id": wide.index.to_numpy(),
                        "label": labels.loc[wide.index].to_numpy(),
                        "prob_sad": prob,
                        "pred": np.where(
                            prob >= cfg.aggregate.threshold, "SAD", "HAPPY"
                        ),
                        "in_middle_band": False,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)[list(PRED_COLUMNS)]


def select_cell(val_rows: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """Per fold, the cell whose cnn_hmm_v3 inner-val macro-F1 is highest.

    Reads inner-validation rows only; ties go to the earlier cell in
    PerTaskV3Config.cells.
    """
    v3 = cfg.per_task_v3
    rows = val_rows[val_rows["model"] == v3.hmm_model]
    out = []
    for fold in range(cfg.cv.n_splits):
        best = None
        for cell in v3.cells:
            part = rows[(rows["fold"] == fold) & (rows["feature_set"] == cell)]
            if part.empty:
                raise ValueError(f"no inner-val rows for {cell} fold {fold}")
            score = float(
                f1_score(
                    part["label"],
                    part["pred"],
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                )
            )
            if best is None or score > best[1]:
                best = (cell, score)
        out.append({"fold": fold, "cell": best[0], "inner_val_macro_f1": best[1]})
    return pd.DataFrame(out)


def selected_predictions(
    test_rows: pd.DataFrame, selected: pd.DataFrame, cfg=config
) -> pd.DataFrame:
    """Each fold's test rows of its selected cell, as cnn_hmm_v3_selected."""
    v3 = cfg.per_task_v3
    frames = []
    for _, row in selected.iterrows():
        part = test_rows[
            (test_rows["model"] == v3.hmm_model)
            & (test_rows["feature_set"] == row["cell"])
            & (test_rows["fold"] == int(row["fold"]))
        ]
        frames.append(
            part.assign(model=v3.selected_model, feature_set=v3.selected_feature_set)
        )
    return pd.concat(frames, ignore_index=True)[list(PRED_COLUMNS)]


def _sorted_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "feature_set", "fold", "participant_id"]
    return frame.sort_values(keys).reset_index(drop=True)[list(PRED_COLUMNS)]


def step_combine(cfg=config) -> Path:
    """predictions.csv, inner_val_predictions.csv and selected_cell.csv."""
    require_frozen_or_synthetic(cfg, "run_per_task_v3 combine")
    v3 = cfg.per_task_v3
    if not is_synthetic_root(cfg.paths.data_root) and tuple(v3.cells) != ALL_CELLS:
        raise ValueError("PerTaskV3Config.cells must equal crop_manifest.ALL_CELLS")
    test_rows = load_parts(cfg, "test")
    val_rows = load_parts(cfg, "val")
    selected = select_cell(val_rows, cfg)
    preds = _sorted_rows(
        pd.concat(
            [
                test_rows,
                fuse(test_rows, v3.cnn_model, v3.fused_cnn_model, cfg),
                fuse(test_rows, v3.hmm_model, v3.fused_hmm_model, cfg),
                selected_predictions(test_rows, selected, cfg),
            ],
            ignore_index=True,
        )
    )
    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "predictions.csv"
    write_part(preds, path)
    write_part(_sorted_rows(val_rows), out / "inner_val_predictions.csv")
    write_part(selected, out / "selected_cell.csv")
    evaluator.validate(
        pd.read_csv(path, dtype={"participant_id": str}), v3.analysis, str(path), cfg
    )
    counts = preds.groupby(["model", "feature_set"], sort=False).size()
    print(f"combine: {len(counts)} groups")
    for (model, fs), n in counts.items():
        print(f"  {model}/{fs}: {n} rows")
    return path


def _read_frame(path: Path, cfg=config) -> pd.DataFrame:
    analysis = cfg.per_task_v3.analysis
    df = evaluator.validate(
        pd.read_csv(path, dtype={"participant_id": str}), analysis, str(path), cfg
    )
    return evaluator.analysis_frame(df, analysis)


def _reference_keys(cfg=config) -> list:
    v3 = cfg.per_task_v3
    keys = list(v3.paired_references)
    keys += [(m, fs) for _, _, m, fs in v3.family_references]
    return list(dict.fromkeys(tuple(k) for k in keys))


def reference_frames(cfg=config) -> tuple:
    """((model, feature_set) -> primary rows, missing (model, feature_set))."""
    analysis = cfg.per_task_v3.analysis
    frames, missing, cache = {}, [], {}
    for model, fs in _reference_keys(cfg):
        source = REFERENCE_SOURCES[model]
        path = cfg.paths.results_dir / source / analysis / "predictions.csv"
        if source not in cache:
            cache[source] = _read_frame(path, cfg) if path.is_file() else None
        df = cache[source]
        rows = (
            None
            if df is None
            else df[(df["model"] == model) & (df["feature_set"] == fs)]
        )
        if rows is None or rows.empty:
            missing.append((model, fs))
        else:
            frames[(model, fs)] = rows
    return frames, missing


def _paired_row(result, label, key_a, key_b, cfg=config) -> dict:
    analysis = cfg.per_task_v3.analysis
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
        raise ValueError(f"v3 group {row['model']} was not evaluated")
    if b is None:
        row["note"] = NO_V1
        return row
    row["delta_macro_f1"], row["ci_low"], row["ci_high"] = evaluator.paired_diff_ci(
        a, b, "macro_f1", cfg.evaluation.n_bootstrap, cfg.training.seed
    )
    return row


def _task_fields(cell: str) -> tuple:
    family = family_of(cell)
    if family == "drawing":
        return family, DRAWING_NAMES[cell], ""
    if family == "cursive":
        return family, f"cursive {cell[2:]}", ""
    item, style = cell.split("_")
    return family, WORD_NAMES[int(item[1:])], style


def per_task_table(comparison: pd.DataFrame, cfg=config) -> pd.DataFrame:
    """One row per cell in PerTaskV3Config.cells order, every per-cell model."""
    imputed = imputed_counts(cfg).set_index("cell")["n_crops_with_imputed_values"]
    rows = []
    for cell in cfg.per_task_v3.cells:
        family, task, style = _task_fields(cell)
        mine = comparison[comparison["feature_set"] == cell].set_index("model")
        first = mine.loc[cfg.per_task_v3.hmm_model]
        row = {
            "cell": cell,
            "family": family,
            "task": task,
            "style": style,
            "n": int(first["n_participants"]),
            "majority_baseline": float(first["majority_baseline"]),
        }
        for model in per_cell_models(cfg):
            for m in TABLE_METRICS:
                for suffix in ("", "_ci_low", "_ci_high"):
                    row[f"{model}_{m}{suffix}"] = float(mine.loc[model, f"{m}{suffix}"])
        row["imputed_crops"] = int(imputed.get(cell, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def step_evaluate(cfg=config) -> Path:
    """metrics, per_fold, confusion, paired and per_task_table; no metric printed."""
    require_frozen_or_synthetic(cfg, "run_per_task_v3 evaluate")
    v3 = cfg.per_task_v3
    if v3.n_cnn_permutations == -1:
        raise RuntimeError("D2 undecided: do plan step 7.1 first")
    out = output_dir(cfg)
    mine = _read_frame(out / "predictions.csv", cfg)
    refs, _ = reference_frames(cfg)
    result = evaluator.evaluate(pd.concat([mine, *refs.values()]), cfg)
    models = set(v3_models(cfg))
    written = []
    for name, table in (
        ("metrics.csv", result.comparison),
        ("per_fold.csv", result.per_fold),
        ("confusion.csv", result.confusion),
    ):
        path = out / name
        table[table["model"].isin(models)].to_csv(path, index=False)
        written.append(path)

    headline = tuple(v3.headline)
    paired = [
        _paired_row(result, "headline", headline, tuple(ref), cfg)
        for ref in v3.paired_references
    ]
    paired += [
        _paired_row(result, "family", (m3, fs3), (m1, fs1), cfg)
        for m3, fs3, m1, fs1 in v3.family_references
    ]
    path = out / "paired.csv"
    pd.DataFrame(paired).to_csv(path, index=False)
    written.append(path)
    path = out / "per_task_table.csv"
    per_task_table(result.comparison, cfg).to_csv(path, index=False)
    written.append(path)
    for path in written:
        print(f"Written: {path}")
    return out


def _ci(row, m: str) -> str:
    return f"{row[m]:.3f} [{row[f'{m}_ci_low']:.3f}, {row[f'{m}_ci_high']:.3f}]"


def _permutation_line(cfg=config) -> str:
    v3 = cfg.per_task_v3
    path = output_dir(cfg) / "permutation_cnn.json"
    if path.is_file():
        perm = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"p = {perm['p_value']:.4f} (N = {perm['n']}, seed {perm['seed']}, "
            f"null mean {perm['null_mean']:.3f})"
        )
    if v3.n_cnn_permutations == 0:
        return "not run (D2 = 0)"
    done = len(list(runs_dir(cfg.training.seed, cfg).glob("perm_*.csv")))
    return f"draws incomplete ({done}/{v3.n_cnn_permutations})"


def _metric_line(row) -> str:
    parts = [f"{METRIC_NAMES[m]} {_ci(row, m)}" for m in TABLE_METRICS]
    return "; ".join(parts) + f"; majority {row['majority_baseline']:.3f}"


def _plot_forest(table: pd.DataFrame, path: Path, cfg=config) -> None:
    models = per_cell_models(cfg)
    n = len(table)
    y = np.arange(n)
    step = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7.0, 1.2 + 0.32 * n))
    fig.patch.set_facecolor(evaluator._SURFACE)
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * step
        mid = table[f"{model}_macro_f1"].to_numpy()
        err = np.vstack(
            [
                mid - table[f"{model}_macro_f1_ci_low"].to_numpy(),
                table[f"{model}_macro_f1_ci_high"].to_numpy() - mid,
            ]
        )
        ax.errorbar(
            mid,
            y + offset,
            xerr=err,
            fmt=MODEL_MARKERS[i],
            color=MODEL_COLORS[i],
            ecolor=MODEL_COLORS[i],
            elinewidth=1,
            markersize=4,
            label=model,
        )
    majority = float(table["majority_baseline"].mean())
    ax.axvline(majority, color=evaluator._MUTED, linestyle="--", linewidth=1)
    ax.text(
        majority,
        n - 0.4,
        " majority baseline",
        color=evaluator._INK_2,
        fontsize=7,
        va="bottom",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table["cell"])
    ax.set_ylim(n - 0.5, -0.5 - 0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1 (95% CI)")
    ax.set_title("v3 per-exercise macro-F1 (exploratory)")
    ax.grid(axis="x", color=evaluator._GRID, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    evaluator._style(ax)
    evaluator._save(fig, path, cfg)


def _plot_heatmap(table: pd.DataFrame, path: Path, cfg=config) -> None:
    models = per_cell_models(cfg)
    values = table[[f"{m}_macro_f1" for m in models]].to_numpy()
    n = len(table)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(models), 1.0 + 0.3 * n))
    fig.patch.set_facecolor(evaluator._SURFACE)
    image = ax.imshow(values, cmap=evaluator._SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(values):
        ax.text(
            c,
            r,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color=evaluator._SURFACE if v > 0.5 else evaluator._INK,
        )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(table["cell"])
    ax.set_title("v3 macro-F1, exercise x model (exploratory)")
    bar = fig.colorbar(image, ax=ax, fraction=0.04)
    bar.ax.tick_params(labelsize=7, colors=evaluator._INK_2)
    evaluator._style(ax)
    evaluator._save(fig, path, cfg)


def step_report(cfg=config) -> Path:
    """results/final/EXPLORATORY_v3.txt and the two per-exercise figures."""
    require_frozen_or_synthetic(cfg, "run_per_task_v3 report")
    v3 = cfg.per_task_v3
    out = output_dir(cfg)
    ev = cfg.evaluation
    metrics = pd.read_csv(out / "metrics.csv")
    paired = pd.read_csv(out / "paired.csv", keep_default_na=False)
    table = pd.read_csv(out / "per_task_table.csv", dtype={"style": str})
    table["style"] = table["style"].fillna("")
    selected = pd.read_csv(out / "selected_cell.csv")
    labels, ids, folds = context(cfg)
    lab = labels.set_index("participant_id")[cfg.labeling.label_column].loc[ids]
    keyed = metrics.set_index(["model", "feature_set"])

    lines = [
        REPORT_HEADER,
        "v3: one model per FINALE exercise (24 cells), primary set, folds.csv.",
        f"CIs: {ev.n_bootstrap} participant bootstrap resamples, seed "
        f"{cfg.training.seed}, {round(ev.ci_level * 100)}% percentile; SAD when "
        f"P(SAD) >= {cfg.aggregate.threshold}.",
        "",
        "1. Data and splits",
        f"  n = {len(ids)} ({int((lab == 'HAPPY').sum())} HAPPY, "
        f"{int((lab == 'SAD').sum())} SAD)",
        "  fold  inner-train  inner-val  test",
    ]
    for fold in range(cfg.cv.n_splits):
        inner_train, inner_val, test = fold_splits(fold, ids, labels, folds, cfg)
        lines.append(
            f"  {fold:>4d}  {len(inner_train):>11d}  {len(inner_val):>9d}  "
            f"{len(test):>4d}"
        )
    imputed = imputed_counts(cfg)
    lines.append("  crops with imputed handcrafted values per cell:")
    lines.append(
        "    "
        + ", ".join(
            f"{c} {n}"
            for c, n in zip(imputed["cell"], imputed["n_crops_with_imputed_values"])
        )
    )

    head = keyed.loc[tuple(v3.headline)]
    lines += [
        "",
        f"2. Pre-declared v3 test: {v3.headline[0]}/{v3.headline[1]}",
        f"  {_metric_line(head)}",
        f"  permutation test: {_permutation_line(cfg)}",
        "  paired macro-F1 difference, headline - reference [CI]:",
    ]
    for _, r in paired[paired["comparison"] == "headline"].iterrows():
        lines.append(f"    {r['reference']:24s} {_paired_text(r)}")

    lines += ["", "3. Fused and selected rows (descriptive)"]
    for model in (v3.fused_cnn_model, v3.fused_hmm_model, v3.selected_model):
        for (m, fs), row in keyed.iterrows():
            if m == model:
                lines.append(f"  {m}/{fs}: {_metric_line(row)}")
    lines.append("  selected cell per fold (inner-val macro-F1 of cnn_hmm_v3):")
    for _, s in selected.iterrows():
        lines.append(
            f"    fold {int(s['fold'])}: {s['cell']} ({s['inner_val_macro_f1']:.3f})"
        )

    lines += ["", "4. Per exercise, form order (macro-F1 [CI])"]
    models = per_cell_models(cfg)
    lines.append("  cell    task          style  " + "  ".join(models))
    for _, r in table.iterrows():
        cells = "  ".join(_ci(r, f"{m}_macro_f1") for m in models)
        lines.append(f"  {r['cell']:6s}  {r['task']:12s}  {r['style']:5s}  {cells}")
    lines.append(
        f"  majority baseline: {float(table['majority_baseline'].iloc[0]):.3f}"
    )
    lines.append("  accuracy, balanced accuracy and ROC-AUC: per_task_table.csv")

    lines += ["", "5. LR permutation tests per cell (p, BH q)"]
    perm_lr = out / "permutation_lr.csv"
    if perm_lr.is_file():
        for _, r in pd.read_csv(perm_lr).iterrows():
            lines.append(
                f"  {r['kind']} {r['cell']}: p = {r['p_value']:.4f}, "
                f"q = {r['q_value']:.4f}"
            )
    else:
        lines.append("  not run yet")

    lines += ["", "6. Family rows vs v1, paired macro-F1 difference [CI]"]
    for _, r in paired[paired["comparison"] == "family"].iterrows():
        lines.append(f"  {r['model']:28s} vs {r['reference']:20s} {_paired_text(r)}")

    forest = figure_path("per_task_forest.png", cfg)
    heatmap = figure_path("per_task_heatmap.png", cfg)
    rel = out.relative_to(cfg.paths.results_dir.parent).as_posix()
    lines += [
        "",
        LIMITATIONS,
        "",
        f"Sources: {rel}/ (predictions, selected_cell, metrics, per_fold, "
        "confusion, paired, per_task_table, permutation files); figures "
        f"{forest.name}, {heatmap.name}.",
    ]
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_forest(table, forest, cfg)
    _plot_heatmap(table, heatmap, cfg)
    for p in (path, forest, heatmap):
        print(f"Written: {p}")
    return path


def _paired_text(row) -> str:
    if row["note"]:
        return row["note"]
    return (
        f"{float(row['delta_macro_f1']):+.3f} "
        f"[{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}]"
    )


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _macro_f1(y_true, y_pred) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
    )


def _label_indices(labels: pd.DataFrame, pids, cfg=config) -> np.ndarray:
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    return lab.loc[[str(p) for p in pids]].map(cfg.data.label_to_index).to_numpy()


def _lr_null_draw(kind, cell, ids, labels, folds, cfg, y_trains, X) -> float:
    """Pooled test macro-F1 of one draw: every fold refit on shuffled labels."""
    lab = labels.set_index("participant_id")[cfg.labeling.label_column]
    y_true, y_pred = [], []
    for fold, y_train in enumerate(y_trains):
        pipe, _, _, test = fit_lr_cell(
            kind, cell, fold, ids, labels, folds, cfg, y_train=y_train, X=X
        )
        prob = _prob_sad(pipe, X.loc[test])
        y_true += list(lab.loc[test])
        y_pred += list(np.where(prob >= cfg.aggregate.threshold, "SAD", "HAPPY"))
    return _macro_f1(y_true, y_pred)


def lr_null(
    kind: str,
    cell: str,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    cfg=config,
    n: int | None = None,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Null macro-F1 of one (LR kind, cell) for draws 0..n-1.

    Draw k, fold f: the outer-train labels are permuted among themselves
    (run_cnn.shuffled_labels, permutation_seed(seed, k, f)) and the LR is
    refit by fit_lr_cell exactly as in the observed run; the test rows are
    scored against their true labels. The cell's features are loaded once
    here and the permuted label vectors drawn here; joblib workers only fit.
    """
    n = cfg.per_task_v3.n_lr_permutations if n is None else n
    seed = cfg.training.seed if seed is None else seed
    n_jobs = cfg.runner.hmm_n_jobs if n_jobs is None else n_jobs
    X, _ = _lr_inputs(kind, cell, cfg)
    trains = [fold_ids(folds, f, ids)[0] for f in range(cfg.cv.n_splits)]
    plans = [
        [
            _label_indices(
                shuffled_labels(labels, train, permutation_seed(seed, k, f)),
                train,
                cfg,
            )
            for f, train in enumerate(trains)
        ]
        for k in range(n)
    ]
    jobs = (
        delayed(_lr_null_draw)(kind, cell, ids, labels, folds, cfg, y_trains, X)
        for y_trains in plans
    )
    return np.asarray(Parallel(n_jobs=n_jobs)(jobs), dtype=float)


def _require_predictions(cfg=config) -> pd.DataFrame:
    path = output_dir(cfg) / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run --step combine first")
    return _read_frame(path, cfg)


def _observed(preds: pd.DataFrame, model: str, fs: str, cfg=config) -> float:
    rows = preds[(preds["model"] == model) & (preds["feature_set"] == fs)]
    if set(rows["fold"]) != set(range(cfg.cv.n_splits)):
        raise ValueError(f"predictions.csv: {model}/{fs} does not cover all folds")
    return evaluator.metrics(rows)["macro_f1"]


def _save_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    tmp.replace(path)


def lr_null_path(kind: str, cell: str, cfg=config) -> Path:
    return output_dir(cfg) / "permutation_lr" / f"{kind}__{cell}.npy"


def step_permutation_lr(cfg=config, n_jobs: int | None = None) -> Path:
    """LR permutation test per (kind, cell); writes permutation_lr.csv.

    Prints only progress and timing, never an observed value, p or q.
    """
    require_frozen_or_synthetic(cfg, "run_per_task_v3 permutation-lr")
    v3 = cfg.per_task_v3
    preds = _require_predictions(cfg)
    labels, ids, folds = context(cfg)
    n, seed = v3.n_lr_permutations, cfg.training.seed
    units = [(kind, cell) for kind in v3.lr_kinds for cell in v3.cells]
    start, first, computed = time.perf_counter(), None, 0
    rows = []
    for i, (kind, cell) in enumerate(units):
        path = lr_null_path(kind, cell, cfg)
        null = np.load(path) if path.is_file() else None
        if null is not None and len(null) == n:
            print(f"resume: {kind} {cell}")
        else:
            began = time.perf_counter()
            null = lr_null(kind, cell, ids, labels, folds, cfg, n, seed, n_jobs)
            _save_npy(null, path)
            computed += 1
            took = time.perf_counter() - began
            if first is None:
                first = took
                todo = sum(
                    1
                    for k, c in units[i + 1 :]
                    if not _cached_null(lr_null_path(k, c, cfg), n)
                )
                print(
                    f"permutation-lr: {kind} {cell} took {took:.1f} s; projected "
                    f"{first * todo / 60:.1f} min for the {todo} left",
                    flush=True,
                )
            else:
                print(f"permutation-lr: {kind} {cell}: {took:.1f} s", flush=True)
        observed = _observed(preds, kind, cell, cfg)
        rows.append(
            {
                "kind": kind,
                "cell": cell,
                "p_value": p_value(null, observed),
                "observed": observed,
                "n": n,
                "seed": seed,
            }
        )
    table = pd.DataFrame(rows)
    table["q_value"] = np.nan
    for kind in v3.lr_kinds:
        mine = table["kind"] == kind
        table.loc[mine, "q_value"] = bh_qvalues(table.loc[mine, "p_value"])
    table = table[["kind", "cell", "p_value", "q_value", "observed", "n", "seed"]]
    out = output_dir(cfg) / "permutation_lr.csv"
    write_part(table, out)
    print(
        f"permutation-lr: {computed} computed, {len(units) - computed} cached, "
        f"{(time.perf_counter() - start) / 60:.1f} min; written {out}"
    )
    return out


def _cached_null(path: Path, n: int) -> bool:
    return path.is_file() and len(np.load(path)) == n


def draw(
    perm: int,
    seed: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    manifest: pd.DataFrame,
    dropped: set,
    cfg=config,
    device: str = "cpu",
) -> pd.DataFrame:
    """One CNN-path null draw: every cell x fold retrained on shuffled labels.

    Fold f uses shuffle_seed permutation_seed(seed, perm, f) for both the
    CNN and the HMM; the inner split stays fold_splits on the true labels.
    Checkpoints live in a temporary folder under models/ that is removed
    afterwards; the HMMs are not saved. Returns the per-cell and fused test
    rows of cnn_head_v3 and cnn_hmm_v3 with a "perm" column.
    """
    v3 = cfg.per_task_v3
    rows = []
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cfg.paths.models_dir) as tmp:
        for cell in v3.cells:
            for fold in range(cfg.cv.n_splits):
                shuffle = permutation_seed(seed, perm, fold)
                ckpt = Path(tmp) / f"{cell}_fold{fold}.pth"
                cnn_test, _ = fit_cnn_cell(
                    cell,
                    fold,
                    ids,
                    labels,
                    folds,
                    manifest,
                    dropped,
                    device,
                    ckpt,
                    cfg,
                    shuffle_seed=shuffle,
                )
                hmm_test, _, _ = fit_hmm_cell(
                    cell,
                    fold,
                    ids,
                    labels,
                    folds,
                    manifest,
                    dropped,
                    device,
                    ckpt,
                    cfg,
                    shuffle_seed=shuffle,
                    save_path=None,
                )
                ckpt.unlink()
                rows += [cnn_test, hmm_test]
    per_cell = pd.concat(rows, ignore_index=True)
    frame = pd.concat(
        [
            per_cell,
            fuse(per_cell, v3.cnn_model, v3.fused_cnn_model, cfg),
            fuse(per_cell, v3.hmm_model, v3.fused_hmm_model, cfg),
        ],
        ignore_index=True,
    )
    return _sorted_rows(frame).assign(perm=perm)


def draw_settings(cfg=config, seed: int | None = None) -> dict:
    seed = cfg.training.seed if seed is None else seed
    return {
        **settings(cfg),
        "seed": seed,
        "analysis": cfg.per_task_v3.analysis,
        "kind": DRAWS_KIND,
    }


def draw_path(perm: int, seed: int, cfg=config) -> Path:
    return runs_dir(seed, cfg) / f"perm_{perm:04d}.csv"


def _n_cnn(cfg=config) -> int:
    n = cfg.per_task_v3.n_cnn_permutations
    if n == -1:
        raise RuntimeError("D2 undecided: do plan step 7.1 first")
    return n


def step_draws(cfg=config, draws: str | None = None, device="auto") -> Path | None:
    """Cache CNN-path null draws (one perm_<k>.csv each); resumable, mergeable."""
    require_frozen_or_synthetic(cfg, "run_per_task_v3 draws")
    n = _n_cnn(cfg)
    if n == 0:
        print("D2 = 0: no CNN-path test")
        return None
    perms = parse_draws(draws if draws else f"0-{n - 1}", n)
    seed = cfg.training.seed
    out = runs_dir(seed, cfg)
    check_resume(out, draw_settings(cfg, seed))
    device = resolve_device(device)
    labels, ids, folds = context(cfg)
    manifest, _, dropped = load_tables(cfg)
    for perm in perms:
        path = draw_path(perm, seed, cfg)
        if path.is_file():
            print(f"resume: draw {perm}")
            continue
        began = time.perf_counter()
        frame = draw(perm, seed, ids, labels, folds, manifest, dropped, cfg, device)
        write_part(frame, path)
        print(f"draw {perm}: {time.perf_counter() - began:.1f} s", flush=True)
    done = len(list(out.glob("perm_*.csv")))
    print(f"draws: {done}/{n} present in {out}")
    return out


def step_permutation_cnn(cfg=config) -> Path:
    """Permutation test of the headline over all cached draws."""
    require_frozen_or_synthetic(cfg, "run_per_task_v3 permutation-cnn")
    v3 = cfg.per_task_v3
    n = _n_cnn(cfg)
    seed = cfg.training.seed
    paths = [draw_path(k, seed, cfg) for k in range(n)]
    done = sum(p.is_file() for p in paths)
    if n == 0 or done < n:
        raise RuntimeError(f"draws incomplete ({done}/{n})")
    preds = _require_predictions(cfg)
    model, fs = v3.headline
    observed = _observed(preds, model, fs, cfg)
    null, per_cell = [], {}
    for path in paths:
        frame = pd.read_csv(path, dtype={"participant_id": str})
        null.append(_observed(frame, model, fs, cfg))
        for m in (v3.cnn_model, v3.hmm_model):
            for cell in v3.cells:
                per_cell.setdefault(f"{m}/{cell}", []).append(
                    _observed(frame, m, cell, cfg)
                )
    null = np.asarray(null, dtype=float)
    p = p_value(null, observed)
    result = {
        "model": model,
        "feature_set": fs,
        "analysis": v3.analysis,
        "observed_macro_f1": observed,
        "p_value": p,
        "n": n,
        "seed": seed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if n > 1 else 0.0,
        "draws_used": [int(path.stem.split("_")[1]) for path in paths],
        "runs_dir": str(runs_dir(seed, cfg)),
        "per_cell_null_macro_f1": {"note": DESCRIPTIVE_ONLY, "null": per_cell},
    }
    out = output_dir(cfg) / "permutation_cnn.json"
    _write_json(result, out)
    figure = evaluator.output_dir(None, cfg) / PERMUTATION_FIGURE
    figure.parent.mkdir(parents=True, exist_ok=True)
    _plot(
        null,
        observed,
        p,
        f"v3 {model}/{fs}: permutation null (N = {n}, exploratory)",
        figure,
        cfg,
    )
    print(f"Written: {out}")
    print(f"Written: {figure}")
    return out


def _parse_list(text: str | None, cast=str) -> list | None:
    if not text:
        return None
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-exercise training v3, exploratory (Deviation 6)."
    )
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--cells", default=None, help="comma list, e.g. D1,W2_RH")
    parser.add_argument("--folds", default=None, help="comma list, e.g. 0,1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--draws", default=None, help="e.g. 0-9 or 7")
    args = parser.parse_args()
    if args.n_jobs is not None:
        config.runner.hmm_n_jobs = args.n_jobs
    steps = {
        "preflight": lambda units: preflight(config),
        "lr": lambda units: step_lr(config),
        "cnn": lambda units: step_cnn(config, **units),
        "hmm": lambda units: step_hmm(config, **units),
        "combine": lambda units: step_combine(config),
        "evaluate": lambda units: step_evaluate(config),
        "report": lambda units: step_report(config),
        "permutation-lr": lambda units: step_permutation_lr(config, args.n_jobs),
        "draws": lambda units: step_draws(config, args.draws, args.device),
        "permutation-cnn": lambda units: step_permutation_cnn(config),
    }
    try:
        units = {
            "cells": _parse_list(args.cells),
            "folds": _parse_list(args.folds, int),
            "device": args.device,
        }
        steps[args.step](units)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
