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

Nothing is appended to results/RUN_LOG.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.crop_manifest import ALL_CELLS, CURSIVE_CELLS, DRAWING_CELLS
from src.data.dataloader import CropDataset, labels_csv_path, load_tables
from src.data.transforms import get_eval_transform, get_train_transform
from src.features.handcrafted import features_for_family
from src.training import evaluator
from src.training.run_baselines import PRED_COLUMNS, _prob_sad, fit_lr
from src.training.splits import (
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    folds_path,
    inner_split,
    load_folds,
)
from src.utils.config import config
from src.utils.protocol_guard import is_synthetic_root, require_frozen_or_synthetic

FOLDS_SHA256 = "40403812420cde0e1f7d1980ce02ad0f77c531fdf59aab62b7987ca4b90751bf"
LABELS_SHA256 = "694d360091e28dc331422cf7171b8e3afcf5dda16545e7ad5b2a9291204d3006"
FAMILIES = ("drawing", "word", "cursive")
HANDCRAFTED_CROP = "handcrafted_crop.csv"
PROCESSED_MANIFEST = "processed_manifest.csv"
IMPUTED_COLUMN = "imputed"
UNDECIDED_KEYS = ("n_cnn_permutations", "n_lr_permutations")
STEPS = ("preflight", "lr")


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


def write_part(frame: pd.DataFrame, path: Path) -> None:
    """Write frame to path atomically (.tmp, then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _lr_inputs(kind: str, cell: str, cfg=config) -> tuple:
    """(feature table indexed by participant_id, use PCA) for one LR kind."""
    hand_kind, emb_kind = cfg.per_task_v3.lr_kinds
    if kind == hand_kind:
        return handcrafted_cell(cell, cfg), False
    if kind == emb_kind:
        return embedding_cell(cell, cfg), True
    raise ValueError(f"unknown LR kind {kind!r}; expected {cfg.per_task_v3.lr_kinds}")


def fit_lr_cell(
    kind: str,
    cell: str,
    fold: int,
    ids: list,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    cfg=config,
    y_train=None,
) -> tuple:
    """(fitted pipeline, features, outer-train IDs, test IDs) of one LR fit.

    Fit on the outer-train participants only (C by inner CV inside fit_lr,
    as in run_baselines); y_train None means their true label indices.
    """
    X, use_pca = _lr_inputs(kind, cell, cfg)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-exercise training v3, exploratory (Deviation 6)."
    )
    parser.add_argument("--step", choices=STEPS, required=True)
    args = parser.parse_args()
    steps = {"preflight": preflight, "lr": step_lr}
    try:
        steps[args.step]()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
