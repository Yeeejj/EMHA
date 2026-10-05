"""
Participant-level cross-validation folds and leakage checks — Stage E.

make_folds assigns every labeled participant (labels.csv) to exactly one of
TrainingConfig.n_folds outer test folds, stratified on SplitConfig
.stratify_columns (label x in_primary_analysis). Within each outer fold, a
stratified SplitConfig.inner_val_fraction of the remaining (training)
participants becomes the inner validation set, used only for early stopping
and Platt calibration -- never the outer test fold.

folds.csv (DATASET/metadata/) is long format, one row per (participant,
outer fold):

    participant_id, label, in_primary_analysis, outer_fold, role

with role in {train, inner_val, test}. Units are participants, never crops,
so a participant's crops can never straddle train and test. The primary
(extreme-groups) analysis restricts every role to in_primary_analysis=True;
the full-sample analysis uses all rows.

assert_no_leakage must be called in every fold (CLAUDE.md Non-Negotiable 1);
fold_ids calls it for you.

Run from the project root (refuses to change an existing, different
folds.csv -- folds are fixed before any result):

    python -m src.training.splits
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.config import config

ROLES = ("train", "inner_val", "test")
FOLD_COLUMNS = ("participant_id", "label", "in_primary_analysis", "outer_fold", "role")


class LeakageError(AssertionError):
    """A participant appears in more than one of train / validation / test."""


def assert_no_leakage(train_ids, test_ids, val_ids=None) -> None:
    """Raise LeakageError if any participant is in more than one set.

    Works on participant IDs (strings); pass crop tables' participant_id
    columns, not crop indices. Raises (not `assert`) so it survives -O.
    """
    sets = {
        "train": {str(p) for p in train_ids},
        "test": {str(p) for p in test_ids},
    }
    if val_ids is not None:
        sets["val"] = {str(p) for p in val_ids}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sets[a] & sets[b]
            if shared:
                raise LeakageError(
                    f"participants in both {a} and {b}: {sorted(shared)[:10]}"
                    f"{' ...' if len(shared) > 10 else ''} ({len(shared)} total)"
                )


def _strata(df: pd.DataFrame, columns) -> np.ndarray:
    return df[list(columns)].astype(str).agg("|".join, axis=1).to_numpy()


def make_folds(labels: pd.DataFrame, cfg) -> pd.DataFrame:
    """Long-format fold table for every participant in labels.

    labels needs participant_id, the label column, and the primary column.
    Deterministic for a given cfg.training.seed.
    """
    scfg = cfg.splits
    label_col = cfg.labeling.label_column
    base = labels[["participant_id", label_col, scfg.primary_column]].copy()
    base = base.rename(columns={label_col: "label"})
    base["participant_id"] = base["participant_id"].astype(str)
    if base["participant_id"].duplicated().any():
        raise ValueError("labels has duplicate participant_id rows")
    if base[["label", scfg.primary_column]].isna().any().any():
        raise ValueError("labels has missing label or primary-set values")
    base[scfg.primary_column] = base[scfg.primary_column].astype(bool)
    base = base.sort_values("participant_id").reset_index(drop=True)

    strata = _strata(base, scfg.stratify_columns)
    seed = cfg.training.seed
    outer = StratifiedKFold(
        n_splits=cfg.training.n_folds, shuffle=True, random_state=seed
    )

    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(base, strata)):
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx,
            test_size=scfg.inner_val_fraction,
            stratify=strata[train_idx],
            random_state=seed + fold,
        )
        for role, idx in (
            ("train", inner_train_idx),
            ("inner_val", inner_val_idx),
            ("test", test_idx),
        ):
            part = base.iloc[np.sort(idx)].copy()
            part["outer_fold"] = fold
            part["role"] = role
            rows.append(part)

    folds = pd.concat(rows, ignore_index=True)
    folds = folds.rename(columns={scfg.primary_column: "in_primary_analysis"})
    folds = folds[list(FOLD_COLUMNS)].sort_values(["outer_fold", "participant_id"])
    folds = folds.reset_index(drop=True)
    validate_folds(folds, cfg.training.n_folds)
    return folds


def validate_folds(folds: pd.DataFrame, n_folds: int) -> None:
    """Raise if folds.csv is not a valid participant-level partition.

    Checks: every participant has exactly one row per outer fold, roles are
    valid, each participant is the test set of exactly one outer fold, and
    train / inner_val / test are disjoint within every fold.
    """
    missing = set(FOLD_COLUMNS) - set(folds.columns)
    if missing:
        raise ValueError(f"folds lacks columns: {sorted(missing)}")
    if not set(folds["role"]) <= set(ROLES):
        raise ValueError(f"unknown roles: {sorted(set(folds['role']) - set(ROLES))}")
    if sorted(folds["outer_fold"].unique()) != list(range(n_folds)):
        raise ValueError(f"outer_fold must be 0..{n_folds - 1}")

    per_fold = folds.groupby("participant_id")["outer_fold"].agg(["count", "nunique"])
    if not ((per_fold["count"] == n_folds) & (per_fold["nunique"] == n_folds)).all():
        raise ValueError("every participant needs exactly one row per outer fold")
    test_count = folds[folds["role"] == "test"].groupby("participant_id").size()
    if len(test_count) != len(per_fold) or not (test_count == 1).all():
        raise ValueError("every participant must be tested in exactly one fold")

    for fold in range(n_folds):
        ids = fold_ids(folds, fold)
        assert_no_leakage(ids["train"], ids["test"], ids["inner_val"])


def fold_ids(folds: pd.DataFrame, outer_fold: int, primary_only: bool = False) -> dict:
    """{'train', 'inner_val', 'test'} -> participant ID lists for one outer fold.

    primary_only=True restricts every role to the extreme-groups primary set.
    Calls assert_no_leakage before returning.
    """
    part = folds[folds["outer_fold"] == outer_fold]
    if primary_only:
        part = part[part["in_primary_analysis"].astype(bool)]
    ids = {
        role: sorted(part.loc[part["role"] == role, "participant_id"]) for role in ROLES
    }
    assert_no_leakage(ids["train"], ids["test"], ids["inner_val"])
    return ids


def folds_path(cfg) -> Path:
    return cfg.paths.metadata_dir / cfg.splits.folds_filename


def load_folds(cfg) -> pd.DataFrame:
    """Read and validate folds.csv."""
    folds = pd.read_csv(folds_path(cfg), dtype={"participant_id": str})
    validate_folds(folds, cfg.training.n_folds)
    return folds


def _summary(folds: pd.DataFrame) -> pd.DataFrame:
    test = folds[folds["role"] == "test"]
    table = pd.crosstab(
        test["outer_fold"], [test["in_primary_analysis"], test["label"]]
    )
    table.columns = [
        f"{'primary' if p else 'middle'}_{lab}" for p, lab in table.columns
    ]
    roles = folds.groupby(["outer_fold", "role"]).size().unstack()[list(ROLES)]
    return pd.concat([roles.add_prefix("n_"), table], axis=1)


def run() -> Path:
    """Write folds.csv from labels.csv; refuse to change an existing different one."""
    from src.data.dataloader import labels_csv_path

    labels = pd.read_csv(labels_csv_path(config), dtype={"participant_id": str})
    folds = make_folds(labels, config)
    out = folds_path(config)

    if out.is_file():
        existing = pd.read_csv(out, dtype={"participant_id": str})
        if existing.equals(folds.astype(existing.dtypes.to_dict())):
            print(f"folds.csv unchanged: {out}")
            return out
        raise RuntimeError(
            f"{out} exists and differs from the folds this code would write. "
            "Folds are fixed before any result; changing them is a logged "
            "Deviation. Remove the file yourself only if that is intended."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, index=False)
    print(f"Participants : {folds['participant_id'].nunique()}")
    print(f"Folds        : {config.training.n_folds} (seed {config.training.seed})")
    print(_summary(folds).to_string())
    print(f"Written      : {out}")
    return out


def main() -> int:
    try:
        run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
