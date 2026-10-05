"""
The one fold assignment, inner splits, and leakage checks — Stage E.

Every runner gets its participants from here:

    folds  = load_folds(cfg)                                  # folds.csv
    ids    = analysis_ids(labels, participants, "primary")    # or "full"
    train, test = fold_ids(folds, k, ids)                     # outer fold k
    inner_train, inner_val = inner_split(train, labels,
                                         cfg.cv.inner_val_fraction,
                                         cfg.training.seed + k)
    assert_no_leakage(inner_train, inner_val, test)

make_outer_folds assigns every labeled participant (labels.csv) to exactly
one of CVConfig.n_splits outer folds. Splitting a one-row-per-participant
table with sklearn StratifiedKFold keeps all of a participant's crops on one
side by construction; strata are label x in_middle_band (CVConfig
.stratify_columns), so both the primary (extreme-groups) and full-sample
analyses get balanced folds from the same assignment.

folds.csv (DATASET/metadata/): one row per participant —
participant_id, label, in_middle_band, in_primary_analysis, outer_fold.
Runners filter it to analysis_ids at load time; the inner split is derived
from the filtered training IDs, so it never needs storing.

PROVISIONAL: more participants are still being labeled; regenerate with
--force once labeling is complete, before the protocol freeze.

Run from the project root:

    python -m src.training.splits            # write, or verify unchanged
    python -m src.training.splits --force    # overwrite a different folds.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.config import config

FOLD_COLUMNS = (
    "participant_id",
    "label",
    "in_middle_band",
    "in_primary_analysis",
    "outer_fold",
)
ANALYSES = ("primary", "full")
SUBSETS = ("pilot",)
QC_PASSED = "qc_passed"


class LeakageError(AssertionError):
    """A participant appears in more than one of the given ID lists."""


def assert_no_leakage(*id_lists) -> None:
    """Raise LeakageError if any participant ID is in more than one list.

    IDs are compared as strings. Raises explicitly (not `assert`) so the
    check survives python -O. Call it in every fold, e.g.
    assert_no_leakage(inner_train, inner_val, test).
    """
    sets = [{str(p) for p in ids} for ids in id_lists]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            if shared:
                shown = sorted(shared)[:10]
                raise LeakageError(
                    f"participants in both id list {i} and id list {j}: {shown}"
                    f"{' ...' if len(shared) > 10 else ''} ({len(shared)} total)"
                )


_BOOL_TEXT = {"true": True, "false": False, "1": True, "0": False}


def _as_bool(values: pd.Series, name: str) -> pd.Series:
    """Parse a flag column strictly; "False" must never become True."""
    if values.dtype == bool:
        return values
    text = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - set(_BOOL_TEXT))
    if unknown:
        raise ValueError(f"{name} has non-boolean values: {unknown[:5]}")
    return text.map(_BOOL_TEXT).astype(bool)


def _participant_table(labels: pd.DataFrame, cfg=config) -> pd.DataFrame:
    label_col = cfg.labeling.label_column
    needed = {"participant_id", label_col, "in_middle_band", "in_primary_analysis"}
    missing = needed - set(labels.columns)
    if missing:
        raise ValueError(f"labels lacks columns: {sorted(missing)}")
    table = labels[list(needed)].rename(columns={label_col: "label"}).copy()
    table["participant_id"] = table["participant_id"].astype(str)
    if table["participant_id"].duplicated().any():
        raise ValueError("labels has duplicate participant_id rows")
    if table.isna().any().any():
        raise ValueError("labels has missing label / band values")
    for col in ("in_middle_band", "in_primary_analysis"):
        table[col] = _as_bool(table[col], col)
    return table.sort_values("participant_id").reset_index(drop=True)


def _strata(table: pd.DataFrame, columns) -> np.ndarray:
    return table[list(columns)].astype(str).agg("|".join, axis=1).to_numpy()


def make_outer_folds(labels: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """One row per labeled participant with its outer_fold (0..n_splits-1).

    Stratified on CVConfig.stratify_columns; independent of the row order
    of labels; identical for the same seed.
    """
    table = _participant_table(labels)
    strata = _strata(table, config.cv.stratify_columns)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    table["outer_fold"] = -1
    for fold, (_, test_idx) in enumerate(skf.split(table, strata)):
        table.loc[test_idx, "outer_fold"] = fold
    folds = table[list(FOLD_COLUMNS)]
    validate_folds(folds, n_splits)
    return folds


def inner_split(
    train_ids: list, labels: pd.DataFrame, val_fraction: float, seed: int
) -> tuple:
    """(inner_train_ids, inner_val_ids) from an outer fold's training IDs.

    Stratified on CVConfig.stratify_columns; if a stratum has fewer than 2
    participants (possible after qc_passed / primary filtering), falls back
    to stratifying on label alone. Deterministic for the same inputs.
    """
    ids = sorted({str(p) for p in train_ids})
    table = _participant_table(labels).set_index("participant_id")
    unknown = set(ids) - set(table.index)
    if unknown:
        raise ValueError(f"train_ids without labels: {sorted(unknown)[:10]}")
    sub = table.loc[ids].reset_index()

    strata = _strata(sub, config.cv.stratify_columns)
    if pd.Series(strata).value_counts().min() < 2:
        strata = sub["label"].astype(str).to_numpy()
    inner_train, inner_val = train_test_split(
        ids, test_size=val_fraction, stratify=strata, random_state=seed
    )
    inner_train, inner_val = sorted(inner_train), sorted(inner_val)
    assert_no_leakage(inner_train, inner_val)
    return inner_train, inner_val


def analysis_ids(
    labels: pd.DataFrame, participants: pd.DataFrame, analysis: str
) -> list:
    """Participant IDs in an analysis set.

    "primary" = in_primary_analysis and status qc_passed;
    "full"    = every qc_passed participant with a label.
    """
    if analysis not in ANALYSES:
        raise ValueError(f"analysis must be one of {ANALYSES}, got {analysis!r}")
    table = _participant_table(labels)
    status = participants.assign(
        participant_id=participants["participant_id"].astype(str)
    )
    passed = set(status.loc[status["status"] == QC_PASSED, "participant_id"])
    keep = table["participant_id"].isin(passed)
    if analysis == "primary":
        keep &= table["in_primary_analysis"]
    return sorted(table.loc[keep, "participant_id"])


def restrict_to_subset(ids: list, subset: str | None, cfg=config) -> list:
    """Keep only the IDs of a named subset ("pilot" = CVConfig.pilot_id_range)."""
    if subset is None:
        return list(ids)
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    lo, hi = cfg.cv.pilot_id_range
    return [p for p in ids if lo <= int(p) <= hi]


def middle_band_ids(
    labels: pd.DataFrame, participants: pd.DataFrame, subset: str | None = None
) -> list:
    """qc_passed participants in the middle band (never in the primary set)."""
    table = _participant_table(labels)
    full = set(analysis_ids(labels, participants, "full"))
    band = table.loc[table["in_middle_band"], "participant_id"]
    return restrict_to_subset(sorted(full & set(band)), subset)


def fold_ids(folds: pd.DataFrame, fold: int, ids: list) -> tuple:
    """(train_ids, test_ids) for outer fold `fold`, restricted to `ids`.

    test = participants of `ids` assigned to `fold`; train = the rest of
    `ids`. IDs absent from folds.csv raise. Calls assert_no_leakage.
    """
    wanted = {str(p) for p in ids}
    known = set(folds["participant_id"].astype(str))
    if wanted - known:
        raise ValueError(f"ids not in folds.csv: {sorted(wanted - known)[:10]}")
    part = folds[folds["participant_id"].astype(str).isin(wanted)]
    in_fold = part["outer_fold"] == fold
    test = sorted(part.loc[in_fold, "participant_id"].astype(str))
    train = sorted(part.loc[~in_fold, "participant_id"].astype(str))
    assert_no_leakage(train, test)
    return train, test


def validate_folds(folds: pd.DataFrame, n_splits: int) -> None:
    """Raise unless every participant has exactly one fold in 0..n_splits-1."""
    missing = set(FOLD_COLUMNS) - set(folds.columns)
    if missing:
        raise ValueError(f"folds lacks columns: {sorted(missing)}")
    if folds["participant_id"].astype(str).duplicated().any():
        raise ValueError("a participant appears in more than one fold row")
    if not folds["outer_fold"].isin(range(n_splits)).all():
        raise ValueError(f"outer_fold must be in 0..{n_splits - 1}")
    if folds["outer_fold"].nunique() != n_splits:
        raise ValueError("every outer fold must contain participants")


def folds_path(cfg=config) -> Path:
    return cfg.paths.metadata_dir / cfg.cv.folds_filename


def load_folds(cfg=config) -> pd.DataFrame:
    """Read and validate folds.csv."""
    folds = pd.read_csv(folds_path(cfg), dtype={"participant_id": str})
    validate_folds(folds, cfg.cv.n_splits)
    return folds


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, part in folds.groupby("outer_fold"):
        primary = part[part["in_primary_analysis"]]
        rows.append(
            {
                "outer_fold": fold,
                "n_full": len(part),
                "sad_pct_full": round(100 * (part["label"] == "SAD").mean(), 1),
                "n_primary": len(primary),
                "sad_pct_primary": round(100 * (primary["label"] == "SAD").mean(), 1),
            }
        )
    return pd.DataFrame(rows).set_index("outer_fold")


def run(force: bool = False) -> Path:
    """Write folds.csv from labels.csv and print its SHA-256.

    An identical existing file is left as is. A different existing file is
    only replaced with force=True (CLI --force), which needs the thesis
    author's confirmation: changing folds after any result is a Deviation.
    """
    from src.data.dataloader import labels_csv_path

    labels = pd.read_csv(labels_csv_path(config), dtype={"participant_id": str})
    folds = make_outer_folds(labels, config.cv.n_splits, config.training.seed)
    out = folds_path(config)

    if out.is_file():
        existing = pd.read_csv(out, dtype={"participant_id": str})
        same = list(existing.columns) == list(folds.columns) and existing.equals(
            folds.reset_index(drop=True).astype(existing.dtypes.to_dict())
        )
        if same:
            print(f"folds.csv unchanged : {out}")
            print(f"SHA-256             : {sha256_of(out)}")
            return out
        if not force:
            raise RuntimeError(
                f"{out} exists (SHA-256 {sha256_of(out)}) and differs from the "
                "folds this code would write. Re-run with --force only with the "
                "thesis author's confirmation; after any result this is a "
                "logged Deviation."
            )
        print(f"Overwriting (--force): {out} (old SHA-256 {sha256_of(out)})")

    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, index=False)
    print(f"Participants : {len(folds)} (PROVISIONAL until labeling is complete)")
    print(f"Folds        : {config.cv.n_splits} (seed {config.training.seed})")
    print(_summary(folds).to_string())
    print(f"Written      : {out}")
    print(f"SHA-256      : {sha256_of(out)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the participant fold file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a different existing folds.csv (needs author confirmation).",
    )
    args = parser.parse_args()
    try:
        run(force=args.force)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
