"""
Participant-level cross-validation runner — Stage E.

CrossValidator is model-agnostic. It reads the one fold assignment
(folds.csv via src.training.splits), restricts it to an analysis set
("primary" = extreme-groups and qc_passed, "full" = all qc_passed), and for
every outer fold builds a FoldSplit:

    inner_train_ids  fit models / scalers / PCA here only
    inner_val_ids    early stopping and Platt calibration only
    test_ids         predict only; never used for any fitting or tuning

assert_no_leakage(inner_train, inner_val, test) is called for every fold.
The inner split is splits.inner_split(train, labels, inner_val_fraction,
seed + fold). Model runners (Stage F) pass a fit_predict(split) callable to
run(); each fold's predictions must cover exactly that fold's test
participants (none missing, none extra), so no participant is dropped or
scored outside its test fold.

Fold assignment lives only in src/training/splits.py; nothing here makes
folds or splits crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import pandas as pd

from src.data.dataloader import labels_csv_path
from src.training.splits import (
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    inner_split,
    load_folds,
)
from src.utils.config import config


@dataclass(frozen=True)
class FoldSplit:
    """Participant IDs for one outer fold of one analysis."""

    analysis: str
    fold: int
    inner_train_ids: tuple
    inner_val_ids: tuple
    test_ids: tuple


class CrossValidator:
    """Iterate the frozen outer folds for one analysis set.

    folds / labels / participants default to folds.csv, labels.csv, and
    participants.csv under config.paths.metadata_dir; pass DataFrames to
    override (tests).
    """

    def __init__(
        self,
        analysis: str,
        cfg=config,
        folds: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
        participants: pd.DataFrame | None = None,
    ):
        self.analysis = analysis
        self.cfg = cfg
        meta = cfg.paths.metadata_dir
        self.folds = folds if folds is not None else load_folds(cfg)
        self.labels = (
            labels
            if labels is not None
            else pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
        )
        self.participants = (
            participants
            if participants is not None
            else pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
        )
        self.ids = analysis_ids(self.labels, self.participants, analysis)
        if not self.ids:
            raise ValueError(
                f"no participants in the {analysis!r} analysis set; mark "
                "qc_passed first (python -m src.data.crop_manifest "
                "--apply-qc-status)."
            )

    def splits(self) -> Iterator[FoldSplit]:
        """Yield one leakage-checked FoldSplit per outer fold."""
        for fold in range(self.cfg.cv.n_splits):
            train, test = fold_ids(self.folds, fold, self.ids)
            inner_train, inner_val = inner_split(
                train,
                self.labels,
                self.cfg.cv.inner_val_fraction,
                self.cfg.training.seed + fold,
            )
            assert_no_leakage(inner_train, inner_val, test)
            yield FoldSplit(
                analysis=self.analysis,
                fold=fold,
                inner_train_ids=tuple(inner_train),
                inner_val_ids=tuple(inner_val),
                test_ids=tuple(test),
            )

    def run(self, fit_predict: Callable[[FoldSplit], pd.DataFrame]) -> pd.DataFrame:
        """Call fit_predict on every fold; return all folds' predictions.

        fit_predict(split) must return a DataFrame with a participant_id
        column whose participants are exactly split.test_ids. The returned
        frame gains analysis and fold columns.
        """
        frames = []
        for split in self.splits():
            preds = fit_predict(split)
            got = set(preds["participant_id"].astype(str))
            expected = set(split.test_ids)
            if got != expected:
                raise ValueError(
                    f"fold {split.fold}: predictions must cover exactly the test "
                    f"participants (missing {sorted(expected - got)[:5]}, "
                    f"extra {sorted(got - expected)[:5]})"
                )
            frames.append(preds.assign(analysis=split.analysis, fold=split.fold))
        return pd.concat(frames, ignore_index=True)
