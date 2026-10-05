"""Tests for the participant-level CrossValidator runner."""

import pandas as pd
import pytest

from src.training.cross_validate import CrossValidator, FoldSplit
from src.training.splits import assert_no_leakage, make_outer_folds
from src.utils.config import config

N = config.cv.n_splits


def _labels():
    groups = (
        ("HAPPY", False, 60),
        ("SAD", False, 50),
        ("HAPPY", True, 15),
        ("SAD", True, 12),
    )
    rows = [
        {"label": lab, "in_middle_band": mid, "in_primary_analysis": not mid}
        for lab, mid, n in groups
        for _ in range(n)
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "participant_id", [f"{i + 1:03d}" for i in range(len(df))])
    return df


def _cv(analysis="primary", failed=()):
    labels = _labels()
    parts = pd.DataFrame(
        {
            "participant_id": labels["participant_id"],
            "status": [
                "indexed" if p in failed else "qc_passed"
                for p in labels["participant_id"]
            ],
        }
    )
    folds = make_outer_folds(labels, N, config.training.seed)
    return CrossValidator(analysis, folds=folds, labels=labels, participants=parts)


def test_splits_cover_analysis_set_once_and_never_leak():
    cv = _cv("primary", failed={"001"})
    seen = []
    for split in cv.splits():
        assert isinstance(split, FoldSplit)
        assert_no_leakage(split.inner_train_ids, split.inner_val_ids, split.test_ids)
        assert set(split.inner_train_ids + split.inner_val_ids + split.test_ids) == set(
            cv.ids
        )
        seen += split.test_ids
    assert sorted(seen) == cv.ids
    assert "001" not in cv.ids


def test_splits_are_deterministic():
    a = [s for s in _cv().splits()]
    b = [s for s in _cv().splits()]
    assert a == b


def test_run_collects_predictions_with_fold_and_analysis():
    cv = _cv("full")

    def fit_predict(split):
        return pd.DataFrame({"participant_id": list(split.test_ids), "prob_sad": 0.5})

    preds = cv.run(fit_predict)
    assert len(preds) == len(cv.ids)
    assert set(preds["analysis"]) == {"full"}
    assert sorted(preds["fold"].unique()) == list(range(N))


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_run_rejects_predictions_outside_the_test_fold(mode):
    cv = _cv()

    def fit_predict(split):
        ids = list(split.test_ids)
        ids = ids[1:] if mode == "missing" else ids + [split.inner_train_ids[0]]
        return pd.DataFrame({"participant_id": ids})

    with pytest.raises(ValueError, match="exactly the test"):
        cv.run(fit_predict)


def test_empty_analysis_set_raises_clearly():
    labels = _labels()
    with pytest.raises(ValueError, match="qc_passed"):
        _cv(failed=set(labels["participant_id"]))
