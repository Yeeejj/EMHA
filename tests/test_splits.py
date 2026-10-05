"""Tests for src.training.splits: participant-level folds and leakage checks."""

import numpy as np
import pandas as pd
import pytest

from src.training import splits
from src.training.splits import (
    FOLD_COLUMNS,
    LeakageError,
    assert_no_leakage,
    fold_ids,
    make_folds,
    validate_folds,
)
from src.utils.config import config

N_FOLDS = config.training.n_folds


def _labels(n_happy_primary=60, n_sad_primary=50, n_happy_mid=16, n_sad_mid=14):
    rows = []
    groups = (
        ("HAPPY", True, n_happy_primary),
        ("SAD", True, n_sad_primary),
        ("HAPPY", False, n_happy_mid),
        ("SAD", False, n_sad_mid),
    )
    for label, primary, n in groups:
        for _ in range(n):
            rows.append({"label": label, "in_primary_analysis": primary})
    df = pd.DataFrame(rows)
    df.insert(0, "participant_id", [f"{i + 1:03d}" for i in range(len(df))])
    return df


def test_assert_no_leakage_passes_on_disjoint_sets():
    assert_no_leakage(["001", "002"], ["003"], ["004"])
    assert_no_leakage([1, 2], ["3"])


@pytest.mark.parametrize(
    "train,test,val,pair",
    [
        (["001", "002"], ["002"], None, "train and test"),
        (["001"], ["002"], ["001"], "train and val"),
        (["001"], ["002"], ["002"], "test and val"),
    ],
)
def test_assert_no_leakage_names_the_overlapping_sets(train, test, val, pair):
    with pytest.raises(LeakageError, match=pair):
        assert_no_leakage(train, test, val)


def test_ids_compared_as_strings():
    with pytest.raises(LeakageError):
        assert_no_leakage([1], ["1"])


def test_leakage_error_is_an_assertion_error():
    assert issubclass(LeakageError, AssertionError)


def test_folds_shape_columns_and_partition():
    labels = _labels()
    folds = make_folds(labels, config)
    assert tuple(folds.columns) == FOLD_COLUMNS
    assert len(folds) == len(labels) * N_FOLDS
    test = folds[folds["role"] == "test"]
    assert test["participant_id"].is_unique
    assert set(test["participant_id"]) == set(labels["participant_id"])


def test_every_fold_is_leak_free_with_expected_role_sizes():
    labels = _labels()
    folds = make_folds(labels, config)
    n = len(labels)
    for fold in range(N_FOLDS):
        ids = fold_ids(folds, fold)
        assert len(ids["test"]) in (n // N_FOLDS, n // N_FOLDS + 1)
        pool = len(ids["train"]) + len(ids["inner_val"])
        assert pool + len(ids["test"]) == n
        frac = len(ids["inner_val"]) / pool
        assert frac == pytest.approx(config.splits.inner_val_fraction, abs=0.02)


def test_test_folds_stratified_on_label_and_primary_set():
    labels = _labels()
    folds = make_folds(labels, config)
    test = folds[folds["role"] == "test"]
    counts = pd.crosstab(
        test["outer_fold"], [test["label"], test["in_primary_analysis"]]
    )
    sizes = labels.groupby(["label", "in_primary_analysis"]).size()
    for stratum, total in sizes.items():
        per_fold = counts[stratum]
        assert per_fold.max() - per_fold.min() <= 1
        assert per_fold.sum() == total


def test_inner_val_stratified_too():
    folds = make_folds(_labels(), config)
    for fold in range(N_FOLDS):
        part = folds[(folds["outer_fold"] == fold) & (folds["role"] == "inner_val")]
        assert set(zip(part["label"], part["in_primary_analysis"])) == {
            ("HAPPY", True),
            ("SAD", True),
            ("HAPPY", False),
            ("SAD", False),
        }


def test_primary_only_restricts_every_role():
    labels = _labels()
    folds = make_folds(labels, config)
    primary = set(labels.loc[labels["in_primary_analysis"], "participant_id"])
    for fold in range(N_FOLDS):
        ids = fold_ids(folds, fold, primary_only=True)
        assert all(set(v) <= primary for v in ids.values())
        full = fold_ids(folds, fold)
        assert set(ids["test"]) == set(full["test"]) & primary


def test_deterministic_for_seed_and_changes_with_seed(monkeypatch):
    labels = _labels()
    a = make_folds(labels, config)
    b = make_folds(labels.sample(frac=1, random_state=1), config)  # row order
    pd.testing.assert_frame_equal(a, b)
    monkeypatch.setattr(config.training, "seed", config.training.seed + 1)
    c = make_folds(labels, config)
    assert not a.equals(c)


def test_validate_folds_catches_corruption():
    folds = make_folds(_labels(), config)
    pid = folds.loc[folds["role"] == "test", "participant_id"].iloc[0]
    # tested in a second fold
    bad = folds.copy()
    other = (bad["participant_id"] == pid) & (bad["role"] != "test")
    bad.loc[bad.index[other][0], "role"] = "test"
    with pytest.raises(ValueError, match="exactly one fold"):
        validate_folds(bad, N_FOLDS)
    # missing a fold row
    with pytest.raises(ValueError, match="one row per outer fold"):
        validate_folds(folds.drop(index=folds.index[0]), N_FOLDS)
    # duplicated into train and test of the same fold
    fold0 = folds[(folds["outer_fold"] == 0) & (folds["role"] == "test")].iloc[[0]]
    leaked = pd.concat([folds, fold0.assign(role="train")], ignore_index=True)
    with pytest.raises((LeakageError, ValueError)):
        validate_folds(leaked, N_FOLDS)


def test_rejects_duplicate_or_missing_labels():
    labels = _labels()
    with pytest.raises(ValueError, match="duplicate"):
        make_folds(pd.concat([labels, labels.iloc[[0]]]), config)
    with pytest.raises(ValueError, match="missing"):
        make_folds(labels.assign(label=labels["label"].where(labels.index > 0)), config)


def _setup_run(tmp_path, monkeypatch):
    meta = tmp_path / "DATASET" / "metadata"
    meta.mkdir(parents=True)
    _labels().to_csv(meta / "labels.csv", index=False)
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "DATASET")
    monkeypatch.setattr(config.labeling, "output_csv", None)
    return meta


def test_run_writes_then_is_idempotent(tmp_path, monkeypatch):
    meta = _setup_run(tmp_path, monkeypatch)
    out = splits.run()
    assert out == meta / "folds.csv"
    first = out.read_bytes()
    assert splits.run() == out
    assert out.read_bytes() == first
    loaded = splits.load_folds(config)
    assert loaded["participant_id"].str.len().eq(3).all()  # zero-padded IDs kept


def test_run_refuses_to_change_existing_folds(tmp_path, monkeypatch):
    _setup_run(tmp_path, monkeypatch)
    out = splits.run()
    before = out.read_bytes()
    monkeypatch.setattr(config.training, "seed", config.training.seed + 1)
    with pytest.raises(RuntimeError, match="Deviation"):
        splits.run()
    assert out.read_bytes() == before
    assert splits.main() == 1


def test_roles_cover_each_participant_once_per_fold():
    folds = make_folds(_labels(), config)
    per = folds.groupby(["participant_id", "outer_fold"]).size()
    assert (per == 1).all()
    assert np.array_equal(
        folds.groupby("participant_id")["outer_fold"].nunique().unique(), [N_FOLDS]
    )
