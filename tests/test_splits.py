"""Tests for src.training.splits: the fold assignment, inner splits, leakage."""

import pandas as pd
import pytest

from src.training import splits
from src.training.splits import (
    FOLD_COLUMNS,
    LeakageError,
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    inner_split,
    make_outer_folds,
    validate_folds,
)
from src.utils.config import config

N = config.cv.n_splits
SEED = config.training.seed
# Real group sizes from labels.csv (2026-10-05).
GROUPS = (
    ("HAPPY", False, 126),
    ("SAD", False, 102),
    ("HAPPY", True, 32),
    ("SAD", True, 25),
)


def _labels(groups=GROUPS):
    rows = [
        {"label": label, "in_middle_band": mid, "in_primary_analysis": not mid}
        for label, mid, n in groups
        for _ in range(n)
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "participant_id", [f"{i + 1:03d}" for i in range(len(df))])
    return df


def _participants(labels, failed=()):
    status = [
        "indexed" if p in failed else "qc_passed" for p in labels["participant_id"]
    ]
    return pd.DataFrame({"participant_id": labels["participant_id"], "status": status})


# ── assert_no_leakage ─────────────────────────────────────────────────────────


def test_assert_no_leakage_passes_on_disjoint_lists():
    assert_no_leakage(["001", "002"], ["003"], ["004"], [])
    assert_no_leakage()


@pytest.mark.parametrize(
    "lists,pair",
    [
        ((["001", "002"], ["002"]), "id list 0 and id list 1"),
        ((["001"], ["002"], ["001"]), "id list 0 and id list 2"),
        ((["001"], ["002"], ["002"]), "id list 1 and id list 2"),
    ],
)
def test_assert_no_leakage_raises_on_overlap(lists, pair):
    with pytest.raises(LeakageError, match=pair):
        assert_no_leakage(*lists)


def test_leakage_ids_compared_as_strings_and_is_assertion_error():
    with pytest.raises(AssertionError):
        assert_no_leakage([1], ["1"])


# ── make_outer_folds ──────────────────────────────────────────────────────────


def test_one_row_per_participant_no_participant_in_two_folds():
    labels = _labels()
    folds = make_outer_folds(labels, N, SEED)
    assert tuple(folds.columns) == FOLD_COLUMNS
    assert len(folds) == len(labels)
    assert folds["participant_id"].is_unique
    assert set(folds["outer_fold"]) == set(range(N))


@pytest.mark.parametrize("analysis", ["primary", "full"])
def test_per_fold_label_ratio_within_3_points(analysis):
    labels = _labels()
    folds = make_outer_folds(labels, N, SEED)
    ids = analysis_ids(labels, _participants(labels), analysis)
    sub = folds[folds["participant_id"].isin(ids)]
    overall = (sub["label"] == "SAD").mean()
    for fold in range(N):
        _, test = fold_ids(folds, fold, ids)
        ratio = (sub.set_index("participant_id").loc[test, "label"] == "SAD").mean()
        assert abs(ratio - overall) * 100 <= 3.0, (analysis, fold, ratio, overall)


def test_same_seed_identical_regardless_of_row_order():
    labels = _labels()
    a = make_outer_folds(labels, N, SEED)
    b = make_outer_folds(labels.sample(frac=1, random_state=3), N, SEED)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    c = make_outer_folds(labels, N, SEED + 1)
    assert not a["outer_fold"].equals(c["outer_fold"])


def test_rejects_bad_labels():
    labels = _labels()
    with pytest.raises(ValueError, match="duplicate"):
        make_outer_folds(pd.concat([labels, labels.iloc[[0]]]), N, SEED)
    with pytest.raises(ValueError, match="missing"):
        bad = labels.assign(label=labels["label"].where(labels.index > 0))
        make_outer_folds(bad, N, SEED)
    with pytest.raises(ValueError, match="lacks columns"):
        make_outer_folds(labels.drop(columns="in_middle_band"), N, SEED)


def test_validate_folds_catches_corruption():
    folds = make_outer_folds(_labels(), N, SEED)
    with pytest.raises(ValueError, match="more than one fold"):
        validate_folds(pd.concat([folds, folds.iloc[[0]].assign(outer_fold=1)]), N)
    with pytest.raises(ValueError, match="outer_fold"):
        validate_folds(folds.assign(outer_fold=folds["outer_fold"] + N), N)


# ── fold_ids / inner_split ────────────────────────────────────────────────────


def test_fold_ids_partition_the_analysis_set():
    labels = _labels()
    folds = make_outer_folds(labels, N, SEED)
    ids = analysis_ids(labels, _participants(labels), "primary")
    tested = []
    for fold in range(N):
        train, test = fold_ids(folds, fold, ids)
        assert set(train) | set(test) == set(ids)
        assert not set(train) & set(test)
        tested += test
    assert sorted(tested) == sorted(ids)


def test_fold_ids_rejects_unknown_ids():
    folds = make_outer_folds(_labels(), N, SEED)
    with pytest.raises(ValueError, match="not in folds.csv"):
        fold_ids(folds, 0, ["999"])


def test_inner_split_disjoint_from_outer_test_and_sized():
    labels = _labels()
    folds = make_outer_folds(labels, N, SEED)
    ids = analysis_ids(labels, _participants(labels), "full")
    for fold in range(N):
        train, test = fold_ids(folds, fold, ids)
        inner_train, inner_val = inner_split(
            train, labels, config.cv.inner_val_fraction, SEED + fold
        )
        assert_no_leakage(inner_train, inner_val, test)
        assert sorted(inner_train + inner_val) == sorted(train)
        frac = len(inner_val) / len(train)
        assert frac == pytest.approx(config.cv.inner_val_fraction, abs=0.01)


def test_inner_split_deterministic_and_stratified():
    labels = _labels()
    train = list(labels["participant_id"][::2])
    a = inner_split(train, labels, 0.15, 7)
    assert a == inner_split(list(reversed(train)), labels, 0.15, 7)
    val = labels.set_index("participant_id").loc[a[1]]
    assert val["label"].nunique() == 2
    assert val["in_middle_band"].nunique() == 2


def test_inner_split_falls_back_when_a_stratum_is_tiny():
    labels = _labels((("HAPPY", False, 20), ("SAD", False, 20), ("SAD", True, 1)))
    train_ids, val_ids = inner_split(list(labels["participant_id"]), labels, 0.2, 0)
    assert len(val_ids) == 9 and not set(train_ids) & set(val_ids)


# ── analysis_ids ──────────────────────────────────────────────────────────────


def test_analysis_ids_primary_and_full_require_qc_passed():
    labels = _labels()
    failed = {"001", "250"}  # 001 primary, 250 middle band
    parts = _participants(labels, failed=failed)
    full = analysis_ids(labels, parts, "full")
    primary = analysis_ids(labels, parts, "primary")
    assert len(full) == len(labels) - 2 and not failed & set(full)
    assert set(primary) <= set(full)
    assert len(primary) == 228 - 1
    mid = set(labels.loc[labels["in_middle_band"], "participant_id"])
    assert not mid & set(primary)
    with pytest.raises(ValueError, match="analysis must be"):
        analysis_ids(labels, parts, "secondary")


def test_analysis_ids_empty_when_nobody_qc_passed():
    labels = _labels()
    parts = _participants(labels, failed=set(labels["participant_id"]))
    assert analysis_ids(labels, parts, "full") == []


# ── run / CLI ─────────────────────────────────────────────────────────────────


def _setup(tmp_path, monkeypatch):
    meta = tmp_path / "DATASET" / "metadata"
    meta.mkdir(parents=True)
    _labels().to_csv(meta / "labels.csv", index=False)
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "DATASET")
    monkeypatch.setattr(config.labeling, "output_csv", None)
    return meta / "folds.csv"


def test_run_writes_prints_sha_and_is_idempotent(tmp_path, monkeypatch, capsys):
    out = _setup(tmp_path, monkeypatch)
    assert splits.run() == out
    sha = splits.sha256_of(out)
    assert f"SHA-256      : {sha}" in capsys.readouterr().out
    splits.run()
    assert "folds.csv unchanged" in capsys.readouterr().out
    assert splits.sha256_of(out) == sha
    assert splits.load_folds(config)["participant_id"].str.len().eq(3).all()


def test_run_refuses_overwrite_without_force(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch)
    splits.run()
    before = out.read_bytes()
    monkeypatch.setattr(config.training, "seed", SEED + 1)
    with pytest.raises(RuntimeError, match="--force"):
        splits.run()
    assert out.read_bytes() == before
    monkeypatch.setattr("sys.argv", ["splits"])
    assert splits.main() == 1
    assert out.read_bytes() == before
    monkeypatch.setattr("sys.argv", ["splits", "--force"])
    assert splits.main() == 0
    assert out.read_bytes() != before


def test_flags_read_as_text_parse_strictly():
    labels = _labels().astype(str)  # "True"/"False" strings, as from dtype=str
    parts = _participants(_labels())
    assert analysis_ids(labels, parts, "primary") == analysis_ids(
        _labels(), parts, "primary"
    )
    bad = labels.assign(in_middle_band="maybe")
    with pytest.raises(ValueError, match="non-boolean"):
        make_outer_folds(bad, N, SEED)
