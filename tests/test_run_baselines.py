"""Tests for src.training.run_baselines on synthetic roots.

Each synthetic run happens once per session (module fixtures) and the tests
read its outputs; the null-root run is timed for the < 1 minute criterion.
"""

import json
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.training import run_baselines
from src.training.run_baselines import PRED_COLUMNS, fit_lr
from src.training.splits import analysis_ids, fold_ids, load_folds
from src.utils.config import config
from src.utils.protocol_guard import ProtocolNotFrozenError


def _point_at(mp, root):
    mp.setattr(config.paths, "data_root", root)
    mp.setattr(config.paths, "output_root", root)
    mp.setattr(config.labeling, "output_csv", None)


@pytest.fixture(scope="module")
def null_run(synthetic_root):
    with pytest.MonkeyPatch.context() as mp:
        _point_at(mp, synthetic_root)
        start = time.perf_counter()
        pred_path = run_baselines.run("primary", predict_middle=True)
        elapsed = time.perf_counter() - start
    return pred_path, elapsed


@pytest.fixture(scope="module")
def signal_run(synthetic_root_signal):
    with pytest.MonkeyPatch.context() as mp:
        _point_at(mp, synthetic_root_signal)
        return run_baselines.run("primary")


def _summary(pred_path):
    return json.loads((pred_path.parent / "summary.json").read_text())


def _pooled(summary, key):
    return summary["models"][key]["pooled_test"]["macro_f1"]


# ── acceptance ────────────────────────────────────────────────────────────────


def test_runs_on_synthetic_fixture_under_a_minute(null_run):
    _, elapsed = null_run
    assert elapsed < 60, f"run took {elapsed:.1f}s"


def test_appends_point_metrics_to_run_log(null_run, synthetic_root):
    pred_path, _ = null_run
    preds = pd.read_csv(pred_path, dtype={"participant_id": str})
    groups = preds.groupby(["model", "feature_set"], sort=False).ngroups
    log = pd.read_csv(synthetic_root / "results" / "RUN_LOG.csv", dtype=str)
    command = (
        "python -m src.training.run_baselines --analysis primary "
        "--predict-middle --stage smoke"
    )
    rows = log[log["command"] == command].tail(groups)
    assert len(rows) == groups
    assert set(zip(rows["model"], rows["feature_set"])) == set(
        zip(preds["model"], preds["feature_set"])
    )
    assert (rows["stage"] == "smoke").all() and (rows["subset"] == "full").all()
    assert rows["macro_f1_ci_low"].isna().all()
    assert rows["macro_f1_ci_high"].isna().all()
    tested = preds.loc[~preds["in_middle_band"], "participant_id"].nunique()
    assert (rows["n_participants"].astype(int) == tested).all()
    majority = rows[rows["model"] == "majority"].iloc[0]
    assert float(majority["accuracy"]) == pytest.approx(
        float(majority["majority_baseline"])
    )


def test_near_chance_on_null_root(null_run):
    summary = _summary(null_run[0])
    for key, entry in summary["models"].items():
        f1 = entry["pooled_test"]["macro_f1"]
        assert f1 < 0.65, (key, f1)
    assert _pooled(summary, "lr_handcrafted/all") > 0.30


def test_clearly_above_chance_with_signal(signal_run):
    summary = _summary(signal_run)
    for key in ("lr_handcrafted/all", "lr_embeddings/concat"):
        assert _pooled(summary, key) > 0.80, key
    assert _pooled(summary, "majority/all") < 0.5  # majority macro-F1 ~ 0.36


def test_no_scaler_or_pca_ever_sees_a_validation_id(
    synthetic_root, use_root, monkeypatch
):
    """Fails if any StandardScaler or PCA is fit with a test or middle-band ID.

    fit_lr is the only place scalers/PCA are fit: record the participant IDs
    each fit_lr call receives, require every StandardScaler row to be one of
    that call's rows, and fail on any scaler/PCA fit outside fit_lr.
    """
    use_root(synthetic_root)
    calls, active = [], []
    real_fit_lr = run_baselines.fit_lr

    def spy_fit_lr(X_train, y_train, cfg, **kwargs):
        rows = {r.tobytes() for r in np.asarray(X_train, dtype=float)}
        calls.append(set(map(str, X_train.index)))
        active.append(rows)
        try:
            return real_fit_lr(X_train, y_train, cfg, **kwargs)
        finally:
            active.pop()

    def spy(cls, name, check_rows):
        original = getattr(cls, name)

        def wrapper(self, X, *args, **kwargs):
            assert active, f"{cls.__name__}.{name} called outside fit_lr"
            if check_rows:
                seen = {r.tobytes() for r in np.asarray(X, dtype=float)}
                assert seen <= active[-1], f"{cls.__name__} fit on rows outside fit_lr"
            return original(self, X, *args, **kwargs)

        monkeypatch.setattr(cls, name, wrapper)

    monkeypatch.setattr(run_baselines, "fit_lr", spy_fit_lr)
    spy(StandardScaler, "fit", check_rows=True)
    spy(PCA, "fit", check_rows=False)
    spy(PCA, "fit_transform", check_rows=False)

    run_baselines.run("primary", predict_middle=True)

    labels = pd.read_csv(synthetic_root / "metadata" / "labels.csv", dtype=str)
    parts = pd.read_csv(synthetic_root / "metadata" / "participants.csv", dtype=str)
    folds = load_folds(config)
    ids = analysis_ids(labels, parts, "primary")
    middle = set(labels.loc[labels["in_middle_band"] == "True", "participant_id"])
    trains = [set(fold_ids(folds, k, ids)[0]) for k in range(config.cv.n_splits)]
    tests = [set(fold_ids(folds, k, ids)[1]) for k in range(config.cv.n_splits)]

    n_lr_sets = 9  # 4 handcrafted + 4 embeddings + 1 handwriting
    assert len(calls) == n_lr_sets * config.cv.n_splits
    for i, seen in enumerate(calls):
        fold = i // n_lr_sets
        assert seen == trains[fold]
        assert not seen & tests[fold] and not seen & middle


# ── outputs ───────────────────────────────────────────────────────────────────


def test_outputs_schema_and_files(null_run, synthetic_root):
    pred_path = null_run[0]
    preds = pd.read_csv(pred_path, dtype={"participant_id": str})
    assert tuple(preds.columns) == PRED_COLUMNS
    assert pred_path.parent.name == "primary"

    labels = pd.read_csv(synthetic_root / "metadata" / "labels.csv", dtype=str)
    n_primary = (labels["in_primary_analysis"] == "True").sum()
    n_middle = (labels["in_middle_band"] == "True").sum()
    one = preds[(preds["model"] == "lr_handcrafted") & (preds["feature_set"] == "all")]
    assert len(one) == n_primary + n_middle
    assert one["participant_id"].is_unique
    assert one["in_middle_band"].sum() == n_middle

    combos = set(zip(preds["model"], preds["feature_set"]))
    assert combos == {
        ("majority", "all"),
        *(("lr_handcrafted", f) for f in ("all", "drawing", "word", "cursive")),
        *(("lr_embeddings", f) for f in ("concat", "drawing", "word", "cursive")),
        ("lr_handwriting", "concat"),
    }

    summary = _summary(pred_path)
    entry = summary["models"]["lr_handcrafted/all"]
    assert len(entry["folds"]) == config.cv.n_splits
    assert all(f["C"] in config.baselines.C_grid for f in entry["folds"])
    assert entry["middle_band"]["n"] == n_middle
    assert summary["models"]["lr_handwriting/concat"]["exploratory"] is True
    emb = summary["models"]["lr_embeddings/concat"]["folds"][0]
    assert 0 < emb["n_pca"] <= config.baselines.pca_components

    coefs = pd.read_csv(pred_path.parent / "lr_coefficients.csv")
    assert set(coefs["model"]) == {"lr_handcrafted"}
    assert set(coefs["fold"]) == set(range(config.cv.n_splits))


def test_pilot_subset_restricts_to_codes(synthetic_root, use_root):
    use_root(synthetic_root)
    pred_path = run_baselines.run("primary", subset="pilot")
    preds = pd.read_csv(pred_path, dtype={"participant_id": str})
    lo, hi = config.cv.pilot_id_range
    assert preds["participant_id"].astype(int).between(lo, hi).all()
    assert pred_path.parent.name == "primary_pilot"


# ── guards and helpers ────────────────────────────────────────────────────────


def test_predict_middle_only_for_primary(synthetic_root, use_root):
    use_root(synthetic_root)
    with pytest.raises(ValueError, match="primary analysis only"):
        run_baselines.run("full", predict_middle=True)


def test_refuses_real_data_before_protocol_tag(tmp_path, use_root, monkeypatch):
    use_root(tmp_path)  # no synthetic marker
    monkeypatch.setattr(
        "src.utils.protocol_guard.protocol_frozen", lambda *a, **k: False
    )
    with pytest.raises(ProtocolNotFrozenError, match="protocol-frozen"):
        run_baselines.run("primary")
    monkeypatch.setattr("sys.argv", ["run_baselines"])
    assert run_baselines.main() == 1


def test_fit_lr_picks_c_from_grid_and_caps_pca():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 100)), index=[f"{i:03d}" for i in range(30)])
    y = np.array([0, 1] * 15)
    pipe = fit_lr(X, y, config, pca=True, seed=0)
    assert pipe["lr"].C in config.baselines.C_grid
    assert pipe["pca"].n_components_ < 30
    assert pipe["lr"].class_weight == "balanced"
    with pytest.raises(ValueError, match=">= 2"):
        fit_lr(X.iloc[:3], np.array([0, 0, 1]), config)
