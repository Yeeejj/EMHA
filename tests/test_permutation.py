"""Tests for src.analysis.permutation on synthetic roots.

LR tests reuse the session synthetic roots (null and signal) after one
run_baselines run each. The CNN-path test builds a tiny image root (tiny
canvases, 1 epoch, no pretrained weights, a 2-setting HMM grid) and runs one
full-retrain draw end to end.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import permutation
from src.analysis.permutation import (
    lr_null,
    lr_predictions,
    p_value,
    permutation_seed,
    permutation_test,
    permuted_train_labels,
)
from src.training import run_baselines, run_cnn, run_ensemble, run_hybrid
from src.training.evaluator import metrics
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

N_SMALL = 24  # smallest attainable p = 1/25 = 0.04 < alpha


def _point_at(mp, root):
    mp.setattr(config.paths, "data_root", root)
    mp.setattr(config.paths, "output_root", root)
    mp.setattr(config.labeling, "output_csv", None)


@pytest.fixture(scope="module")
def null_root(synthetic_root):
    with pytest.MonkeyPatch.context() as mp:
        _point_at(mp, synthetic_root)
        run_baselines.run("primary")
    return synthetic_root


@pytest.fixture(scope="module")
def signal_root(synthetic_root_signal):
    with pytest.MonkeyPatch.context() as mp:
        _point_at(mp, synthetic_root_signal)
        run_baselines.run("primary")
    return synthetic_root_signal


# ── formula and shuffling ─────────────────────────────────────────────────────


def test_p_value_formula():
    assert p_value([0.1, 0.2, 0.3], 0.25) == pytest.approx(2 / 4)
    assert p_value([0.1, 0.2, 0.3], 0.9) == pytest.approx(1 / 4)
    assert p_value([0.5, 0.5], 0.5) == pytest.approx(3 / 3)  # ties count


def test_shuffle_stays_inside_outer_train(null_root, monkeypatch):
    _point_at(monkeypatch, null_root)
    ctx = permutation._context("primary")
    shuffled = permuted_train_labels(ctx, perm=0, seed=42)
    true = [ctx.true_index(train) for train, _ in ctx.splits]
    for s, t in zip(shuffled, true):
        assert sorted(s) == sorted(t) and len(s) == len(t)  # same labels, permuted
    assert any((s != t).any() for s, t in zip(shuffled, true))
    assert permutation_seed(42, 0, 1) != permutation_seed(42, 1, 0)


def test_unshuffled_lr_reproduces_the_observed_run(null_root, monkeypatch):
    _point_at(monkeypatch, null_root)
    ctx = permutation._context("primary")
    mine = lr_predictions("lr_handcrafted", "all", ctx, None)
    obs = pd.read_csv(
        null_root / "results" / "baselines" / "primary" / "predictions.csv",
        dtype={"participant_id": str},
    )
    obs = obs[
        (obs["model"] == "lr_handcrafted")
        & (obs["feature_set"] == "all")
        & ~obs["in_middle_band"]
    ]
    merged = mine.merge(obs, on=["participant_id", "fold"], suffixes=("", "_obs"))
    assert len(merged) == len(obs) == len(mine)
    np.testing.assert_allclose(merged["prob_sad"], merged["prob_sad_obs"], atol=1e-9)
    assert (merged["label"] == merged["label_obs"]).all()


# ── acceptance ────────────────────────────────────────────────────────────────


def test_not_significant_on_null_root(null_root, monkeypatch):
    _point_at(monkeypatch, null_root)
    result = permutation_test("lr_handcrafted", "primary", n=N_SMALL, n_jobs=2)
    assert result["p_value"] > config.permutation.alpha
    assert not result["significant"]


def test_significant_on_signal_root(signal_root, monkeypatch):
    _point_at(monkeypatch, signal_root)
    result = permutation_test("lr_handcrafted", "primary", n=N_SMALL, n_jobs=2)
    assert result["p_value"] == pytest.approx(1 / (N_SMALL + 1))
    assert result["significant"]
    out = signal_root / "results" / "final"
    null = np.load(result["null_path"])
    assert null.shape == (N_SMALL,) and result["observed"] > null.max()
    assert (out / "permutation_lr_handcrafted_primary.png").is_file()
    saved = json.loads((out / "permutation_lr_handcrafted_primary.json").read_text())
    assert saved["p_value"] == result["p_value"]
    text = (out / "model_comparison.txt").read_text(encoding="utf-8")
    assert f"lr_handcrafted/all: observed macro-F1 {result['observed']:.3f}" in text


def test_seeded_nulls_are_reproducible(signal_root, monkeypatch):
    _point_at(monkeypatch, signal_root)
    args = ("lr_handcrafted", "all", "primary", 4)
    a = lr_null(*args, n_jobs=1, seed=7)
    b = lr_null(*args, n_jobs=2, seed=7)  # worker count never changes a draw
    c = lr_null(*args, n_jobs=1, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_refuses_untested_models_and_cnn_on_full(null_root, monkeypatch, capsys):
    _point_at(monkeypatch, null_root)
    with pytest.raises(ValueError, match="no permutation test"):
        permutation_test("lr_handcrafted_word", n=2)
    with pytest.raises(ValueError, match="primary"):
        permutation_test("ensemble", "full", n=2)
    monkeypatch.setattr("sys.argv", ["permutation", "--model", "majority"])
    with pytest.raises(SystemExit):
        permutation.main()


# ── CNN path: one full-retrain draw ───────────────────────────────────────────

SMALL_CANVAS = {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)}


@pytest.fixture(scope="module")
def cnn_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
        mp.setattr(config.training, "epochs", 1)
        mp.setattr(config.data, "num_workers", 0)
        mp.setattr(config.cnn, "use_pretrained", False)
        mp.setattr(config.hmm, "n_restarts", 1)
        mp.setattr(config.hmm, "n_iter", 20)
        mp.setattr(config.hybrid, "n_states_grid", (2,))
        mp.setattr(config.hybrid, "pca_grid", (4,))
        mp.setattr(config.hybrid, "topology_grid", ("ergodic",))
        mp.setattr(config.hybrid, "selection_restarts", 1)
        root = make_synthetic_root(
            tmp_path_factory.mktemp("perm_cnn") / "root", n_participants=40, images=True
        )
        _point_at(mp, root)
        # observed runs for the ensemble and CNN-HMM need word + cursive only;
        # the permutation draw itself retrains all three families
        for family in config.hybrid.families:
            run_cnn.run(task_family=family, device="cpu")
        run_hybrid.run(device="cpu")
        run_baselines.run("primary")
        run_ensemble.run("primary")
        calls = []
        real = permutation.cnn_null_run

        def counted(*a, **k):
            calls.append(a[1])
            return real(*a, **k)

        mp.setattr(permutation, "cnn_null_run", counted)
        first = permutation_test("ensemble", "primary", n=1, device="cpu")
        second = permutation_test("cnn_hmm_fused", "primary", n=1, device="cpu")
        yield root, first, second, calls


def test_cnn_draw_runs_once_and_feeds_every_cnn_model(cnn_root):
    root, first, second, calls = cnn_root
    assert calls == [0]  # second test resumed from the cached draw
    run = root / "results" / "final" / "permutation_runs" / "primary" / "seed_42"
    frame = pd.read_csv(run / "perm_0000.csv", dtype={"participant_id": str})
    models = set(zip(frame["model"], frame["feature_set"]))
    assert models >= set(config.permutation.cnn_models)
    assert models >= set(config.permutation.lr_models)
    for (model, fs), part in frame.groupby(["model", "feature_set"]):
        assert part["participant_id"].is_unique and not part["in_middle_band"].any()
    hmm = frame[frame["model"] == "cnn_hmm_fused"]
    assert first["n"] == second["n"] == 1
    assert np.load(second["null_path"])[0] == pytest.approx(metrics(hmm)["macro_f1"])
    assert first["p_value"] in (0.5, 1.0)


def test_cnn_draw_scores_test_rows_on_true_labels(cnn_root):
    root, *_ = cnn_root
    labels = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    label_of = labels.set_index("participant_id")["label"]
    run = root / "results" / "final" / "permutation_runs" / "primary" / "seed_42"
    frame = pd.read_csv(run / "perm_0000.csv", dtype={"participant_id": str})
    assert (frame["label"].to_numpy() == label_of.loc[frame["participant_id"]]).all()
    assert not [d for d in (root / "models").iterdir() if d.name.startswith("tmp")]


def test_cnn_draw_cache_refuses_changed_settings(cnn_root, monkeypatch):
    root, *_ = cnn_root
    _point_at(monkeypatch, root)
    monkeypatch.setattr(config.training, "epochs", 2)
    with pytest.raises(RuntimeError, match="different settings"):
        permutation.cnn_null("ensemble", "all", "primary", 1, 42, "cpu")


def test_shuffled_cnn_matches_the_lr_shuffle(cnn_root, monkeypatch):
    """The LR path shuffles with permutation_seed(seed, perm, fold), the seed
    the CNN path passes to run_cnn._fit_fold / run_hybrid._fold."""
    root, *_ = cnn_root
    _point_at(monkeypatch, root)
    ctx = permutation._context("primary")
    seen = {}

    def spy(labels, ids, seed):
        out = real(labels, ids, seed)
        seen[seed] = out.set_index("participant_id").loc[list(ids), "label"].tolist()
        return out

    real = run_cnn.shuffled_labels
    monkeypatch.setattr(run_cnn, "shuffled_labels", spy)
    lr_y = permuted_train_labels(ctx, perm=3, seed=42)
    idx_to_label = {v: k for k, v in config.data.label_to_index.items()}
    for fold, (train, _) in enumerate(ctx.splits):
        s = permutation_seed(42, 3, fold)
        assert [idx_to_label[i] for i in lr_y[fold]] == seen[s]
