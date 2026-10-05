"""Tests for src.models.hmm.HMMClassifier on sequences from two known HMMs."""

import logging
from dataclasses import replace

import numpy as np
import pytest
from hmmlearn import hmm

from src.models.hmm import HMMClassifier, UncalibratedWarning, left_right_params
from src.utils.config import config

N_FEATURES = 10
HAPPY, SAD = config.data.label_to_index["HAPPY"], config.data.label_to_index["SAD"]


def _generator(shift: float, seed: int) -> hmm.GaussianHMM:
    g = hmm.GaussianHMM(n_components=3, covariance_type="diag", random_state=seed)
    g.startprob_ = np.array([0.6, 0.3, 0.1])
    g.transmat_ = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
    base = np.arange(3)[:, None] * np.linspace(-1, 1, N_FEATURES)[None, :]
    g.means_ = base + shift
    g.covars_ = np.full((3, N_FEATURES), 1.0)
    return g


def _sample(n_per_class: int, seed: int, shift: float = 0.6):
    rng = np.random.default_rng(seed)
    gens = {HAPPY: _generator(0.0, seed), SAD: _generator(shift, seed + 1)}
    seqs, labels = [], []
    for label, gen in gens.items():
        for _ in range(n_per_class):
            length = int(rng.integers(20, 35))
            x, _ = gen.sample(length, random_state=int(rng.integers(1 << 31)))
            seqs.append(x)
            labels.append(label)
    order = rng.permutation(len(seqs))
    return [seqs[i] for i in order], np.array(labels)[order]


@pytest.fixture(scope="module")
def data():
    return {
        "train": _sample(60, seed=1),
        "val": _sample(40, seed=2),
        "test": _sample(100, seed=3),
    }


@pytest.fixture(scope="module")
def fitted(data):
    clf = HMMClassifier(seed=7).fit(*data["train"])
    clf.fit_calibrator(clf.decision_scores(data["val"][0]), data["val"][1])
    return clf


def test_accuracy_above_90_percent(fitted, data):
    seqs, labels = data["test"]
    acc = np.mean(fitted.predict(seqs) == labels)
    assert acc > 0.90, acc
    proba = fitted.predict_proba(seqs)
    assert proba.shape == (len(seqs), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    sad_col = list(fitted.classes_).index(SAD)
    assert np.mean((proba[:, sad_col] >= 0.5) == (labels == SAD)) > 0.90


def test_scaler_and_pca_fit_on_training_frames_only(fitted, data):
    seqs, _ = data["train"]
    n_frames = sum(len(s) for s in seqs)
    assert fitted.scaler.n_samples_seen_ == n_frames
    assert fitted.pca.n_samples_ == n_frames
    assert fitted.pca.n_components_ == min(config.hmm.pca_components, N_FEATURES)


def test_decision_score_is_length_normalised_ll_difference_plus_prior(fitted, data):
    x = data["test"][0][0]
    z = fitted.pca.transform(fitted.scaler.transform(x))
    expected = (
        fitted.models["SAD"].score(z) / len(z)
        - fitted.models["HAPPY"].score(z) / len(z)
        + fitted.log_prior_ratio
    )
    assert fitted.decision_scores([x])[0] == pytest.approx(expected)
    assert fitted.log_prior_ratio == pytest.approx(0.0)  # balanced training


def test_log_prior_ratio_from_training_labels(data):
    seqs, labels = data["train"]
    keep = [i for i, lab in enumerate(labels) if lab == HAPPY or i % 2 == 0]
    clf = HMMClassifier(replace(config.hmm, n_restarts=1), seed=0)
    clf.fit([seqs[i] for i in keep], labels[keep])
    n_sad = np.sum(labels[keep] == SAD)
    n_happy = np.sum(labels[keep] == HAPPY)
    assert clf.log_prior_ratio == pytest.approx(np.log(n_sad / n_happy))


def test_save_load_round_trip_identical_probabilities(fitted, data, tmp_path):
    path = tmp_path / "hmm.joblib"
    fitted.save(path)
    loaded = HMMClassifier().load(path)
    seqs = data["test"][0]
    np.testing.assert_array_equal(
        loaded.predict_proba(seqs), fitted.predict_proba(seqs)
    )
    assert loaded.calibrator is not None and loaded.pca is not None


def test_restarts_reproducible_with_seed(data):
    seqs, labels = data["train"]
    cfg = replace(config.hmm, n_restarts=3)
    a = HMMClassifier(cfg, seed=11).fit(seqs, labels)
    b = HMMClassifier(cfg, seed=11).fit(seqs, labels)
    test = data["test"][0][:20]
    np.testing.assert_array_equal(a.decision_scores(test), b.decision_scores(test))
    assert a.restart_log == b.restart_log
    seeds = [r["seed"] for r in a.restart_log["SAD"]["runs"]]
    assert seeds == [11, 12, 13]


def test_best_restart_is_kept(fitted):
    for name in ("HAPPY", "SAD"):
        log = fitted.restart_log[name]
        lls = [r["log_likelihood"] for r in log["runs"]]
        assert log["winner"] == int(np.argmax(lls))
        assert len(lls) == config.hmm.n_restarts


def test_convergence_logged_with_winning_restart(data, caplog):
    seqs, labels = data["train"]
    cfg = replace(config.hmm, n_restarts=2, n_iter=1)  # cannot converge in 1 step
    with caplog.at_level(logging.INFO, logger="src.models.hmm"):
        HMMClassifier(cfg, seed=0).fit(seqs, labels)
    winning = [r for r in caplog.records if "winning restart" in r.getMessage()]
    assert len(winning) == 2  # one per class
    assert all(r.levelno == logging.WARNING for r in winning)
    assert all("converged=False" in r.getMessage() for r in winning)


def test_normal_fit_logs_converged_winner_at_info(data, caplog):
    seqs, labels = data["train"]
    with caplog.at_level(logging.INFO, logger="src.models.hmm"):
        clf = HMMClassifier(replace(config.hmm, n_restarts=2), seed=0).fit(seqs, labels)
    winning = [r for r in caplog.records if "winning restart" in r.getMessage()]
    assert len(winning) == 2
    assert all(r.levelno == logging.INFO for r in winning)
    assert all("converged=True" in r.getMessage() for r in winning)
    for log in clf.restart_log.values():
        assert all(r["n_iter"] < config.hmm.n_iter for r in log["runs"])


def test_uncalibrated_predict_proba_warns(data):
    seqs, labels = data["train"]
    clf = HMMClassifier(replace(config.hmm, n_restarts=1), seed=0).fit(seqs, labels)
    with pytest.warns(UncalibratedWarning):
        proba = clf.predict_proba(data["test"][0][:5])
    assert proba.shape == (5, 2)


def test_left_right_topology_keeps_zero_transitions(data):
    seqs, labels = data["train"]
    cfg = replace(config.hmm, topology="left_right", n_restarts=2)
    clf = HMMClassifier(cfg, seed=0).fit(seqs, labels)
    _, banded = left_right_params(cfg.n_states)
    for model in clf.models.values():
        assert np.all(model.transmat_[banded == 0] == 0)
        assert model.startprob_[0] == 1.0
        assert np.allclose(model.transmat_.sum(axis=1), 1.0)


def test_input_validation():
    with pytest.raises(ValueError, match="topology"):
        HMMClassifier(replace(config.hmm, topology="cyclic"))
    clf = HMMClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.decision_scores([np.zeros((5, 3))])
    seqs, labels = _sample(5, seed=4)
    with pytest.raises(ValueError, match="both classes"):
        clf.fit(seqs[:3], np.full(3, HAPPY))
    with pytest.raises(ValueError, match="both classes"):
        HMMClassifier(replace(config.hmm, n_restarts=1)).fit(
            seqs, labels
        ).fit_calibrator([0.1, 0.2], [HAPPY, HAPPY])
