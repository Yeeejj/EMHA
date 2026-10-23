"""Parallel CNN-HMM fitting gives results identical to the serial code.

EVALUATION_PROTOCOL.md section 9, Deviation 2: the candidate fits of
run_hybrid.select_hmm and the restarts of HMMClassifier run in
RunnerConfig.hmm_n_jobs joblib processes. Every comparison here is exact
(==, np.array_equal; max absolute difference 0.0), never approximate:
serial (n_jobs = 1) against 4 processes on the same inputs.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from hmmlearn import hmm
from threadpoolctl import threadpool_limits

from src.analysis import permutation
from src.data.dataloader import load_tables
from src.training import run_cnn, run_hybrid
from src.training.run_hybrid import SequenceSet, fit_final, select_hmm
from src.training.splits import analysis_ids, load_folds
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

N_FEATURES = 10
PARALLEL = 4
HAPPY, SAD = config.data.label_to_index["HAPPY"], config.data.label_to_index["SAD"]
SMALL_CANVAS = {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)}


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _generator(shift: float, seed: int) -> hmm.GaussianHMM:
    g = hmm.GaussianHMM(n_components=3, covariance_type="diag", random_state=seed)
    g.startprob_ = np.array([0.6, 0.3, 0.1])
    g.transmat_ = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
    g.means_ = np.arange(3)[:, None] * np.linspace(-1, 1, N_FEATURES)[None, :] + shift
    g.covars_ = np.full((3, N_FEATURES), 1.0)
    return g


def _sequence_set(n_per_class: int, seed: int, first_id: int) -> SequenceSet:
    rng = np.random.default_rng(seed)
    gens = {HAPPY: _generator(0.0, seed), SAD: _generator(0.25, seed + 1)}
    seqs, labels = [], []
    for label, gen in gens.items():
        for _ in range(n_per_class):
            x, _ = gen.sample(
                int(rng.integers(12, 25)), random_state=int(rng.integers(1 << 31))
            )
            seqs.append(x)
            labels.append(label)
    order = rng.permutation(len(seqs))
    labels = np.array(labels)[order]
    names = {HAPPY: "HAPPY", SAD: "SAD"}
    meta = pd.DataFrame(
        {
            "participant_id": [f"{first_id + i:03d}" for i in range(len(seqs))],
            "task_family": "word",
            "label": [names[v] for v in labels],
            "label_index": labels,
        }
    )
    return SequenceSet(meta, [seqs[i] for i in order])


@pytest.fixture(scope="module")
def sequence_sets():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config.hmm, "n_iter", 30)
        yield {
            "train": _sequence_set(20, seed=1, first_id=1),
            "val": _sequence_set(12, seed=2, first_id=200),
            "test": _sequence_set(15, seed=3, first_id=400),
        }


def _select(sets, n_jobs):
    return select_hmm(sets["train"], sets["val"], seed=42, n_jobs=n_jobs)


def test_select_hmm_serial_and_parallel_identical(sequence_sets):
    best_1, table_1 = _select(sequence_sets, 1)
    best_n, table_n = _select(sequence_sets, PARALLEL)
    assert len(table_1) == len(run_hybrid._grid())
    assert table_1 == table_n
    assert best_1 == best_n
    f1_1 = [row["inner_val_macro_f1"] for row in table_1]
    f1_n = [row["inner_val_macro_f1"] for row in table_n]
    assert len(set(f1_1)) > 1
    assert _max_abs_diff(f1_1, f1_n) == 0.0


def test_select_hmm_log_lines_identical_and_in_grid_order(sequence_sets, caplog):
    messages = {}
    for n_jobs in (1, PARALLEL):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="src.models.hmm"):
            _select(sequence_sets, n_jobs)
        messages[n_jobs] = [(r.levelname, r.getMessage()) for r in caplog.records]
    restarts = config.hybrid.selection_restarts
    assert len(messages[1]) == len(run_hybrid._grid()) * 2 * (restarts + 1)
    assert messages[1] == messages[PARALLEL]


def test_fit_final_serial_and_parallel_identical(sequence_sets, monkeypatch):
    best, _ = _select(sequence_sets, 1)
    out = {}
    for n_jobs in (1, PARALLEL):
        monkeypatch.setattr(config.runner, "hmm_n_jobs", n_jobs)
        clf = fit_final(sequence_sets["train"], sequence_sets["val"], best, 42)
        out[n_jobs] = (clf.predict_proba(sequence_sets["test"].seqs), clf.restart_log)
    assert config.hmm.n_restarts > 1
    assert np.array_equal(out[1][0], out[PARALLEL][0])
    assert _max_abs_diff(out[1][0], out[PARALLEL][0]) == 0.0
    assert out[1][1] == out[PARALLEL][1]


def test_restart_error_is_raised_in_restart_order(sequence_sets, monkeypatch):
    """A failing restart raises the same error on both paths."""
    from src.models import hmm as hmm_module

    def boom(model, X, lengths):
        raise ValueError(f"restart seed {model.random_state} failed")

    monkeypatch.setattr(hmm_module, "_fit_restart", boom)
    for n_jobs in (1, PARALLEL):
        clf = hmm_module.HMMClassifier(config.hmm, 7, n_jobs=n_jobs)
        with pytest.raises(ValueError, match="restart seed 7 failed"):
            clf.fit(sequence_sets["train"].seqs, sequence_sets["train"].labels)


@pytest.fixture(scope="module")
def image_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
        mp.setattr(config.training, "epochs", 1)
        mp.setattr(config.data, "num_workers", 0)
        mp.setattr(config.cnn, "use_pretrained", False)
        mp.setattr(config.hmm, "n_restarts", 3)
        mp.setattr(config.hmm, "n_iter", 20)
        mp.setattr(config.hybrid, "n_states_grid", (2, 3))
        mp.setattr(config.hybrid, "pca_grid", (4, 8))
        mp.setattr(config.hybrid, "topology_grid", ("ergodic", "left_right"))
        mp.setattr(config.hybrid, "selection_restarts", 2)
        root = make_synthetic_root(
            tmp_path_factory.mktemp("parallel_hmm") / "root",
            n_participants=40,
            images=True,
            signal=True,
        )
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        run_cnn.run(task_family="word", device="cpu")
        yield root


def _one_fold(n_jobs, monkeypatch):
    monkeypatch.setattr(config.runner, "hmm_n_jobs", n_jobs)
    meta = config.paths.metadata_dir
    labels = pd.read_csv(meta / "labels.csv", dtype={"participant_id": str})
    parts = pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
    ids = analysis_ids(labels, parts, "primary")
    manifest, _, dropped = load_tables(config)
    return run_hybrid._fold(
        "word",
        0,
        ids,
        [],
        load_folds(config),
        labels,
        manifest,
        dropped,
        "primary",
        "cpu",
        save=False,
    )


def test_one_fold_serial_and_parallel_identical(image_root, monkeypatch):
    crops_1, sel_1 = _one_fold(1, monkeypatch)
    crops_n, sel_n = _one_fold(PARALLEL, monkeypatch)
    assert np.array_equal(crops_1["prob_sad"], crops_n["prob_sad"])
    assert _max_abs_diff(crops_1["prob_sad"], crops_n["prob_sad"]) == 0.0
    assert crops_1.equals(crops_n)
    assert len(sel_1["grid"]) == 8
    assert sel_1["grid"] == sel_n["grid"]
    assert sel_1["chosen"] == sel_n["chosen"]
    assert sel_1["restart_log"] == sel_n["restart_log"]


def test_permutation_draw_serial_and_parallel_identical(image_root, monkeypatch):
    ctx = permutation._context("primary")
    frames = {}
    for n_jobs in (1, PARALLEL):
        monkeypatch.setattr(config.runner, "hmm_n_jobs", n_jobs)
        frames[n_jobs] = permutation.cnn_null_run(ctx, 0, 42, "cpu")
    serial, parallel = frames[1], frames[PARALLEL]
    assert set(serial["model"]) >= {m for m, _ in config.permutation.cnn_models}
    assert serial.to_csv(index=False) == parallel.to_csv(index=False)
    assert _max_abs_diff(serial["prob_sad"], parallel["prob_sad"]) == 0.0


def test_pinned_threads_n_jobs_minus1_matches_serial(sequence_sets, monkeypatch):
    """Deviation 2: with BLAS/OpenMP pinned to one thread in the parent (as
    OMP/MKL/OPENBLAS_NUM_THREADS = 1 do), all cores give the serial result."""
    out = {}
    for n_jobs in (1, -1):
        monkeypatch.setattr(config.runner, "hmm_n_jobs", n_jobs)
        with threadpool_limits(limits=1):
            best, table = select_hmm(
                sequence_sets["train"], sequence_sets["val"], seed=42
            )
            clf = fit_final(sequence_sets["train"], sequence_sets["val"], best, 42)
            proba = clf.predict_proba(sequence_sets["test"].seqs)
        out[n_jobs] = (best, table, proba, clf.restart_log)
    assert out[1][0] == out[-1][0]
    assert out[1][1] == out[-1][1]
    assert np.array_equal(out[1][2], out[-1][2])
    assert _max_abs_diff(out[1][2], out[-1][2]) == 0.0
    assert out[1][3] == out[-1][3]
