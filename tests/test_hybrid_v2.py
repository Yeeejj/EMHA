"""Tests for src.training.run_hybrid_v2 (CNN-HMM v2, exploratory, Deviation 5).

A module fixture builds a synthetic root with signal, runs the baselines
(paired references), and plants stand-in v1 outputs (results/hybrid,
models/hmm, results/final, permutation_runs/primary, RUN_LOG.csv). Tiny HMM
settings keep it fast on CPU.
"""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.analysis.permutation import p_value
from src.data.crop_manifest import ALL_CELLS
from src.models.hmm import HMMClassifier
from src.training import run_baselines, run_hybrid_v2
from src.training.run_baselines import PRED_COLUMNS
from src.training.splits import fold_ids, inner_split
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

V1_FILES = (
    "results/hybrid/primary/predictions.csv",
    "models/hmm/primary/word/fold_0.joblib",
    "results/final/RESULTS.txt",
    "results/final/model_comparison.csv",
    "results/final/permutation_runs/primary/seed_42/perm_0.csv",
)
V2_PREFIXES = (
    "results/hybrid_v2/",
    "models/hmm_v2/",
    "results/final/permutation_runs/primary_v2/",
    "results/final/EXPLORATORY_v2.txt",
)


def _hashes(root) -> dict:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture(scope="module")
def v2_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config.hmm, "n_restarts", 1)
        mp.setattr(config.hmm, "n_iter", 10)
        mp.setattr(config.hybrid, "selection_restarts", 1)
        mp.setattr(config.hybrid_v2, "n_states_grid", (2,))
        mp.setattr(config.hybrid_v2, "pca_grid", (4,))
        mp.setattr(config.hybrid_v2, "topology_grid", ("ergodic", "left_right"))
        mp.setattr(config.runner, "hmm_n_jobs", 1)
        mp.setattr(config.evaluation, "n_bootstrap", 50)
        root = make_synthetic_root(
            tmp_path_factory.mktemp("v2") / "root", n_participants=60, signal=True
        )
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        run_baselines.run("primary")
        base = pd.read_csv(
            root / "results" / "baselines" / "primary" / "predictions.csv",
            dtype={"participant_id": str},
        )
        v1 = base[(base["model"] == "lr_handcrafted") & (base["feature_set"] == "all")]
        v1 = v1.assign(model="cnn_hmm_fused", feature_set="fused")
        for rel in V1_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if rel.startswith("results/hybrid/"):
                v1.to_csv(path, index=False)
            else:
                path.write_text(f"stand-in v1 file {rel}\n")
        before = _hashes(root)
        run_hybrid_v2.run()
        perm = run_hybrid_v2.permutation(n=3, n_jobs=1)
        yield root, before, perm


def _context():
    labels, ids, folds = run_hybrid_v2._context()
    order, array = run_hybrid_v2.load_sequences(ids)
    return labels, ids, folds, dict(zip(order, array))


def test_sequence_order_equals_all_cells(v2_root):
    labels, ids, folds, seqs = _context()
    pid = ids[0]
    assert seqs[pid].shape == (len(ALL_CELLS), 512)
    vectors = {}
    for family in run_hybrid_v2.FAMILIES:
        with np.load(run_hybrid_v2.embedding_path(family)) as z:
            mine = z["participant_id"].astype(str) == pid
            vectors.update(zip(z["cell"][mine].astype(str), z["embeddings"][mine]))
    for k, cell in enumerate(ALL_CELLS):
        np.testing.assert_array_equal(seqs[pid][k], vectors[cell])


def test_missing_cell_stops(v2_root, tmp_path, monkeypatch):
    root, _, _ = v2_root
    copy = tmp_path / "root"
    shutil.copytree(root / "results" / "embeddings", copy / "results" / "embeddings")
    monkeypatch.setattr(config.paths, "output_root", copy)
    path = run_hybrid_v2.embedding_path("word")
    with np.load(path) as z:
        keep = {k: z[k][1:] if z[k].ndim else z[k] for k in z.files}
        dropped = str(z["participant_id"][0])
    np.savez_compressed(path, **keep)
    with pytest.raises(ValueError, match="not imputed"):
        run_hybrid_v2.load_sequences([dropped])


def test_no_outer_test_participant_touches_any_fit(v2_root, monkeypatch):
    labels, ids, folds, seqs = _context()
    owner = {id(a): pid for pid, a in seqs.items()}
    fitted, calibrated = [], []
    real_fit, real_cal = HMMClassifier.fit, HMMClassifier.fit_calibrator

    def spy_fit(self, sequences, y):
        fitted.append({owner[id(s)] for s in sequences})
        return real_fit(self, sequences, y)

    def spy_cal(self, scores, y):
        calibrated.append(len(y))
        return real_cal(self, scores, y)

    monkeypatch.setattr(HMMClassifier, "fit", spy_fit)
    monkeypatch.setattr(HMMClassifier, "fit_calibrator", spy_cal)
    for fold in range(config.cv.n_splits):
        fitted.clear(), calibrated.clear()
        train, test = fold_ids(folds, fold, ids)
        inner_train, inner_val = inner_split(
            train, labels, config.cv.inner_val_fraction, config.training.seed + fold
        )
        preds, _, _ = run_hybrid_v2.run_fold(fold, ids, labels, folds, seqs, n_jobs=1)
        assert set(preds["participant_id"]) == set(test)
        assert fitted and all(f == set(inner_train) for f in fitted)
        assert calibrated == [len(inner_val)]


def test_deterministic_with_fixed_seed(v2_root):
    labels, ids, folds, seqs = _context()
    a = run_hybrid_v2.run_fold(0, ids, labels, folds, seqs, n_jobs=1)[0]
    b = run_hybrid_v2.run_fold(0, ids, labels, folds, seqs, n_jobs=1)[0]
    pd.testing.assert_frame_equal(a, b)


def test_outputs_written(v2_root):
    root, _, _ = v2_root
    out = root / "results" / "hybrid_v2" / "primary"
    preds = pd.read_csv(out / "predictions.csv", dtype={"participant_id": str})
    assert tuple(preds.columns) == PRED_COLUMNS
    assert set(preds["model"]) == {"cnn_hmm_v2"}
    assert set(preds["feature_set"]) == {"task_order"}
    assert set(preds["fold"]) == set(range(config.cv.n_splits))
    assert len(pd.read_csv(out / "selected_settings.csv")) == config.cv.n_splits
    for name in ("run_config.json", "metrics.csv", "per_fold.csv", "paired.csv"):
        assert (out / name).is_file()
    paired = pd.read_csv(out / "paired.csv")
    assert list(paired["reference"]) == [
        "lr_handcrafted/all",
        "lr_embeddings/concat",
        "cnn_hmm_fused/fused",
    ]
    for k in range(config.cv.n_splits):
        assert run_hybrid_v2.model_path(k).is_file()
    text = run_hybrid_v2.report_path().read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "EXPLORATORY, POST HOC (Deviation 5). Not part of the primary claim."
    )
    assert "CNN-HMM v2 (task order) — exploratory" in text


def test_permutation_p_formula(v2_root):
    root, _, perm = v2_root
    runs = run_hybrid_v2.runs_dir(config.training.seed)
    null = [
        json.loads((runs / f"perm_{k}.json").read_text())["macro_f1"] for k in range(3)
    ]
    expected = (1 + sum(v >= perm["observed_macro_f1"] for v in null)) / (1 + 3)
    assert perm["p_value"] == pytest.approx(expected)
    assert p_value(null, perm["observed_macro_f1"]) == pytest.approx(expected)


def test_v1_files_byte_identical_and_only_v2_paths_written(v2_root):
    root, before, _ = v2_root
    after = _hashes(root)
    for rel, digest in before.items():
        assert after.get(rel) == digest, rel
    new = sorted(set(after) - set(before))
    assert new
    assert all(rel.startswith(V2_PREFIXES) for rel in new), new


def test_rerun_resumes_from_cached_folds(v2_root, monkeypatch, capsys):
    root, _, _ = v2_root
    out = root / "results" / "hybrid_v2" / "primary"
    before = (out / "predictions.csv").read_bytes()
    (out / "predictions.csv").unlink()

    def no_refit(*args, **kwargs):
        raise AssertionError("a cached fold was refit")

    monkeypatch.setattr(run_hybrid_v2, "run_fold", no_refit)
    run_hybrid_v2.run()
    assert capsys.readouterr().out.count("cached (fold_") == config.cv.n_splits
    assert (out / "predictions.csv").read_bytes() == before


def test_rerun_with_changed_settings_refuses(v2_root, monkeypatch):
    monkeypatch.setattr(config.hybrid_v2, "pca_grid", (2,))
    with pytest.raises(RuntimeError, match="runs once"):
        run_hybrid_v2.run()


def test_v2_settings_leave_the_run_fingerprint_unchanged(monkeypatch):
    from src.pipeline_runner import config_fingerprint

    before = config_fingerprint()
    monkeypatch.setattr(config.hybrid_v2, "n_permutations", 7)
    assert config_fingerprint() == before
