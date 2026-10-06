"""Tests for src.app.predict (defense demo) on a tiny synthetic root.

A module fixture runs run_cnn (word + cursive), run_hybrid and run_baselines
once with --predict-middle on 40 synthetic participants (CPU, 1 epoch,
no pretrained weights); the demo is then checked against those runs.
"""

import types
from pathlib import Path

import pandas as pd
import pytest

from src.app import predict
from src.training import run_baselines, run_cnn, run_ensemble, run_hybrid
from src.training.splits import analysis_ids, load_folds, middle_band_ids
from src.utils.config import config
from src.utils import protocol_guard
from src.utils.protocol_guard import ProtocolNotFrozenError
from src.utils.synthetic import make_synthetic_root

SMALL_CANVAS = {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)}


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
        mp.setattr(config.training, "epochs", 1)
        mp.setattr(config.training, "batch_size", 32)
        mp.setattr(config.data, "num_workers", 0)
        mp.setattr(config.cnn, "use_pretrained", False)
        mp.setattr(config.hmm, "n_restarts", 1)
        mp.setattr(config.hmm, "n_iter", 20)
        mp.setattr(config.hybrid, "n_states_grid", (2,))
        mp.setattr(config.hybrid, "pca_grid", (4,))
        mp.setattr(config.hybrid, "topology_grid", ("ergodic",))
        mp.setattr(config.hybrid, "selection_restarts", 1)
        mp.setattr(config.demo, "device", "cpu")
        root = make_synthetic_root(
            tmp_path_factory.mktemp("demo") / "root",
            n_participants=40,
            signal=True,
            images=True,
        )
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        for family in config.hybrid.families:
            run_cnn.run(task_family=family, predict_middle=True, device="cpu")
        run_hybrid.run(predict_middle=True, device="cpu")
        run_baselines.run(predict_middle=True)
        run_ensemble.run("primary")
        yield root


def _tables(root):
    labels = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    parts = pd.read_csv(root / "metadata" / "participants.csv", dtype=str)
    return analysis_ids(labels, parts, "primary"), middle_band_ids(labels, parts)


def _sample(root):
    """One primary participant per outer fold plus one middle-band one."""
    ids, middle = _tables(root)
    folds = load_folds(config).set_index("participant_id")
    picks = {int(folds.loc[p, "outer_fold"]): p for p in ids}
    return list(picks.values()) + middle[:1], folds


def test_chosen_fold_model_never_had_participant_in_training(demo_root, monkeypatch):
    fitted, states = [], []
    real_fit, real_load = predict.fit_lr, predict.load_cnn

    def spy_fit(X, y, cfg, pca=False, seed=None):
        fitted.append(set(X.index))
        return real_fit(X, y, cfg, pca=pca, seed=seed)

    def spy_load(path):
        cnn, state = real_load(path)
        states.append(state)
        return cnn, state

    monkeypatch.setattr(predict, "fit_lr", spy_fit)
    monkeypatch.setattr(predict, "load_cnn", spy_load)
    codes, folds = _sample(demo_root)
    assert len(codes) == config.cv.n_splits + 1
    for code in codes:
        fitted.clear(), states.clear()
        result = predict.predict_participant(code, "ensemble")
        k = int(folds.loc[code, "outer_fold"])
        assert result["fold"] == k
        assert len(fitted) == 2 and len(states) == len(config.hybrid.families)
        for train_ids in fitted:
            assert code not in train_ids
        for state in states:
            assert state["fold"] == k
            assert code not in set(state["inner_train_ids"] + state["inner_val_ids"])
            assert code in set(state["test_ids"] + state["middle_ids"])


def test_ensemble_matches_reported_out_of_fold_prediction(demo_root):
    codes, _ = _sample(demo_root)
    result = predict.predict_participant(codes[0])
    assert result["model"] == "ensemble"
    comps = result["component_probs"]
    assert set(comps) == {"lr_handcrafted", "lr_embeddings", "cnn_hmm_fused"}
    assert result["prob_sad"] == pytest.approx(sum(comps.values()) / 3)
    assert set(result["per_family_probs"]) == set(config.hybrid.families)
    assert result["label"] == ("SAD" if result["prob_sad"] >= 0.5 else "HAPPY")
    ens = pd.read_csv(
        demo_root / "results" / "ensemble" / "primary" / "predictions.csv",
        dtype={"participant_id": str},
    ).set_index("participant_id")
    assert result["prob_sad"] == pytest.approx(ens.loc[codes[0], "prob_sad"], abs=1e-4)
    assert (demo_root / "results" / "demo" / codes[0] / "prediction.json").is_file()
    paths = result["gradcam_paths"]  # H4 src.analysis.gradcam overlays
    assert len(paths) == config.demo.n_gradcam
    assert all(Path(p).is_file() for p in paths)


@pytest.mark.parametrize("model", ["cnn_hmm", "lr_handcrafted", "lr_embeddings"])
def test_single_models(demo_root, model):
    codes, _ = _sample(demo_root)
    result = predict.predict_participant(codes[1], model)
    assert list(result["component_probs"]) == [
        "cnn_hmm_fused" if model == "cnn_hmm" else model
    ]
    if model == "cnn_hmm":
        assert set(result["per_family_probs"]) == set(config.hybrid.families)
    else:
        assert result["per_family_probs"] == {}


def test_other_fold_refused(demo_root):
    codes, folds = _sample(demo_root)
    k = int(folds.loc[codes[0], "outer_fold"])
    with pytest.raises(predict.DemoRefusal, match="trained on them"):
        predict.predict_participant(codes[0], fold=(k + 1) % config.cv.n_splits)


def test_unknown_code_refused(demo_root):
    with pytest.raises(predict.DemoRefusal, match="not in the primary"):
        predict.predict_participant("999")


def test_checkpoint_that_trained_on_participant_refused(demo_root, monkeypatch):
    codes, _ = _sample(demo_root)
    real_load = predict.load_cnn

    def tampered(path):
        cnn, state = real_load(path)
        state["inner_train_ids"] = list(state["inner_train_ids"]) + [codes[0]]
        return cnn, state

    monkeypatch.setattr(predict, "load_cnn", tampered)
    with pytest.raises(predict.DemoRefusal, match="was trained on"):
        predict.predict_participant(codes[0], "cnn_hmm")


def test_live_mismatch_with_reported_run_refused(demo_root, monkeypatch):
    codes, _ = _sample(demo_root)
    real = predict.lr_component
    monkeypatch.setattr(predict, "lr_component", lambda *a: real(*a) + 0.25)
    monkeypatch.setattr(config.demo, "match_atol", 1e-6)
    with pytest.raises(predict.DemoRefusal, match="differs from the reported"):
        predict.predict_participant(codes[0], "lr_handcrafted")


def test_gradcam_overlays_saved_when_h4_exists(demo_root, monkeypatch):
    def save_overlay(cnn, image, out_path):
        assert image.dim() == 3
        out_path.write_bytes(b"png")
        return out_path

    stub = types.SimpleNamespace(save_overlay=save_overlay)
    monkeypatch.setattr(predict, "_gradcam_module", lambda: stub)
    codes, _ = _sample(demo_root)
    result = predict.predict_participant(codes[0])
    assert len(result["gradcam_paths"]) == config.demo.n_gradcam
    assert all(p.endswith(".png") for p in result["gradcam_paths"])


def test_cli(demo_root, capsys):
    codes, _ = _sample(demo_root)
    assert predict.main(["--code", codes[0], "--model", "cnn_hmm"]) == 0
    out = capsys.readouterr().out
    assert "Prediction" in out and "P(SAD)" in out and "Per family" in out
    assert predict.main(["--code", "999"]) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_real_data_refused_before_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path)  # no synthetic marker
    monkeypatch.setattr(protocol_guard, "protocol_frozen", lambda: False)
    with pytest.raises(ProtocolNotFrozenError):
        predict.predict_participant("001")
