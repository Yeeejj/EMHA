"""Tests for src.training.run_hybrid and HybridCNNHMM (tiny CPU settings).

A module fixture builds a small synthetic image root and runs run_cnn once
(word + cursive, all folds, epochs = 1, use_pretrained = False); each test
then starts from empty hybrid outputs.
"""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.hybrid import HybridCNNHMM, cnn_checkpoint_path
from src.training import run_cnn, run_hybrid
from src.training.run_baselines import PRED_COLUMNS
from src.training.splits import analysis_ids, fold_ids, inner_split, load_folds
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

SMALL_CANVAS = {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)}


@pytest.fixture(scope="module")
def hybrid_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
        mp.setattr(config.training, "epochs", 1)
        mp.setattr(config.training, "batch_size", 32)
        mp.setattr(config.data, "num_workers", 0)
        mp.setattr(config.cnn, "use_pretrained", False)
        mp.setattr(config.hmm, "n_restarts", 1)
        mp.setattr(config.hmm, "n_iter", 20)
        mp.setattr(config.hybrid, "n_states_grid", (2,))
        mp.setattr(config.hybrid, "pca_grid", (4, 8))
        mp.setattr(config.hybrid, "topology_grid", ("ergodic", "left_right"))
        mp.setattr(config.hybrid, "selection_restarts", 1)
        root = make_synthetic_root(
            tmp_path_factory.mktemp("hybrid") / "root", n_participants=40, images=True
        )
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        for family in ("word", "cursive", "drawing"):
            run_cnn.run(task_family=family, predict_middle=True, device="cpu")
        yield root


@pytest.fixture
def fresh(hybrid_root):
    for sub in ("results/hybrid", "models/hmm"):
        shutil.rmtree(hybrid_root / sub, ignore_errors=True)
    return hybrid_root


def _ids(root):
    labels = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    parts = pd.read_csv(root / "metadata" / "participants.csv", dtype=str)
    return labels, analysis_ids(labels, parts, "primary")


def _log(root):
    path = root / "results" / "RUN_LOG.csv"
    return pd.read_csv(path, dtype=str) if path.is_file() else pd.DataFrame()


def test_run_cnn_logs_each_family_once_and_fused_when_complete(hybrid_root):
    log = _log(hybrid_root)
    cnn = log[log["command"].str.startswith("python -m src.training.run_cnn")]
    assert sorted(zip(cnn["model"], cnn["feature_set"])) == [
        ("cnn_head", "cursive"),
        ("cnn_head", "drawing"),
        ("cnn_head", "word"),
        ("cnn_head_fused", "fused"),
    ]
    assert (cnn["stage"] == "smoke").all()
    assert cnn["macro_f1_ci_low"].isna().all()


def test_end_to_end_word_and_cursive(fresh):
    before = len(_log(fresh))
    pred_path = run_hybrid.run(predict_middle=True, device="cpu")
    new = _log(fresh).iloc[before:]
    assert sorted(zip(new["model"], new["feature_set"])) == [
        ("cnn_hmm", "cursive"),
        ("cnn_hmm", "word"),
        ("cnn_hmm_fused", "fused"),
    ]
    assert new["macro_f1_ci_low"].isna().all()
    assert new["macro_f1_ci_high"].isna().all()
    out = pred_path.parent
    assert out == fresh / "results" / "hybrid" / "primary"
    preds = pd.read_csv(pred_path, dtype={"participant_id": str})
    assert tuple(preds.columns) == PRED_COLUMNS
    assert set(preds["model"]) == {"cnn_hmm", "cnn_hmm_fused"}
    assert set(preds.loc[preds["model"] == "cnn_hmm", "feature_set"]) == {
        "word",
        "cursive",
    }
    assert set(preds.loc[preds["model"] == "cnn_hmm_fused", "feature_set"]) == {"fused"}
    assert preds["prob_sad"].between(0, 1).all()

    labels, ids = _ids(fresh)
    fused = preds[(preds["model"] == "cnn_hmm_fused") & ~preds["in_middle_band"]]
    assert sorted(fused["participant_id"]) == sorted(ids)  # each tested once
    assert preds["in_middle_band"].any()

    selection = json.loads((out / "selection.json").read_text())
    assert len(selection) == 2 * config.cv.n_splits
    for record in selection:
        assert record["chosen"]["n_states"] == 2
        assert record["chosen"]["pca_components"] in (4, 8)
        assert len(record["grid"]) == 4
        assert record["secondary"] is False
        hmm_file = (
            fresh
            / "models"
            / "hmm"
            / "primary"
            / record["family"]
            / f"fold_{record['fold']}.joblib"
        )
        assert hmm_file.is_file()


def test_no_outer_val_id_used_for_fitting_selection_or_calibration(fresh, monkeypatch):
    seen = []
    real_select, real_final = run_hybrid.select_hmm, run_hybrid.fit_final

    def spy_select(train, val, seed):
        seen.append(("select", seed, train.ids, val.ids))
        return real_select(train, val, seed)

    def spy_final(train, val, setting, seed):
        seen.append(("final", seed, train.ids, val.ids))
        return real_final(train, val, setting, seed)

    monkeypatch.setattr(run_hybrid, "select_hmm", spy_select)
    monkeypatch.setattr(run_hybrid, "fit_final", spy_final)
    run_hybrid.run(predict_middle=True, device="cpu")

    labels, ids = _ids(fresh)
    folds = load_folds(config)
    middle = set(labels.loc[labels["in_middle_band"], "participant_id"])
    assert len(seen) == 2 * 2 * config.cv.n_splits  # select + final, 2 families
    for step, seed, train_ids, val_ids in seen:
        fold = seed - config.training.seed
        train, test = fold_ids(folds, fold, ids)
        inner_train, inner_val = inner_split(
            train, labels, config.cv.inner_val_fraction, seed
        )
        assert train_ids == set(inner_train), step
        assert val_ids == set(inner_val), step
        assert not (train_ids | val_ids) & (set(test) | middle), step


def test_resume_skips_finished_pairs(fresh, monkeypatch):
    run_hybrid.run(device="cpu")
    calls = []
    monkeypatch.setattr(run_hybrid, "_fold", lambda *a, **k: calls.append(a))
    run_hybrid.run(device="cpu")
    assert calls == []


def test_include_drawing_is_secondary_and_not_fused(fresh):
    pred_path = run_hybrid.run(include_drawing=True, device="cpu")
    preds = pd.read_csv(pred_path, dtype={"participant_id": str})
    assert "drawing" in set(preds.loc[preds["model"] == "cnn_hmm", "feature_set"])
    selection = json.loads((pred_path.parent / "selection.json").read_text())
    drawing = [r for r in selection if r["family"] == "drawing"]
    assert drawing and all(r["secondary"] for r in drawing)
    # fused = word + cursive only
    fused = preds[preds["model"] == "cnn_hmm_fused"].set_index(
        ["fold", "participant_id"]
    )
    head = preds[preds["model"] == "cnn_hmm"].pivot_table(
        index=["fold", "participant_id"], columns="feature_set", values="prob_sad"
    )
    expected = head[["word", "cursive"]].mean(axis=1)
    assert np.allclose(fused["prob_sad"], expected.loc[fused.index])


def test_split_mismatch_with_cnn_checkpoint_is_refused(fresh, tmp_path):
    ckpt = cnn_checkpoint_path(config, "primary", "word", 0)
    backup = tmp_path / "fold_0.pth"
    shutil.copy(ckpt, backup)
    try:
        state = torch.load(ckpt, map_location="cpu")
        state["inner_val_ids"] = state["inner_val_ids"][1:]
        torch.save(state, ckpt)
        with pytest.raises(RuntimeError, match="inner_val_ids differ"):
            run_hybrid.run(device="cpu")
    finally:
        shutil.copy(backup, ckpt)


def test_hybrid_predicts_crops_and_participants(fresh):
    run_hybrid.run(device="cpu")
    hybrid = HybridCNNHMM.from_fold(0, "primary", device="cpu")
    pid = "001"
    crops = {
        family: np.stack(
            [
                np.array(Image.open(fresh / "processed" / pid / f"{cell}.png"))
                for cell in cells
            ]
        )
        for family, cells in (("word", ["W1_LH", "W2_RH"]), ("cursive", ["CS1"]))
    }
    label, p = hybrid.predict(crops["word"][0], "word")
    assert label in ("HAPPY", "SAD") and 0 <= p <= 1
    label, p = hybrid.predict_participant(crops)
    expected = np.mean(
        [hybrid.crop_prob_sad(crops[f], f).mean() for f in ("word", "cursive")]
    )
    assert p == pytest.approx(expected)
    assert label == ("SAD" if p >= 0.5 else "HAPPY")
    with pytest.raises(ValueError, match="at least one"):
        hybrid.predict_participant({"drawing": crops["word"]})


def test_missing_cnn_checkpoint_is_reported(fresh, monkeypatch):
    monkeypatch.setattr(config.paths, "output_root", fresh / "elsewhere")
    with pytest.raises(FileNotFoundError, match="run_cnn"):
        run_hybrid.run(device="cpu")
