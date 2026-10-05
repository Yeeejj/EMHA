"""Tests for src.training.run_cnn: end to end on a tiny synthetic image root.

CPU only, epochs = 1, use_pretrained = False, canvases shrunk to 32-64 px so
the full drawing/word/cursive x fold loop runs in seconds.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from src.training import run_cnn
from src.training.run_baselines import PRED_COLUMNS
from src.training.splits import analysis_ids, fold_ids, inner_split, load_folds
from src.training.trainer import Trainer
from src.utils.config import config
from src.utils.protocol_guard import ProtocolNotFrozenError
from src.utils.synthetic import make_synthetic_root

SMALL_CANVAS = {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)}


@pytest.fixture
def cnn_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
    monkeypatch.setattr(config.training, "epochs", 1)
    monkeypatch.setattr(config.training, "batch_size", 16)
    monkeypatch.setattr(config.data, "num_workers", 0)
    monkeypatch.setattr(config.cnn, "use_pretrained", False)
    root = make_synthetic_root(tmp_path / "root", n_participants=30, images=True)
    monkeypatch.setattr(config.paths, "data_root", root)
    monkeypatch.setattr(config.paths, "output_root", root)
    monkeypatch.setattr(config.labeling, "output_csv", None)
    return root


def _count_fits(monkeypatch):
    calls = []
    real = Trainer.fit

    def spy(self, train_loader, val_loader):
        calls.append(train_loader.dataset.table["task_family"].iloc[0])
        return real(self, train_loader, val_loader)

    monkeypatch.setattr(Trainer, "fit", spy)
    return calls


def test_end_to_end_all_families_two_folds(cnn_root):
    pred_path = run_cnn.run(folds=[0, 1], predict_middle=True, device="cpu")
    out = pred_path.parent
    assert out == cnn_root / "results" / "cnn" / "primary"

    preds = pd.read_csv(pred_path, dtype={"participant_id": str})
    assert tuple(preds.columns) == PRED_COLUMNS
    assert set(preds["model"]) == {"cnn_head", "cnn_head_fused"}
    head = preds[preds["model"] == "cnn_head"]
    assert set(head["feature_set"]) == {"drawing", "word", "cursive"}
    assert set(preds.loc[preds["model"] == "cnn_head_fused", "feature_set"]) == {
        "fused"
    }
    assert set(preds["fold"]) == {0, 1}
    assert preds["prob_sad"].between(0, 1).all()
    assert preds["in_middle_band"].any()

    labels = pd.read_csv(
        cnn_root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    parts = pd.read_csv(cnn_root / "metadata" / "participants.csv", dtype=str)
    folds = load_folds(config)
    ids = analysis_ids(labels, parts, "primary")
    tested = set(fold_ids(folds, 0, ids)[1]) | set(fold_ids(folds, 1, ids)[1])
    fused = preds[(preds["model"] == "cnn_head_fused") & ~preds["in_middle_band"]]
    assert set(fused["participant_id"]) == tested

    crops = pd.read_csv(out / "crop_predictions.csv", dtype={"participant_id": str})
    assert len(crops) == len(preds[preds["model"] == "cnn_head"]) // 3 * 24
    for family in run_cnn.FAMILIES:
        for fold in (0, 1):
            ckpt = cnn_root / "models" / "cnn" / "primary" / family / f"fold_{fold}.pth"
            state = torch.load(ckpt, map_location="cpu")
            assert state["best_epoch"] == 1 and state["fold"] == fold
    settings = json.loads((out / "run_config.json").read_text())
    assert settings["training"]["epochs"] == 1


def test_resume_skips_finished_folds(cnn_root, monkeypatch):
    run_cnn.run(task_family="cursive", folds=[0], device="cpu", backbone="simple")
    calls = _count_fits(monkeypatch)
    run_cnn.run(task_family="cursive", folds=[0], device="cpu", backbone="simple")
    assert calls == []
    run_cnn.run(task_family="cursive", folds=[0, 1], device="cpu", backbone="simple")
    assert calls == ["cursive"]
    preds = pd.read_csv(
        cnn_root / "results" / "cnn" / "primary_simple" / "predictions.csv",
        dtype={"participant_id": str},
    )
    assert set(preds["fold"]) == {0, 1}
    assert "cnn_head_fused" not in set(preds["model"])  # needs all 3 families


def test_resume_refuses_changed_settings(cnn_root, monkeypatch):
    run_cnn.run(task_family="cursive", folds=[0], device="cpu", backbone="simple")
    monkeypatch.setattr(config.training, "lr_head", 5e-3)
    with pytest.raises(RuntimeError, match="different settings"):
        run_cnn.run(task_family="cursive", folds=[0], device="cpu", backbone="simple")


def test_leakage_asserted_for_outer_and_inner_splits(cnn_root, monkeypatch):
    calls = []
    real = run_cnn.assert_no_leakage

    def spy(*id_lists):
        calls.append([sorted(map(str, ids)) for ids in id_lists])
        return real(*id_lists)

    monkeypatch.setattr(run_cnn, "assert_no_leakage", spy)
    run_cnn.run(
        task_family="word",
        folds=[0, 2],
        predict_middle=True,
        device="cpu",
        backbone="simple",
    )

    labels = pd.read_csv(
        cnn_root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    parts = pd.read_csv(cnn_root / "metadata" / "participants.csv", dtype=str)
    folds = load_folds(config)
    ids = analysis_ids(labels, parts, "primary")
    assert len(calls) == 4  # outer + inner, for each of the two folds
    for i, fold in enumerate((0, 2)):
        train, test = fold_ids(folds, fold, ids)
        inner_train, inner_val = inner_split(
            train, labels, config.cv.inner_val_fraction, config.training.seed + fold
        )
        outer, inner = calls[2 * i], calls[2 * i + 1]
        assert outer[0] == sorted(train) and outer[1] == sorted(test)
        assert inner[:3] == [sorted(inner_train), sorted(inner_val), sorted(test)]
        assert outer[2] == inner[3] and len(outer[2]) > 0  # middle-band IDs


def test_shuffle_labels_writes_separately_and_keeps_true_test_labels(cnn_root):
    pred_path = run_cnn.run(
        task_family="cursive",
        folds=[0],
        shuffle_labels=True,
        device="cpu",
        backbone="simple",
    )
    assert pred_path.parent == cnn_root / "results" / "cnn_shuffled" / "primary_simple"
    assert (
        cnn_root / "models" / "cnn_shuffled" / "primary_simple" / "cursive"
    ).is_dir()
    preds = pd.read_csv(pred_path, dtype={"participant_id": str})
    labels = pd.read_csv(
        cnn_root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    true = labels.set_index("participant_id")["label"]
    assert (
        preds["label"].to_numpy() == true.loc[preds["participant_id"]].to_numpy()
    ).all()


def test_shuffled_labels_permutes_only_given_ids():
    labels = pd.DataFrame(
        {
            "participant_id": [f"{i:03d}" for i in range(20)],
            "label": ["HAPPY", "SAD"] * 10,
        }
    )
    ids = labels["participant_id"][:12].tolist()
    out = run_cnn.shuffled_labels(labels, ids, seed=1)
    assert (out["label"][12:] == labels["label"][12:]).all()
    assert sorted(out["label"][:12]) == sorted(labels["label"][:12])
    assert not out["label"][:12].equals(labels["label"][:12])
    pd.testing.assert_frame_equal(out, run_cnn.shuffled_labels(labels, ids, seed=1))


def test_simple_backbone_ablation_run_name(cnn_root):
    pred_path = run_cnn.run(
        task_family="cursive", folds=[0], backbone="simple", device="cpu"
    )
    assert pred_path.parent.name == "primary_simple"


def test_pilot_subset_run_name(cnn_root):
    pred_path = run_cnn.run(
        task_family="cursive",
        folds=[0],
        subset="pilot",
        device="cpu",
        backbone="simple",
    )
    assert pred_path.parent.name == "primary_pilot_simple"


def test_refuses_real_data_before_protocol_tag(tmp_path, use_root, monkeypatch):
    use_root(tmp_path)
    monkeypatch.setattr(
        "src.utils.protocol_guard.protocol_frozen", lambda *a, **k: False
    )
    with pytest.raises(ProtocolNotFrozenError):
        run_cnn.run(device="cpu", backbone="simple")


def test_rejects_bad_arguments(cnn_root):
    with pytest.raises(ValueError, match="backbone"):
        run_cnn.run(backbone="vgg", device="cpu")
    with pytest.raises(ValueError, match="primary analysis only"):
        run_cnn.run(
            analysis="full", predict_middle=True, device="cpu", backbone="simple"
        )


def test_trainer_early_stopping_and_balanced_weights(cnn_root, monkeypatch):
    from src.data.dataloader import build_loaders
    from src.models.cnn import EmotionCNN

    monkeypatch.setattr(config.training, "epochs", 4)
    monkeypatch.setattr(config.training, "patience", 1)
    monkeypatch.setattr(config.training, "min_delta", 1.0)  # nothing improves enough
    labels = pd.read_csv(
        cnn_root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    ids = labels["participant_id"].tolist()
    train_loader, val_loader = build_loaders(ids[:20], ids[20:], "cursive", config)
    trainer = Trainer(EmotionCNN(replace(config.cnn, backbone="simple")), config, "cpu")
    history = trainer.fit(train_loader, val_loader)
    assert len(history) == 2 and trainer.best_epoch == 1  # stop after patience

    weights = trainer._criterion(train_loader).weight.numpy()
    counts = np.bincount(train_loader.dataset.table["label_index"], minlength=2)
    assert np.allclose(weights * counts, weights[0] * counts[0])  # inverse frequency
