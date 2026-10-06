"""Tests for src.analysis.gradcam: a toy model, the real CNN, and run()."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.analysis import gradcam as gc
from src.models.cnn import EmotionCNN
from src.training import run_cnn
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

H = W = 24
BLOB_A = (slice(2, 8), slice(3, 9))  # bright (+1): the only SAD evidence
BLOB_B = (slice(15, 21), slice(14, 20))  # dark (-1): the only HAPPY evidence


class BlobModel(nn.Module):
    """Channel 0 fires on bright pixels, channel 1 on dark ones; class 1
    reads only channel 0 and class 0 only channel 1 (after global pooling)."""

    def __init__(self):
        super().__init__()
        conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight[:] = torch.tensor([1.0, -1.0]).view(2, 1, 1, 1)
        self.features = nn.Sequential(conv, nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight[:] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


def _blob_image():
    img = torch.zeros(1, H, W)
    img[(0,) + BLOB_A] = 1.0
    img[(0,) + BLOB_B] = -1.0
    return img


def _mass_in(cam, region):
    return cam[region].sum() / cam.sum()


@pytest.mark.parametrize("class_idx, region", [(1, BLOB_A), (0, BLOB_B)])
def test_heat_concentrates_on_the_deciding_blob(class_idx, region):
    cam = gc.gradcam(BlobModel(), _blob_image(), "features", class_idx)
    assert cam.shape == (H, W)
    assert cam.min() >= 0 and cam.max() == pytest.approx(1.0)
    assert _mass_in(cam, region) > 0.9
    peak = np.unravel_index(cam.argmax(), cam.shape)
    assert region[0].start <= peak[0] < region[0].stop
    assert region[1].start <= peak[1] < region[1].stop


def test_cam_is_zero_without_evidence_and_leaves_model_state():
    model = BlobModel().train()
    cam = gc.gradcam(model, torch.zeros(1, 1, H, W), "features", 1)
    assert not cam.any()
    assert model.training and all(p.grad is not None for p in model.parameters())


def test_layer_names_must_resolve_to_one_module():
    with pytest.raises(ValueError, match="matches 0"):
        gc.resolve_layer(BlobModel(), "layer9")
    cnn = EmotionCNN(
        config.cnn.__class__(**{**vars(config.cnn), "use_pretrained": False})
    )
    assert gc.resolve_layer(cnn, "layer4") is cnn.extractor.gradcam_layer


def test_real_cnn_map_and_app_hook(tmp_path):
    cfg = config.cnn.__class__(**{**vars(config.cnn), "use_pretrained": False})
    cnn = EmotionCNN(cfg).eval()
    image = torch.randn(1, 64, 96)
    cam = gc.gradcam(cnn, image, config.gradcam.target_layer, 1)
    assert cam.shape == (64, 96) and 0 <= cam.min() and cam.max() <= 1
    out = gc.save_overlay(cnn, image, tmp_path / "x" / "overlay.png")
    assert out.is_file() and out.stat().st_size > 0


def test_display_image_is_dark_ink_on_white(monkeypatch):
    stored_ink = torch.full((1, 2, 2), 1.0)  # inverted crop: ink bright, +1
    monkeypatch.setattr(config.preprocessing, "invert", True)
    assert gc.display_image(stored_ink).max() == pytest.approx(0.0)
    monkeypatch.setattr(config.preprocessing, "invert", False)
    assert gc.display_image(stored_ink).min() == pytest.approx(1.0)


def test_selection_picks_most_confident_per_category():
    rows = []
    for i, (label, prob) in enumerate(
        [("HAPPY", 0.1), ("HAPPY", 0.3), ("HAPPY", 0.8), ("HAPPY", 0.6)]
        + [("SAD", 0.9), ("SAD", 0.7), ("SAD", 0.2), ("SAD", 0.4)]
    ):
        rows.append(
            {
                "model": "cnn_head",
                "task_family": "word",
                "participant_id": f"{i:03d}",
                "cell": "W1LH",
                "fold": 0,
                "label": label,
                "prob_sad": prob,
                "in_middle_band": False,
            }
        )
    rows.append(
        {**rows[0], "participant_id": "099", "prob_sad": 0.01, "in_middle_band": True}
    )
    picked = gc.select_examples(pd.DataFrame(rows), "word").groupby("category")
    first = {k: g.iloc[0]["prob_sad"] for k, g in picked}
    assert first == {
        "correct_HAPPY": 0.1,  # middle-band 0.01 excluded
        "correct_SAD": 0.9,
        "HAPPY_predicted_SAD": 0.8,
        "SAD_predicted_HAPPY": 0.2,
    }


# ── run() on a tiny trained CNN ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def word_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            config.preprocessing,
            "canvas_size",
            {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)},
        )
        mp.setattr(config.training, "epochs", 1)
        mp.setattr(config.data, "num_workers", 0)
        mp.setattr(config.cnn, "use_pretrained", False)
        root = make_synthetic_root(
            tmp_path_factory.mktemp("gradcam") / "root", n_participants=40, images=True
        )
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        run_cnn.run(task_family="word", predict_middle=True, device="cpu")
        yield root, gc.run("word")


def test_run_writes_grid_and_overlays(word_root):
    root, paths = word_root
    out = root / "results" / "final" / "gradcam"
    assert paths[0] == out / "gradcam_word.png" and paths[0].is_file()
    overlays = paths[1:]
    assert overlays and all(p.parent == out / "word" and p.is_file() for p in overlays)
    assert len(overlays) <= len(gc.CATEGORIES) * config.gradcam.n_examples
    crops = pd.read_csv(gc.crop_predictions_path(), dtype={"participant_id": str})
    middle = set(crops.loc[crops["in_middle_band"], "participant_id"])
    assert middle  # the run predicted middle-band crops too
    assert not any(f"_{pid}_" in p.stem for p in overlays for pid in middle)


def test_run_refuses_a_mismatched_checkpoint(word_root, monkeypatch):
    real = gc._predict
    monkeypatch.setattr(gc, "_predict", lambda cnn, img: (real(cnn, img)[0], 0.123456))
    with pytest.raises(RuntimeError, match="wrong checkpoint"):
        gc.run("word")
