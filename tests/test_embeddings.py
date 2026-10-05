"""Tests for src.features.embeddings (weights=None: no download, CPU only)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.features import embeddings
from src.features.embeddings import (
    EMBED_DIM,
    _to_imagenet_input,
    embed_crops,
    load_backbone,
)
from src.utils.config import config


@pytest.fixture(autouse=True)
def _no_download(monkeypatch):
    monkeypatch.setattr(config.embedding, "weights", None)
    monkeypatch.setattr(config.embedding, "batch_size", 3)  # forces 2 batches


def _write_crops(root: Path, family: str, n: int, seed: int = 0) -> list:
    w, h = config.preprocessing.canvas_size[family]
    rng = np.random.default_rng(seed)
    paths = []
    for k in range(n):
        img = np.zeros((h, w), dtype=np.uint8)  # inverted canvas: dark paper
        img[h // 4 : h // 2, 10 + 5 * k : 30 + 5 * k] = rng.integers(150, 255)
        path = root / f"{k:03d}_{family}.png"
        Image.fromarray(img).save(path)
        paths.append(path)
    return paths


def test_backbone_outputs_512_and_is_frozen():
    model = load_backbone("cpu")
    assert not model.training
    assert isinstance(model.fc, torch.nn.Identity)
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("family", ["drawing", "word", "cursive"])
def test_embed_four_synthetic_crops_shape(tmp_path, family):
    paths = _write_crops(tmp_path, family, 4)
    torch.manual_seed(0)
    out = embed_crops(paths, family, "cpu", model=load_backbone("cpu"))
    assert out.shape == (4, EMBED_DIM)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_embeddings_follow_path_order_across_batches(tmp_path):
    paths = _write_crops(tmp_path, "word", 4)
    torch.manual_seed(0)
    model = load_backbone("cpu")
    forward = embed_crops(paths, "word", "cpu", model=model)
    backward = embed_crops(paths[::-1], "word", "cpu", model=model)
    np.testing.assert_allclose(forward, backward[::-1], rtol=1e-5, atol=1e-5)


def test_input_is_three_channel_imagenet_normalized():
    white = torch.ones(1, 4, 4)  # eval-transform output for pixel 255
    x = _to_imagenet_input(white)
    assert x.shape == (3, 4, 4)
    expected = (1.0 - torch.tensor(embeddings.IMAGENET_MEAN)) / torch.tensor(
        embeddings.IMAGENET_STD
    )
    torch.testing.assert_close(x[:, 0, 0], expected)


def _write_fixture(tmp_path):
    data_root = tmp_path / "DATASET"
    meta = data_root / "metadata"
    meta.mkdir(parents=True)
    rows = []
    for family in ("word", "cursive"):
        fam_dir = data_root / "processed" / family
        fam_dir.mkdir(parents=True)
        for k, path in enumerate(_write_crops(fam_dir, family, 4)):
            rows.append(
                {
                    "participant_id": f"{k + 1:03d}",
                    "cell": "W1_LH" if family == "word" else "CS1",
                    "task_family": family,
                    "processed_path": str(path),
                }
            )
    pd.DataFrame(rows).to_csv(meta / "processed_manifest.csv", index=False)
    return data_root


def _count_calls(monkeypatch):
    calls = []
    real = embeddings.embed_crops

    def counting(paths, task_family, device, model=None):
        calls.append(task_family)
        return real(paths, task_family, device, model=model)

    monkeypatch.setattr(embeddings, "embed_crops", counting)
    return calls


def test_run_writes_cache_and_reuse_skips_work(tmp_path, monkeypatch):
    data_root = _write_fixture(tmp_path)
    monkeypatch.setattr(config.paths, "data_root", data_root)
    monkeypatch.setattr(config.paths, "output_root", tmp_path / "out")
    calls = _count_calls(monkeypatch)

    first = embeddings.run(device="cpu")
    assert sorted(calls) == ["cursive", "word"]
    assert [p.name for p in first] == ["resnet18_word.npz", "resnet18_cursive.npz"]
    with np.load(first[0]) as cached:
        assert cached["embeddings"].shape == (4, EMBED_DIM)
        assert cached["participant_id"].tolist() == ["001", "002", "003", "004"]
        assert cached["cell"].tolist() == ["W1_LH"] * 4

    calls.clear()
    second = embeddings.run(device="cpu")
    assert calls == []
    assert second == first


def test_changed_processed_image_invalidates_only_its_family(tmp_path, monkeypatch):
    data_root = _write_fixture(tmp_path)
    monkeypatch.setattr(config.paths, "data_root", data_root)
    monkeypatch.setattr(config.paths, "output_root", tmp_path / "out")
    calls = _count_calls(monkeypatch)
    embeddings.run(device="cpu")

    changed = data_root / "processed" / "cursive" / "000_cursive.png"
    img = np.array(Image.open(changed))
    img[0, 0] = 255 - img[0, 0]
    Image.fromarray(img).save(changed)

    calls.clear()
    embeddings.run(device="cpu")
    assert calls == ["cursive"]


def test_module_fits_no_scaler_or_pca():
    source = Path(embeddings.__file__).read_text(encoding="utf-8")
    assert "sklearn" not in source
    assert ".fit(" not in source and "fit_transform" not in source
