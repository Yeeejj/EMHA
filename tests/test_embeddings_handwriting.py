"""Tests for the EXPLORATORY src.features.embeddings_handwriting.

A mocked encoder and image processor stand in for TrOCR, so these run on CPU
with no download and without the optional transformers dependency.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.features import embeddings_handwriting as hw
from src.utils.config import config

HIDDEN = 6
IMAGE_SIZE, PATCH = 8, 4  # -> 4 patch tokens + 1 CLS token
CLS_VALUE = 1000.0


class MockProcessor:
    """Records the images it receives; returns fixed-size pixel_values."""

    def __init__(self):
        self.seen = []

    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        self.seen.extend(images)
        arr = np.stack(
            [np.asarray(im.resize((IMAGE_SIZE, IMAGE_SIZE))) for im in images]
        )
        pixels = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        return {"pixel_values": pixels}


class MockEncoder(torch.nn.Module):
    """ViT-like: one CLS token (value CLS_VALUE) + one token per patch."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_size=IMAGE_SIZE, patch_size=PATCH)
        self.proj = torch.nn.Linear(3 * PATCH * PATCH, HIDDEN)

    def forward(self, pixel_values):
        n = pixel_values.shape[0]
        patches = pixel_values.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n, -1, 3 * PATCH * PATCH)
        tokens = self.proj(patches)
        cls = torch.full((n, 1, HIDDEN), CLS_VALUE)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, tokens], dim=1))


def _write_crop(path: Path, family: str, ink: int = 255) -> Path:
    w, h = config.preprocessing.canvas_size[family]
    img = np.zeros((h, w), dtype=np.uint8)  # stored inverted: dark paper
    img[h // 4 : h // 2, 10:40] = ink  # bright ink
    Image.fromarray(img).save(path)
    return path


def test_embeds_two_synthetic_crops_and_excludes_cls(tmp_path):
    paths = [
        _write_crop(tmp_path / "a.png", "word"),
        _write_crop(tmp_path / "b.png", "cursive"),
    ]
    torch.manual_seed(0)
    out = hw.embed_crops(
        paths, "cpu", encoder=MockEncoder().eval(), processor=MockProcessor()
    )
    assert out.shape == (2, HIDDEN)
    assert out.dtype == np.float32
    assert np.abs(out).max() < CLS_VALUE / 10  # CLS token not in the mean


def test_crops_are_uninverted_to_dark_on_white(tmp_path, monkeypatch):
    monkeypatch.setattr(config.preprocessing, "invert", True)
    proc = MockProcessor()
    path = _write_crop(tmp_path / "a.png", "word")
    hw.embed_crops([path], "cpu", encoder=MockEncoder().eval(), processor=proc)
    rgb = np.asarray(proc.seen[0])
    assert rgb.shape[2] == 3
    assert np.median(rgb) == 255  # paper is white
    assert rgb.min() == 0  # ink is dark


def test_batches_cover_every_crop_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(config.handwriting_embedding, "batch_size", 2)
    paths = [
        _write_crop(tmp_path / f"{k}.png", "word", ink=60 + 60 * k) for k in range(3)
    ]
    torch.manual_seed(0)
    enc = MockEncoder().eval()
    together = hw.embed_crops(paths, "cpu", encoder=enc, processor=MockProcessor())
    single = np.concatenate(
        [
            hw.embed_crops([p], "cpu", encoder=enc, processor=MockProcessor())
            for p in paths
        ]
    )
    np.testing.assert_allclose(together, single, rtol=1e-5, atol=1e-6)


def _fixture(tmp_path, monkeypatch):
    data_root = tmp_path / "DATASET"
    meta = data_root / "metadata"
    proc_dir = data_root / "processed"
    meta.mkdir(parents=True)
    proc_dir.mkdir(parents=True)
    rows = []
    for pid in ("001", "002"):
        for cell, family in (("D1", "drawing"), ("W1_LH", "word"), ("CS1", "cursive")):
            path = _write_crop(proc_dir / f"{pid}_{cell}.png", family)
            rows.append(
                {
                    "participant_id": pid,
                    "cell": cell,
                    "task_family": family,
                    "processed_path": str(path),
                }
            )
    pd.DataFrame(rows).to_csv(meta / "processed_manifest.csv", index=False)
    monkeypatch.setattr(config.paths, "data_root", data_root)
    monkeypatch.setattr(config.paths, "output_root", tmp_path / "out")

    loads = []

    def fake_load_encoder(name, device):
        loads.append(name)
        torch.manual_seed(0)
        return MockEncoder().eval(), MockProcessor(), "abc123"

    monkeypatch.setattr(hw, "load_encoder", fake_load_encoder)
    return loads


def test_run_writes_text_families_only_labelled_exploratory(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    written = hw.run(device="cpu")
    assert [p.name for p in written] == [
        "handwriting_word.npz",
        "handwriting_cursive.npz",
    ]
    with np.load(written[0]) as z:
        assert z["embeddings"].shape == (2, HIDDEN)
        assert z["participant_id"].tolist() == ["001", "002"]
        assert bool(z["exploratory"]) is True
        assert str(z["model_name"]) == config.handwriting_embedding.model_name
        assert str(z["model_commit"]) == "abc123"


def test_cache_reuse_skips_loading_and_embedding(tmp_path, monkeypatch):
    loads = _fixture(tmp_path, monkeypatch)
    first = hw.run(device="cpu")
    assert len(loads) == 1
    second = hw.run(device="cpu")
    assert len(loads) == 1  # encoder never loaded again
    assert second == first


def test_cli_reports_missing_transformers(tmp_path, monkeypatch, capsys):
    _fixture(tmp_path, monkeypatch)

    def missing(name, device):
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(hw, "load_encoder", missing)
    monkeypatch.setattr("sys.argv", ["embeddings_handwriting", "--device", "cpu"])
    assert hw.main() == 1
    assert "transformers is not installed" in capsys.readouterr().out


def test_exploratory_module_is_not_imported_by_preregistered_code():
    src = Path(hw.__file__).resolve().parents[2] / "src"
    import_line = re.compile(
        r"^\s*(from|import)\s+[\w.]*(embeddings_handwriting|features\s+import"
        r"[^\n]*embeddings_handwriting)",
        re.M,
    )
    users = [
        p
        for p in src.rglob("*.py")
        if p.name != "embeddings_handwriting.py"
        and import_line.search(p.read_text(encoding="utf-8"))
    ]
    assert users == []
    assert hw.EXPLORATORY is True


def test_config_names_the_chosen_model():
    hcfg = config.handwriting_embedding
    assert hcfg.model_name == "microsoft/trocr-base-handwritten"
    assert hcfg.batch_size == 16
    assert hcfg.task_families == ("word", "cursive")


@pytest.mark.parametrize("seq_len,expected", [(5, 1), (6, 2), (4, 0)])
def test_prefix_token_count(seq_len, expected):
    assert hw._num_prefix_tokens(MockEncoder(), seq_len) == expected
