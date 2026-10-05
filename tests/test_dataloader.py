"""Tests for src.data.dataloader: CropDataset and build_loaders (Stage E)."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataloader import CropDataset, build_loaders
from src.training.splits import LeakageError
from src.utils.config import config

PIDS = ("001", "002", "003", "004")
LABELS = {"001": "HAPPY", "002": "SAD", "003": "HAPPY", "004": "SAD"}
CELLS = (("D1", "drawing"), ("W1_LH", "word"), ("W2_RH", "word"), ("CS1", "cursive"))


def _write(path, family, value, rgb=False):
    w, h = config.preprocessing.canvas_size[family]
    arr = np.full((h, w), value, dtype=np.uint8)
    img = Image.fromarray(arr)
    (img.convert("RGB") if rgb else img).save(path)
    return str(path)


def _tables(tmp_path, rgb_cell=None):
    rows = []
    for k, pid in enumerate(PIDS):
        for cell, family in CELLS:
            rgb = rgb_cell == (pid, cell)
            path = _write(tmp_path / f"{pid}_{cell}.png", family, 10 * k + 5, rgb)
            rows.append(
                {
                    "participant_id": pid,
                    "cell": cell,
                    "task_family": family,
                    "processed_path": path,
                }
            )
    manifest = pd.DataFrame(rows)
    labels = pd.DataFrame(
        {"participant_id": list(LABELS), "label": list(LABELS.values())}
    )
    return manifest, labels


def test_length_and_participant_filter(tmp_path):
    manifest, labels = _tables(tmp_path)
    ds = CropDataset(manifest, labels, ["001", "003"], None)
    assert len(ds) == 2 * len(CELLS)
    assert ds.participant_ids == {"001", "003"}


def test_label_mapping_and_item_fields(tmp_path):
    manifest, labels = _tables(tmp_path)
    ds = CropDataset(manifest, labels, ["001", "002"], ["word"])
    items = [ds[i] for i in range(len(ds))]
    assert {it["participant_id"]: it["label"] for it in items} == {"001": 0, "002": 1}
    assert config.data.label_to_index == {"HAPPY": 0, "SAD": 1}
    first = items[0]
    assert set(first) == {"image", "label", "participant_id", "task_family", "cell"}
    assert (first["participant_id"], first["cell"]) == ("001", "W1_LH")
    assert first["task_family"] == "word"


def test_task_family_filter(tmp_path):
    manifest, labels = _tables(tmp_path)
    ds = CropDataset(manifest, labels, list(PIDS), ["word"])
    assert len(ds) == 2 * len(PIDS)
    assert set(ds.table["task_family"]) == {"word"}
    both = CropDataset(manifest, labels, list(PIDS), ["drawing", "cursive"])
    assert set(both.table["cell"]) == {"D1", "CS1"}


def test_dropped_crops_excluded(tmp_path):
    manifest, labels = _tables(tmp_path)
    dropped = {("002", "W1_LH"), ("003", "CS1")}
    ds = CropDataset(manifest, labels, list(PIDS), None, dropped=dropped)
    assert len(ds) == len(PIDS) * len(CELLS) - 2
    kept = set(zip(ds.table["participant_id"], ds.table["cell"]))
    assert not kept & dropped


def test_rgb_file_read_as_single_channel_grayscale(tmp_path):
    manifest, labels = _tables(tmp_path, rgb_cell=("001", "D1"))
    ds = CropDataset(manifest, labels, ["001"], ["drawing"])
    image = ds[0]["image"]
    w, h = config.preprocessing.canvas_size["drawing"]
    assert image.shape == (1, h, w)
    assert torch.allclose(image, torch.full_like(image, 5 / 255))


def test_unknown_participant_or_label_raises(tmp_path):
    manifest, labels = _tables(tmp_path)
    with pytest.raises(ValueError, match="no crops"):
        CropDataset(manifest, labels, ["999"], None)
    with pytest.raises(ValueError, match="no label"):
        CropDataset(manifest, labels[labels["participant_id"] != "004"], ["004"], None)
    bad = labels.assign(label=["HAPPY", "SAD", "HAPPY", "NEUTRAL"])
    with pytest.raises(ValueError, match="label_to_index"):
        CropDataset(manifest, bad, list(PIDS), None)


def _write_metadata(tmp_path, monkeypatch, dropped=()):
    data_root = tmp_path / "DATASET"
    meta = data_root / "metadata"
    meta.mkdir(parents=True)
    crops = tmp_path / "crops"
    crops.mkdir()
    manifest, labels = _tables(crops)
    manifest.to_csv(meta / "processed_manifest.csv", index=False)
    labels.to_csv(meta / "labels.csv", index=False)
    qc = manifest[["participant_id", "cell"]].copy()
    qc["dropped"] = [(p, c) in set(dropped) for p, c in zip(qc.participant_id, qc.cell)]
    qc.to_csv(meta / "qc_log.csv", index=False)
    monkeypatch.setattr(config.paths, "data_root", data_root)
    monkeypatch.setattr(config.labeling, "output_csv", None)
    monkeypatch.setattr(config.data, "num_workers", 0)
    monkeypatch.setattr(config.training, "batch_size", 2)


def _ids(loader):
    return {pid for batch in loader for pid in batch["participant_id"]}


def test_build_loaders_disjoint_participants_one_family(tmp_path, monkeypatch):
    _write_metadata(tmp_path, monkeypatch, dropped=[("002", "W2_RH")])
    train, val = build_loaders(["001", "002", "003"], ["004"], "word", config)
    assert _ids(train) == {"001", "002", "003"}
    assert _ids(val) == {"004"}
    assert not _ids(train) & _ids(val)
    assert len(train.dataset) == 3 * 2 - 1  # one dropped crop
    for loader in (train, val):
        for batch in loader:
            assert set(batch["task_family"]) == {"word"}
            assert batch["image"].shape[1:] == (1, 118, 368)


def test_build_loaders_rejects_overlapping_ids(tmp_path, monkeypatch):
    _write_metadata(tmp_path, monkeypatch)
    with pytest.raises(LeakageError, match="both train and val"):
        build_loaders(["001", "002"], ["002", "003"], "word", config)


def test_train_shuffle_is_seeded(tmp_path, monkeypatch):
    _write_metadata(tmp_path, monkeypatch)
    order = []
    for _ in range(2):
        train, _ = build_loaders(list(PIDS[:3]), [PIDS[3]], "word", config)
        order.append([c for b in train for c in b["cell"]])
    assert order[0] == order[1]
