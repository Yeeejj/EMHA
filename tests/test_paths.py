"""Manifest paths under PackageConfig.mount_root resolve against the data root.

Kaggle may mount the packaged dataset somewhere other than mount_root; every
loader that opens a manifest path must still find the files, and ordinary
local paths must stay unchanged.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image

from src.data import dataloader
from src.features import embeddings, embeddings_handwriting, handcrafted
from src.features.handcrafted import ALL_FEATURES
from src.utils.config import config
from src.utils.paths import read_manifest, resolve_data_path

FAKE_MOUNT = "/kaggle/input/fake-old-mount"
PIDS = ("001", "002", "003", "004")
LABELS = {"001": "HAPPY", "002": "SAD", "003": "HAPPY", "004": "SAD"}
CELLS = (("D1", "drawing"), ("W1_LH", "word"), ("CS1", "cursive"))


def _mounted_root(tmp_path, monkeypatch):
    """Files under tmp/mounted; every manifest path written under FAKE_MOUNT."""
    root = tmp_path / "mounted"
    meta = root / "metadata"
    meta.mkdir(parents=True)
    rows = []
    for k, pid in enumerate(PIDS):
        for cell, family in CELLS:
            w, h = config.preprocessing.canvas_size[family]
            rel_raw = f"raw/{cell}/{pid}_{cell}.png"
            rel_proc = f"processed/{pid}/{cell}.png"
            for rel in (rel_raw, rel_proc):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((h, w), 40 * k + 20, np.uint8)).save(root / rel)
            rows.append(
                {
                    "participant_id": pid,
                    "cell": cell,
                    "task_family": family,
                    "path": f"{FAKE_MOUNT}/{rel_raw}",
                    "processed_path": f"{FAKE_MOUNT}/{rel_proc}",
                    "baseline_angle_deg": None if family == "drawing" else 0.0,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.drop(columns="processed_path").to_csv(
        meta / "crop_manifest.csv", index=False
    )
    manifest.to_csv(meta / "processed_manifest.csv", index=False)
    qc = manifest[["participant_id", "cell"]].assign(dropped=False)
    qc.to_csv(meta / "qc_log.csv", index=False)
    pd.DataFrame(
        {"participant_id": list(LABELS), "label": list(LABELS.values())}
    ).to_csv(meta / "labels.csv", index=False)
    pd.DataFrame({"participant_id": list(PIDS), "status": "qc_passed"}).to_csv(
        meta / "participants.csv", index=False
    )
    monkeypatch.setattr(config.package, "mount_root", FAKE_MOUNT)
    monkeypatch.setattr(config.paths, "data_root", root)
    monkeypatch.setattr(config.paths, "output_root", tmp_path / "out")
    monkeypatch.setattr(config.labeling, "output_csv", None)
    monkeypatch.setattr(config.data, "num_workers", 0)
    return root


def test_existing_local_paths_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(config.package, "mount_root", FAKE_MOUNT)
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    assert resolve_data_path(str(f)) == str(f)
    monkeypatch.chdir(tmp_path)
    assert resolve_data_path("a.png") == "a.png"


def test_missing_path_outside_mount_root_is_unchanged(monkeypatch):
    monkeypatch.setattr(config.package, "mount_root", FAKE_MOUNT)
    assert resolve_data_path("DATASET/processed/001/D1.png") == (
        "DATASET/processed/001/D1.png"
    )
    assert resolve_data_path(FAKE_MOUNT + "-other/x.png") == FAKE_MOUNT + "-other/x.png"


def test_mount_root_path_rerooted_at_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config.package, "mount_root", FAKE_MOUNT)
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "new-mount")
    expected = str(tmp_path / "new-mount" / "processed" / "001" / "D1.png")
    assert resolve_data_path(f"{FAKE_MOUNT}/processed/001/D1.png") == expected
    backslashed = FAKE_MOUNT.replace("/", "\\") + "\\processed\\001\\D1.png"
    assert resolve_data_path(backslashed) == expected


def test_read_manifest_resolves_both_path_columns_and_keeps_blanks(
    tmp_path, monkeypatch
):
    root = _mounted_root(tmp_path, monkeypatch)
    frame = read_manifest(root / "metadata" / "processed_manifest.csv")
    assert frame["participant_id"].tolist()[:3] == ["001"] * 3
    for col in ("path", "processed_path"):
        assert all(p.startswith(str(root)) for p in frame[col])
        assert all(os.path.isfile(p) for p in frame[col])
    blank = pd.DataFrame({"participant_id": ["001"], "path": [None]})
    blank.to_csv(tmp_path / "blank.csv", index=False)
    assert read_manifest(tmp_path / "blank.csv")["path"].isna().all()


def test_dataloader_load_tables_and_loaders(tmp_path, monkeypatch):
    _mounted_root(tmp_path, monkeypatch)
    manifest, _, _ = dataloader.load_tables(config)
    assert all(os.path.isfile(p) for p in manifest["processed_path"])
    train, val = dataloader.build_loaders(
        ["001", "002", "003"], ["004"], "word", config
    )
    assert sum(len(b["participant_id"]) for b in train) == 3
    assert sum(len(b["participant_id"]) for b in val) == 1


def test_handcrafted_run_opens_resolved_raw_paths(tmp_path, monkeypatch):
    root = _mounted_root(tmp_path, monkeypatch)
    opened = []

    def fake_features(gray, row, cfg):
        opened.append(row["path"])
        return {f: 1.0 for f in ALL_FEATURES}

    monkeypatch.setattr(handcrafted, "crop_features", fake_features)
    handcrafted.run()
    assert len(opened) == len(PIDS) * len(CELLS)
    assert all(p.startswith(str(root)) for p in opened)


def test_embeddings_run_reads_resolved_processed_paths(tmp_path, monkeypatch):
    root = _mounted_root(tmp_path, monkeypatch)
    seen = []

    def fake_embed(paths, task_family, device, model=None):
        seen.extend(paths)
        assert all(p.is_file() for p in paths)
        return np.zeros((len(paths), 4), dtype=np.float32)

    monkeypatch.setattr(embeddings, "load_backbone", lambda device: object())
    monkeypatch.setattr(embeddings, "embed_crops", fake_embed)
    embeddings.run(device="cpu")
    assert len(seen) == len(PIDS) * len(CELLS)
    assert all(str(p).startswith(str(root)) for p in seen)


def test_handwriting_embeddings_run_reads_resolved_processed_paths(
    tmp_path, monkeypatch
):
    root = _mounted_root(tmp_path, monkeypatch)
    seen = []

    def fake_embed(paths, device, encoder=None, processor=None):
        seen.extend(paths)
        assert all(p.is_file() for p in paths)
        return np.zeros((len(paths), 4), dtype=np.float32)

    monkeypatch.setattr(
        embeddings_handwriting, "load_encoder", lambda name, device: (1, 2, "abc")
    )
    monkeypatch.setattr(embeddings_handwriting, "embed_crops", fake_embed)
    embeddings_handwriting.run(device="cpu")
    assert seen and all(str(p).startswith(str(root)) for p in seen)
