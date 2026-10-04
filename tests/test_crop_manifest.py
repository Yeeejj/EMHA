"""Tests for src.data.crop_manifest."""

import hashlib

import numpy as np
import pandas as pd
from PIL import Image

from src.data.crop_manifest import (
    CURSIVE_CELLS,
    DRAWING_CELLS,
    WORD_CELLS,
    build_manifest,
    contact_sheets,
    measure_baseline_angle,
    verify,
)
from src.utils.config import config


def _save_crop(path, size, mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (255, 255, 255) if mode == "RGB" else 255
    Image.new(mode, size, color=color).save(path)


def _make_participant(
    raw3, raw4, code, skip_cell=None, wrong_size_cell=None, bad_name_cell=None
):
    sizes = config.crop.expected_size_px

    for cell in DRAWING_CELLS:
        if cell == skip_cell:
            continue
        size = (10, 10) if cell == wrong_size_cell else sizes[cell]
        name = f"EMHA-P3_DrawingExercise_{code}_{cell}.png"
        if cell == bad_name_cell:
            name = f"EMHA-P3_DrawingExercise_{code}_{cell}_bad.png"
        _save_crop(raw3 / cell / name, size)

    for cell in WORD_CELLS + CURSIVE_CELLS:
        if cell == skip_cell:
            continue
        size = (10, 10) if cell == wrong_size_cell else sizes[cell]
        name = f"EMHA-P4_WritingExercise_{code}_{cell}.png"
        if cell == bad_name_cell:
            name = f"EMHA-P4_WritingExercise_{code}_{cell}_bad.png"
        _save_crop(raw4 / cell / name, size)


def _make_fixture(tmp_path):
    raw3 = tmp_path / "raw-3Page"
    raw4 = tmp_path / "raw-4Page"
    _make_participant(raw3, raw4, "001")  # clean
    _make_participant(raw3, raw4, "002", skip_cell="CS5")  # missing crop
    _make_participant(raw3, raw4, "003", wrong_size_cell="D1")  # wrong size
    _make_participant(raw3, raw4, "004", bad_name_cell="W2_RH")  # bad name
    return raw3, raw4


def test_build_manifest_describes_every_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    raw3, raw4 = _make_fixture(tmp_path)

    df = build_manifest()

    # 4 participants x 24 - 1 skipped (002's CS5) = 95
    assert len(df) == 95
    assert set(df["participant_id"]) == {"001", "002", "003", "004"}
    assert set(df["task_family"]) == {"drawing", "word", "cursive"}

    d1 = df[(df["participant_id"] == "001") & (df["cell"] == "D1")].iloc[0]
    assert d1["task"] == "circles"
    assert d1["item"] == "1"
    assert d1["style"] == ""

    w1_lh = df[(df["participant_id"] == "001") & (df["cell"] == "W1_LH")].iloc[0]
    assert w1_lh["task"] == "content"
    assert w1_lh["style"] == "LH"

    # baseline_angle_deg only for word/cursive, never drawing
    assert pd.isna(d1["baseline_angle_deg"])
    assert not pd.isna(w1_lh["baseline_angle_deg"])


def test_verify_flags_missing_wrong_size_and_bad_name(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    _make_fixture(tmp_path)
    df = build_manifest()

    errors = verify(df)

    assert any("002" in e and "missing cell" in e and "CS5" in e for e in errors)
    assert any("003" in e and "size mismatch" in e for e in errors)
    assert any("004" in e and "bad file name" in e for e in errors)


def test_verify_clean_participant_has_no_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    _make_fixture(tmp_path)
    df = build_manifest()

    errors = verify(df)

    assert not any("'001'" in e for e in errors)


def test_verify_does_not_flag_non_grayscale_or_edge_ink(tmp_path, monkeypatch):
    # Every real crop is RGB -- this must never become a "problem" or
    # "zero problems on the first 100 participants" would be impossible.
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    _make_fixture(tmp_path)
    df = build_manifest()

    assert (df["mode"] == "RGB").all()

    errors = verify(df)

    assert not any("grayscale" in e.lower() for e in errors)
    assert not any("edge_ink" in e.lower() for e in errors)


def test_measure_baseline_angle_recovers_four_degree_line():
    width, height = 400, 200
    img = np.full((height, width), 255, dtype=np.uint8)
    slope = np.tan(np.radians(4.0))
    cx, cy = width / 2, height / 2
    for x in range(width):
        y = int(round(cy + slope * (x - cx)))
        for t in (-1, 0, 1):
            yy = y + t
            if 0 <= yy < height:
                img[yy, x] = 0

    angle = measure_baseline_angle(img)

    assert abs(angle - 4.0) <= 1.0


def test_measure_baseline_angle_never_modifies_input():
    img = np.full((50, 200), 255, dtype=np.uint8)
    img[25, :] = 0
    before = img.copy()

    measure_baseline_angle(img)

    assert np.array_equal(img, before)


def _hash_all(root):
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_crop_sha256_unchanged_after_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    monkeypatch.setattr(config.paths, "output_root", tmp_path / "out")
    raw3, raw4 = _make_fixture(tmp_path)

    before = {**_hash_all(raw3), **_hash_all(raw4)}

    df = build_manifest()
    verify(df)
    contact_sheets(df, n=2)

    after = {**_hash_all(raw3), **_hash_all(raw4)}

    assert before == after
