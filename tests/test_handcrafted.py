"""Tests for src.features.handcrafted, on synthetic crops with known geometry."""

import math

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.features import handcrafted
from src.features.handcrafted import (
    crop_features,
    features_for_family,
    impute_crop_features,
    participant_table,
)
from src.utils.config import config

PX_TO_MM = config.features.px_to_mm
WORD_H, WORD_W = 118, 368


def _paper(height=WORD_H, width=WORD_W, value=245):
    rng = np.random.default_rng(0)
    noise = rng.integers(-3, 4, size=(height, width))
    return np.clip(value + noise, 0, 255).astype(np.uint8)


def _row(task_family="word", baseline=1.5):
    return pd.Series({"task_family": task_family, "baseline_angle_deg": baseline})


def _slanted_strokes(angle_deg, n=8, height=60, thickness=4):
    img = _paper()
    bottom, top = 90, 90 - height
    dx = height * math.tan(math.radians(angle_deg))
    for k in range(n):
        xb = 60 + 30 * k
        xt = int(round(xb + dx))
        cv2.line(img, (xb, bottom), (xt, top), 40, thickness, cv2.LINE_AA)
    return img


def _vertical_bars(bar_w, bar_h, gap, n=6, x0=40, y0=30):
    img = _paper()
    for k in range(n):
        x = x0 + k * (bar_w + gap)
        img[y0 : y0 + bar_h, x : x + bar_w] = 40
    return img


@pytest.mark.parametrize("angle", [-20.0, 0.0, 15.0, 30.0])
def test_slant_within_two_degrees(angle):
    feats = crop_features(_slanted_strokes(angle), _row(), config)
    assert abs(feats["slant_deg"] - angle) <= 2.0


def test_slant_sign_positive_is_rightward():
    feats = crop_features(_slanted_strokes(20.0), _row(), config)
    assert feats["slant_deg"] > 0


@pytest.mark.parametrize("width", [3, 5, 7, 9])
def test_stroke_width_within_one_px(width):
    img = _vertical_bars(bar_w=width, bar_h=60, gap=25)
    feats = crop_features(img, _row(), config)
    assert abs(feats["stroke_width_median_mm"] / PX_TO_MM - width) <= 1.0


@pytest.mark.parametrize("bar_h", [30, 50, 70])
def test_size_within_five_percent(bar_h):
    img = _vertical_bars(bar_w=4, bar_h=bar_h, gap=26)
    feats = crop_features(img, _row(), config)
    expected_mm = bar_h * PX_TO_MM
    assert feats["ink_bbox_height_mm"] == pytest.approx(expected_mm, rel=0.05)
    assert feats["letter_height_mm"] == pytest.approx(expected_mm, rel=0.05)
    expected_w = (6 * 4 + 5 * 26) * PX_TO_MM
    assert feats["ink_bbox_width_mm"] == pytest.approx(expected_w, rel=0.05)


def test_letter_height_ignores_sparse_ascenders():
    img = _vertical_bars(bar_w=4, bar_h=30, gap=26, y0=50)
    img[20:50, 40:44] = 40  # one tall ascender on the first bar
    feats = crop_features(img, _row(), config)
    assert feats["letter_height_mm"] == pytest.approx(30 * PX_TO_MM, rel=0.05)
    assert feats["ink_bbox_height_mm"] == pytest.approx(60 * PX_TO_MM, rel=0.05)


def test_gap_margin_and_component_count():
    img = _vertical_bars(bar_w=10, bar_h=40, gap=20, n=5, x0=50)
    feats = crop_features(img, _row(), config)
    assert feats["n_components"] == 5
    assert feats["inter_component_gap_mm"] == pytest.approx(20 * PX_TO_MM, rel=0.05)
    rightmost = 50 + 4 * 30 + 10 - 1
    expected_margin = (WORD_W - 1 - rightmost) * PX_TO_MM
    assert feats["right_margin_mm"] == pytest.approx(expected_margin, rel=0.05)


def test_gap_not_measured_across_text_lines():
    img = _paper(height=174, width=600)
    img[20:50, 100:110] = 40  # line 1: two bars 30 px apart
    img[20:50, 140:150] = 40
    img[110:140, 300:310] = 40  # line 2: two bars 30 px apart
    img[110:140, 340:350] = 40
    feats = crop_features(img, _row("cursive"), config)
    assert feats["inter_component_gap_mm"] == pytest.approx(30 * PX_TO_MM, rel=0.05)
    assert feats["letter_height_mm"] == pytest.approx(30 * PX_TO_MM, rel=0.05)


def test_box_border_lines_are_ignored():
    clean = _vertical_bars(bar_w=5, bar_h=40, gap=20)
    bordered = clean.copy()
    bordered[3, :] = 30
    bordered[-4, :] = 30
    bordered[:, 2] = 30
    bordered[:, -5] = 30
    a = crop_features(clean, _row(), config)
    b = crop_features(bordered, _row(), config)
    for key in ("n_components", "ink_bbox_height_mm", "right_margin_mm"):
        assert b[key] == pytest.approx(a[key], rel=0.01)


def test_darkness_reflects_ink_value():
    dark = crop_features(_vertical_bars(5, 40, 20), _row(), config)
    light_img = _vertical_bars(5, 40, 20)
    light_img[light_img == 40] = 120
    light = crop_features(light_img, _row(), config)
    assert dark["ink_darkness_mean"] > light["ink_darkness_mean"] + 50


def test_baseline_taken_from_manifest_and_input_untouched():
    img = _vertical_bars(5, 40, 20)
    before = img.copy()
    feats = crop_features(img, _row(baseline=-3.25), config)
    assert feats["baseline_angle_deg"] == -3.25
    np.testing.assert_array_equal(img, before)


def test_drawing_crop_gets_drawing_feature_set():
    img = _paper(300, 300)
    cv2.circle(img, (150, 150), 80, 40, 3)
    feats = crop_features(img, _row("drawing", baseline=float("nan")), config)
    assert set(feats) == set(features_for_family("drawing"))
    assert feats["n_components"] == 1


def test_blank_crop_gives_nan_not_error():
    feats = crop_features(_paper(), _row(), config)
    assert feats["ink_ratio"] == 0.0
    assert math.isnan(feats["slant_deg"])
    assert math.isnan(feats["stroke_width_median_mm"])


def _crop_df():
    rows = []
    for pid in ("001", "002"):
        for cell, family in (
            ("D1", "drawing"),
            ("D2", "drawing"),
            ("W1_LH", "word"),
            ("W1_RH", "word"),
            ("W1_UC", "word"),
            ("CS1", "cursive"),
            ("CS2", "cursive"),
        ):
            rec = {"participant_id": pid, "cell": cell, "task_family": family}
            for i, feat in enumerate(features_for_family(family)):
                rec[feat] = float(i + int(pid) + len(cell))
            rows.append(rec)
    return pd.DataFrame(rows)


def test_imputation_uses_same_participant_family_median():
    df = _crop_df()
    target = (df["participant_id"] == "001") & (df["cell"] == "W1_LH")
    df.loc[target, "slant_deg"] = np.nan
    others = df[
        (df["participant_id"] == "001") & (df["task_family"] == "word") & ~target
    ]["slant_deg"]
    out, log = impute_crop_features(df, config.features)
    assert out.loc[target, "slant_deg"].iloc[0] == others.median()
    assert out.loc[target, "imputed"].iloc[0] == "slant_deg"
    assert log.to_dict("records") == [
        {
            "task_family": "word",
            "feature": "slant_deg",
            "rule": "participant_family_median",
            "n_values": 1,
            "n_participants": 1,
        }
    ]


def test_imputation_falls_back_when_participant_family_all_missing():
    df = _crop_df()
    target = (df["participant_id"] == "002") & (df["task_family"] == "cursive")
    df.loc[target, "inter_component_gap_mm"] = np.nan
    out, log = impute_crop_features(df, config.features)
    assert (out.loc[target, "inter_component_gap_mm"] == 0.0).all()
    assert log["rule"].tolist() == ["fallback_value"]
    assert log["n_values"].tolist() == [2]
    # the other participant's values are never used
    other = (out["participant_id"] == "001") & (out["task_family"] == "cursive")
    assert (out.loc[other, "inter_component_gap_mm"] != 0.0).all()


def test_participant_table_shape_names_and_no_nan():
    table = participant_table(_crop_df())
    assert table["participant_id"].tolist() == ["001", "002"]
    assert "word__slant_deg_mean" in table.columns
    assert "cursive__right_margin_mm_std" in table.columns
    assert "drawing__ink_ratio_mean" in table.columns
    assert "drawing__slant_deg_mean" not in table.columns
    n_cols = 2 * (
        2 * len(features_for_family("word")) + len(features_for_family("drawing"))
    )
    assert table.shape == (2, n_cols + 1)
    assert not table.isna().any().any()


def test_participant_table_mean_and_std_values():
    df = _crop_df()
    df.loc[df["participant_id"] == "001", "ink_ratio"] = [0.1, 0.3, 0, 0, 0, 0, 0]
    table = participant_table(df).set_index("participant_id")
    assert table.loc["001", "drawing__ink_ratio_mean"] == pytest.approx(0.2)
    assert table.loc["001", "drawing__ink_ratio_std"] == pytest.approx(0.1)


def _write_fixture(tmp_path, statuses):
    data_root = tmp_path / "DATASET"
    meta = data_root / "metadata"
    meta.mkdir(parents=True)
    crops = tmp_path / "crops"
    crops.mkdir()
    cells = (
        ("D1", "drawing"),
        ("W1_LH", "word"),
        ("W1_RH", "word"),
        ("CS1", "cursive"),
    )
    rows = []
    for k, pid in enumerate(statuses):
        for cell, family in cells:
            if family == "drawing":
                img = _paper(300, 300)
                cv2.circle(img, (150, 150), 60 + 10 * k, 40, 3)
            elif family == "word":
                img = _slanted_strokes(10.0 + 5 * k)
            else:
                img = _paper(174, 1342)
                img[40:70, 100:400:30] = 40
            path = crops / f"{pid}_{cell}.png"
            Image.fromarray(img).save(path)
            rows.append(
                {
                    "participant_id": pid,
                    "cell": cell,
                    "task_family": family,
                    "path": str(path),
                    "baseline_angle_deg": None if family == "drawing" else 0.5,
                }
            )
    pd.DataFrame(rows).to_csv(meta / "crop_manifest.csv", index=False)
    pd.DataFrame(
        {"participant_id": list(statuses), "status": list(statuses.values())}
    ).to_csv(meta / "participants.csv", index=False)
    return data_root


def test_run_writes_one_row_per_qc_passed_participant(tmp_path, monkeypatch):
    statuses = {"001": "qc_passed", "002": "qc_passed", "003": "indexed"}
    data_root = _write_fixture(tmp_path, statuses)
    monkeypatch.setattr(handcrafted.config.paths, "data_root", data_root)
    monkeypatch.setattr(handcrafted.config.paths, "output_root", tmp_path / "out")

    out_path = handcrafted.run()

    table = pd.read_csv(out_path, dtype={"participant_id": str})
    assert table["participant_id"].tolist() == ["001", "002"]
    assert not table.isna().any().any()
    feat_dir = out_path.parent
    crop_df = pd.read_csv(feat_dir / "handcrafted_crop.csv", dtype=str)
    assert set(crop_df["participant_id"]) == {"001", "002"}
    assert (feat_dir / "handcrafted_imputation_log.csv").is_file()


def test_run_refuses_without_qc_passed(tmp_path, monkeypatch):
    data_root = _write_fixture(tmp_path, {"001": "indexed"})
    monkeypatch.setattr(handcrafted.config.paths, "data_root", data_root)
    monkeypatch.setattr(handcrafted.config.paths, "output_root", tmp_path / "out")
    with pytest.raises(RuntimeError, match="qc_passed"):
        handcrafted.run()


def test_px_to_mm_derives_from_scan_dpi():
    assert PX_TO_MM == pytest.approx(25.4 / config.ingest.expected_dpi)
