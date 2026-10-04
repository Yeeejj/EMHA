"""Tests for src.preprocessing.pipeline."""

import numpy as np

from src.data.crop_manifest import measure_baseline_angle
from src.preprocessing.pipeline import (
    flatten_background,
    ink_mask_for_stats,
    preprocess_crop,
)
from src.utils.config import config


def _blank_page(height=400, width=300, value=250):
    """A roughly uniform "paper" background (not pure white -- real paper varies)."""
    rng = np.random.default_rng(0)
    noise = rng.integers(-3, 4, size=(height, width))
    return np.clip(value + noise, 0, 255).astype(np.uint8)


def _draw_horizontal_stroke(img, row, thickness, gray_value):
    half = thickness // 2
    img = img.copy()
    img[max(0, row - half) : row + half + 1, :] = gray_value
    return img


def _draw_line(width, height, angle_deg, gray_value=20, thickness=3, bg=250):
    img = np.full((height, width), bg, dtype=np.uint8)
    slope = np.tan(np.radians(angle_deg))
    cx, cy = width / 2, height / 2
    for x in range(width):
        y = int(round(cy + slope * (x - cx)))
        for t in range(-(thickness // 2), thickness // 2 + 1):
            yy = y + t
            if 0 <= yy < height:
                img[yy, x] = gray_value
    return img


def test_flatten_background_rescales_uniform_paper_near_255():
    page = _blank_page(value=180)  # unevenly lit, darker paper

    flattened = flatten_background(page, ksize=51)

    assert abs(int(flattened.mean()) - 255) <= 10


def test_output_shape_equals_canvas_size_for_every_family():
    cfg = config.preprocessing
    for family in ("drawing", "word", "cursive"):
        img = _blank_page(height=200, width=150)
        out = preprocess_crop(img, family, cfg)
        canvas_w, canvas_h = cfg.canvas_size[family]
        assert out.shape == (canvas_h, canvas_w)


def test_stroke_twice_as_thick_stays_about_twice_as_thick(monkeypatch):
    cfg = config.preprocessing
    monkeypatch.setattr(cfg, "invert", False)

    thin = _draw_horizontal_stroke(_blank_page(), row=200, thickness=4, gray_value=20)
    thick = _draw_horizontal_stroke(_blank_page(), row=200, thickness=8, gray_value=20)

    out_thin = preprocess_crop(thin, "drawing", cfg)
    out_thick = preprocess_crop(thick, "drawing", cfg)

    def _measured_thickness(out_img):
        col = out_img[:, out_img.shape[1] // 2]
        ink_rows = np.where(col < 128)[0]
        return ink_rows.max() - ink_rows.min() + 1 if ink_rows.size else 0

    t1 = _measured_thickness(out_thin)
    t2 = _measured_thickness(out_thick)

    assert t1 > 0 and t2 > 0
    ratio = t2 / t1
    assert 1.6 <= ratio <= 2.4


def test_eight_degree_line_keeps_its_angle(monkeypatch):
    cfg = config.preprocessing
    monkeypatch.setattr(cfg, "invert", False)
    monkeypatch.setattr(config.crop, "ink_threshold", 128)

    img = _draw_line(600, 900, angle_deg=8.0, gray_value=20, thickness=4)

    out = preprocess_crop(img, "drawing", cfg)
    angle = measure_baseline_angle(out)

    assert abs(angle - 8.0) <= 0.5


def test_gray_20_stroke_stays_darker_than_gray_90(monkeypatch):
    cfg = config.preprocessing
    monkeypatch.setattr(cfg, "invert", False)

    dark = _draw_horizontal_stroke(_blank_page(), row=200, thickness=6, gray_value=20)
    light = _draw_horizontal_stroke(_blank_page(), row=200, thickness=6, gray_value=90)

    out_dark = preprocess_crop(dark, "drawing", cfg)
    out_light = preprocess_crop(light, "drawing", cfg)

    def _min_value_near_center(out_img):
        col = out_img[:, out_img.shape[1] // 2]
        return int(col.min())

    assert _min_value_near_center(out_dark) < _min_value_near_center(out_light)


def test_ink_mask_for_stats_separates_ink_from_background():
    img = _draw_horizontal_stroke(
        _blank_page(value=250), row=200, thickness=6, gray_value=10
    )

    mask = ink_mask_for_stats(img)

    row = mask[:, mask.shape[1] // 2]
    assert row.max() == 255  # ink pixels detected
    # Most of the page is background, so most of the mask should be 0.
    assert mask.mean() < 50


def test_preprocess_crop_never_mutates_input():
    cfg = config.preprocessing
    img = _blank_page()
    before = img.copy()

    preprocess_crop(img, "word", cfg)

    assert np.array_equal(img, before)
