"""Tests for src.features.strokes, on synthetic crops with known structure."""

import math

import cv2
import numpy as np
import pytest

from src.features.handcrafted import ink_mask
from src.features.strokes import (
    DRAWING_FEATURES,
    STROKE_FEATURES,
    crossing_number,
    drawing_features,
    path_length_px,
    skeleton_paths,
    stroke_features,
)
from src.preprocessing.pipeline import flatten_background
from src.utils.config import config

MM = config.features.px_to_mm


def _paper(height=118, width=368, value=245):
    rng = np.random.default_rng(0)
    noise = rng.integers(-3, 4, size=(height, width))
    return np.clip(value + noise, 0, 255).astype(np.uint8)


def _prep(img):
    flat = flatten_background(img, config.preprocessing.background_blur_ksize)
    mask, _ = ink_mask(flat, config.features)
    return flat, mask


def _wave(jitter_amp=0.0, jitter_period=10.0, thickness=3):
    img = _paper()
    xs = np.arange(30, 340, 0.5)
    ys = 59 + 25 * np.sin(2 * np.pi * xs / 200)
    ys = ys + jitter_amp * np.sin(2 * np.pi * xs / jitter_period)
    pts = np.round(np.column_stack([xs, ys])).astype(np.int32)
    cv2.polylines(img, [pts], False, 40, thickness, cv2.LINE_AA)
    return img


def test_jittered_curve_has_higher_tremor_than_smooth_curve():
    smooth = stroke_features(*_prep(_wave()), config)["tremor_index"]
    shaky = stroke_features(*_prep(_wave(jitter_amp=1.5)), config)["tremor_index"]
    assert np.isfinite(smooth)
    assert shaky > 1.5 * smooth


def test_tremor_does_not_depend_on_large_scale_curvature():
    straight = _paper()
    cv2.line(straight, (30, 59), (340, 59), 40, 3, cv2.LINE_AA)
    flat_tremor = stroke_features(*_prep(straight), config)["tremor_index"]
    wave_tremor = stroke_features(*_prep(_wave()), config)["tremor_index"]
    shaky = stroke_features(*_prep(_wave(jitter_amp=1.5)), config)["tremor_index"]
    assert abs(wave_tremor - flat_tremor) < 0.5 * (shaky - flat_tremor)


def test_uniform_width_stroke_has_width_cv_near_zero():
    img = _paper()
    img[55:60, 30:340] = 40  # 5 px wide bar
    feats = stroke_features(*_prep(img), config)
    assert feats["stroke_width_cv"] < 0.05


def test_uniform_width_curve_has_low_width_cv():
    feats = stroke_features(*_prep(_wave(thickness=5)), config)
    assert feats["stroke_width_cv"] < 0.15


def test_varying_width_stroke_has_higher_width_cv():
    uniform = _paper()
    uniform[55:62, 30:340] = 40
    tapered = _paper()
    for x in range(30, 340):
        half = 1 + int(4 * (x - 30) / 310)
        tapered[58 - half : 58 + half + 1, x] = 40
    cv_uniform = stroke_features(*_prep(uniform), config)["stroke_width_cv"]
    cv_tapered = stroke_features(*_prep(tapered), config)["stroke_width_cv"]
    assert cv_tapered > cv_uniform + 0.1


def test_varying_darkness_raises_darkness_cv():
    even = _paper()
    even[55:60, 30:340] = 40
    uneven = even.copy()
    uneven[55:60, 185:340] = 130
    cv_even = stroke_features(*_prep(even), config)["darkness_cv"]
    cv_uneven = stroke_features(*_prep(uneven), config)["darkness_cv"]
    assert cv_even < 0.05
    assert cv_uneven > cv_even + 0.1


def test_skeleton_paths_straight_line_is_one_ordered_path():
    mask = np.zeros((40, 200), dtype=bool)
    mask[18:23, 20:180] = True
    paths = skeleton_paths(mask)
    assert len(paths) == 1
    assert path_length_px(paths[0]) == pytest.approx(160, rel=0.05)
    steps = np.abs(np.diff(paths[0], axis=0)).max(axis=1)
    assert (steps == 1).all()  # consecutive pixels are 8-neighbours


def test_skeleton_paths_cross_splits_into_four_arms():
    mask = np.zeros((101, 101), dtype=bool)
    mask[48:53, 10:91] = True
    mask[10:91, 48:53] = True
    paths = skeleton_paths(mask, min_px=10)
    assert len(paths) == 4


def test_endpoint_and_junction_density_of_a_cross():
    img = _paper(200, 200)
    img[98:103, 20:181] = 40
    img[20:181, 98:103] = 40
    flat, mask = _prep(img)
    feats = stroke_features(flat, mask, config)
    from skimage.morphology import skeletonize

    ink_cm = skeletonize(mask).sum() * MM / 10
    assert feats["junctions_per_cm"] == pytest.approx(1 / ink_cm, rel=0.01)
    assert feats["endpoints_per_cm"] == pytest.approx(4 / ink_cm, rel=0.01)
    assert feats["mean_segment_length_mm"] == pytest.approx(78 * MM, rel=0.1)


def test_crossing_number_classifies_end_interior_and_junction():
    skel = np.zeros((7, 7), dtype=bool)
    skel[3, 1:6] = True
    skel[1:3, 3] = True  # T junction at (3, 3)
    cn = crossing_number(skel)
    assert cn[3, 1] == 1
    assert cn[3, 2] == 2
    assert cn[3, 3] == 3


def test_blank_crop_stroke_features_are_nan():
    feats = stroke_features(*_prep(_paper()), config)
    assert set(feats) == set(STROKE_FEATURES)
    assert all(math.isnan(v) for v in feats.values())


def _drawing_feats(img):
    return drawing_features(img, _prep(img)[1], config)


def _drawing(with_smudge=False, with_faint_line=False):
    img = _paper(400, 300)
    cv2.circle(img, (200, 280), 60, 40, 3)  # figure in the lower right
    cv2.circle(img, (180, 265), 4, 40, -1)  # two "eyes" inside
    cv2.circle(img, (220, 265), 4, 40, -1)
    if with_smudge:
        cv2.ellipse(img, (80, 100), (30, 18), 0, 0, 360, 205, -1)
    if with_faint_line:
        cv2.line(img, (30, 60), (130, 160), 205, 1)
    return img


def test_gray_smudge_raises_erasure_score():
    clean = _drawing_feats(_drawing())["erasure_score"]
    smudged = _drawing_feats(_drawing(True))["erasure_score"]
    assert clean < 1e-4
    assert smudged > clean + 0.005


def test_faint_thin_line_is_not_counted_as_erasure():
    clean = _drawing_feats(_drawing())["erasure_score"]
    lined = _drawing_feats(_drawing(with_faint_line=True))
    assert lined["erasure_score"] == pytest.approx(clean, abs=1e-4)


def test_drawing_placement_size_and_detail():
    feats = _drawing_feats(_drawing())
    assert set(feats) == set(DRAWING_FEATURES)
    assert feats["centroid_x_rel"] == pytest.approx(200 / 300, abs=0.02)
    assert feats["centroid_y_rel"] == pytest.approx(280 / 400, abs=0.02)
    assert feats["lower_half_ink_fraction"] == pytest.approx(1.0)
    assert feats["figure_height_mm"] == pytest.approx(123 * MM, rel=0.05)
    assert feats["detail_count"] == 3
    expected_bbox = (123 * 123) / (400 * 300)
    assert feats["ink_bbox_area_fraction"] == pytest.approx(expected_bbox, rel=0.05)
    assert 0.8 < feats["empty_space_ratio"] < 1.0


def test_blank_drawing_has_full_empty_space():
    feats = _drawing_feats(_paper(400, 300))
    assert feats["empty_space_ratio"] == 1.0
    assert feats["detail_count"] == 0.0
    assert math.isnan(feats["centroid_x_rel"])
