"""Tests for src.data.transforms."""

import numpy as np
from PIL import Image
from torchvision import transforms as tv_transforms

from src.data.crop_manifest import measure_baseline_angle
from src.data.transforms import (
    RandomIntegerTranslate,
    get_eval_transform,
    get_train_transform,
)
from src.utils.config import config

BANNED_CLASS_NAMES = {
    "RandomRotation",
    "RandomAffine",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomResizedCrop",
    "ElasticTransform",
}


def _flatten_transform_steps(t):
    steps = []
    if isinstance(t, tv_transforms.Compose):
        for inner in t.transforms:
            steps.extend(_flatten_transform_steps(inner))
    elif isinstance(t, tv_transforms.RandomApply):
        for inner in t.transforms:
            steps.extend(_flatten_transform_steps(inner))
    else:
        steps.append(t)
    return steps


def _make_line_image(
    width=300, height=200, angle_deg=8.0, gray_value=20, thickness=3, bg=250
):
    arr = np.full((height, width), bg, dtype=np.uint8)
    slope = np.tan(np.radians(angle_deg))
    cx, cy = width / 2, height / 2
    for x in range(width):
        y = int(round(cy + slope * (x - cx)))
        for t in range(-(thickness // 2), thickness // 2 + 1):
            yy = y + t
            if 0 <= yy < height:
                arr[yy, x] = gray_value
    return Image.fromarray(arr)


def _unnormalize_to_uint8(tensor):
    arr = tensor.squeeze(0).numpy()
    arr = arr * 0.5 + 0.5
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def test_train_transform_contains_no_banned_classes():
    t = get_train_transform(config, "drawing")

    names = {type(step).__name__ for step in _flatten_transform_steps(t)}

    assert names.isdisjoint(BANNED_CLASS_NAMES)


def test_eval_transform_has_no_augmentation_steps():
    t = get_eval_transform(config, "drawing")
    names = [type(step).__name__ for step in _flatten_transform_steps(t)]

    assert names == ["ToTensor", "Normalize"]
    assert set(names).isdisjoint(BANNED_CLASS_NAMES)


def test_disabled_augment_falls_back_to_eval_transform(monkeypatch):
    monkeypatch.setattr(config.augment, "enabled", False)

    t = get_train_transform(config, "drawing")
    names = [type(step).__name__ for step in _flatten_transform_steps(t)]

    assert names == ["ToTensor", "Normalize"]


def test_random_integer_translate_shifts_by_a_bounded_integer_offset():
    img = Image.new("L", (50, 50), color=255)
    translate = RandomIntegerTranslate(max_translate_px=5, fill=255)

    out = translate(img)

    assert out.size == img.size


def test_train_transform_preserves_eight_degree_angle_over_50_applications(monkeypatch):
    # Keep ink dark-on-light for measure_baseline_angle's convention; this
    # test checks the transform's geometry, independent of pipeline.py's
    # invert step.
    monkeypatch.setattr(config.preprocessing, "invert", False)
    monkeypatch.setattr(config.crop, "ink_threshold", 128)

    img = _make_line_image()
    transform = get_train_transform(config, "drawing")

    for _ in range(50):
        out = transform(img)
        arr = _unnormalize_to_uint8(out)
        angle = measure_baseline_angle(arr)
        assert abs(angle - 8.0) <= 0.5
