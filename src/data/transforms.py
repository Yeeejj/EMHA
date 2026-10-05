"""
Train/eval transforms — Stage D.

Translation, brightness/contrast, and light blur only (CLAUDE.md Accuracy
Strategy): never rotation, affine scale or shear, flips, RandomResizedCrop,
or elastic transforms. Operates on the already-canvas-sized grayscale PIL
images src/preprocessing/pipeline.py produces — no resize here.

    get_train_transform(config, "drawing")
    get_eval_transform(config, "drawing")
"""

from __future__ import annotations

import random
from typing import Callable

from PIL import Image
from torchvision import transforms

NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]
BLUR_KERNEL_SIZE = 3  # light blur only; sigma is the real control (cfg.blur_sigma)


class RandomIntegerTranslate:
    """Shift an image by an integer pixel offset in x and y.

    Pure translation: the affine matrix handed to PIL carries no rotation,
    scale, or shear terms, only the translation offset. Exposed canvas is
    filled with `fill` (the paper/background value).
    """

    def __init__(self, max_translate_px: int, fill: int):
        self.max_translate_px = max_translate_px
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.max_translate_px == 0:
            return img
        dx = random.randint(-self.max_translate_px, self.max_translate_px)
        dy = random.randint(-self.max_translate_px, self.max_translate_px)
        return img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, -dx, 0, 1, -dy),
            fillcolor=self.fill,
        )


def _paper_fill_value(cfg) -> int:
    """0 if the saved crop is inverted (ink bright on dark paper), else 255."""
    return 0 if cfg.preprocessing.invert else 255


def get_train_transform(cfg, task_family: str) -> Callable:
    """Translation + brightness/contrast + occasional light blur, then
    ToTensor + Normalize(mean=[0.5], std=[0.5]).

    task_family is accepted for interface/future-use consistency with
    preprocess_crop(); no current augment parameter varies by family.
    Falls back to get_eval_transform if cfg.augment.enabled is False.
    """
    if not cfg.augment.enabled:
        return get_eval_transform(cfg, task_family)

    aug = cfg.augment
    fill = _paper_fill_value(cfg)

    steps = [
        RandomIntegerTranslate(aug.max_translate_px, fill=fill),
        transforms.ColorJitter(brightness=aug.brightness, contrast=aug.contrast),
        transforms.RandomApply(
            [
                transforms.GaussianBlur(
                    kernel_size=BLUR_KERNEL_SIZE, sigma=aug.blur_sigma
                )
            ],
            p=aug.blur_prob,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
    return transforms.Compose(steps)


def get_eval_transform(cfg, task_family: str) -> Callable:
    """ToTensor + Normalize only — no augmentation.

    task_family is accepted for interface symmetry with get_train_transform;
    unused here since eval has nothing family-specific to apply.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
