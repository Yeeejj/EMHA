"""Tests for src.utils.seed.set_seed."""

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_makes_torch_rand_reproducible():
    set_seed(42)
    first = torch.rand(5)
    set_seed(42)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_set_seed_makes_numpy_rand_reproducible():
    set_seed(42)
    first = np.random.rand(5)
    set_seed(42)
    second = np.random.rand(5)
    assert np.array_equal(first, second)
