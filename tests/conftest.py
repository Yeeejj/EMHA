"""Shared pytest fixtures: synthetic data roots (never real data)."""

import pytest

from src.utils.config import config
from src.utils.synthetic import make_synthetic_root


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """Null fixture: features and embeddings independent of the label."""
    return make_synthetic_root(
        tmp_path_factory.mktemp("synthetic_root"), signal=False, handwriting=True
    )


@pytest.fixture(scope="session")
def synthetic_root_signal(tmp_path_factory):
    """Signal fixture: SAD participants shifted on a few features/dimensions."""
    return make_synthetic_root(
        tmp_path_factory.mktemp("synthetic_root_signal"), signal=True, handwriting=True
    )


@pytest.fixture
def use_root(monkeypatch):
    """Point config at a data root (inputs and outputs) for one test."""

    def _use(root):
        monkeypatch.setattr(config.paths, "data_root", root)
        monkeypatch.setattr(config.paths, "output_root", root)
        monkeypatch.setattr(config.labeling, "output_csv", None)
        return root

    return _use


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: needs pretrained weights (download or local torch cache); "
        "skipped when they cannot be loaded. Deselect with -m 'not network'.",
    )
