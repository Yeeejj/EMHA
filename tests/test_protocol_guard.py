"""Tests for src.utils.protocol_guard and the synthetic-root marker."""

import pytest

from src.utils import protocol_guard
from src.utils.config import config
from src.utils.protocol_guard import (
    ProtocolNotFrozenError,
    is_synthetic_root,
    require_frozen_or_synthetic,
)


def test_synthetic_root_always_allowed(synthetic_root, use_root, monkeypatch):
    use_root(synthetic_root)
    monkeypatch.setattr(protocol_guard, "protocol_frozen", lambda *a, **k: False)
    assert is_synthetic_root(synthetic_root)
    require_frozen_or_synthetic(config, "test")


def test_real_root_needs_tag(tmp_path, use_root, monkeypatch):
    use_root(tmp_path)
    monkeypatch.setattr(protocol_guard, "protocol_frozen", lambda *a, **k: False)
    with pytest.raises(ProtocolNotFrozenError, match="protocol-frozen"):
        require_frozen_or_synthetic(config, "test")
    monkeypatch.setattr(protocol_guard, "protocol_frozen", lambda *a, **k: True)
    require_frozen_or_synthetic(config, "test")


def test_git_unavailable_counts_as_not_frozen(tmp_path):
    assert protocol_guard.protocol_frozen(tmp_path / "not_a_repo") is False
