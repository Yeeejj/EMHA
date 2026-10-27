"""Tests for v3 per-exercise training (exploratory, Deviation 6)."""

from src.data.crop_manifest import ALL_CELLS
from src.utils.config import config


def test_cells_match_crop_manifest():
    assert config.per_task_v3.cells == ALL_CELLS


def test_v3_settings_leave_the_run_fingerprint_unchanged(monkeypatch):
    from src.pipeline_runner import config_fingerprint

    before = config_fingerprint()
    monkeypatch.setattr(config.per_task_v3, "n_lr_permutations", 7)
    assert config_fingerprint() == before
