"""Placeholder test.

Stands in for the real suite until tests/ is built out against the synthetic
fixture (CLAUDE.md Non-Negotiable 7, Build Order stage A). Checks the same
basic invariants smoke_test.py's stage_a_config already does.
"""

from src.utils.config import config, DRAWING_CROPS, WORD_CROPS, CURSIVE_CROPS


def test_config_imports():
    assert config.project_name == "INSIDE-OUT"


def test_crop_maps_have_expected_counts():
    assert len(DRAWING_CROPS) == 4
    assert len(WORD_CROPS) == 15
    assert len(CURSIVE_CROPS) == 5
