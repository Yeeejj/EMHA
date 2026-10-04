"""Tests for src.analysis.design_table.band_table."""

import pandas as pd

from src.analysis.design_table import band_table
from src.utils.config import config


def _synthetic_labels():
    # cutoff = config.labeling.cutoff = 72.0. Distances from cutoff:
    # 001:48 002:28 003:18 004:8 005:1 | 006:48 007:32 008:22 009:7 010:1
    # Sorted ascending (stable): 005,010,009,004,003,008,002,007,001,006
    return pd.DataFrame(
        {
            "participant_id": [f"{i:03d}" for i in range(1, 11)],
            "total_score": [120, 100, 90, 80, 73, 24, 40, 50, 65, 71],
            "label": ["HAPPY"] * 5 + ["SAD"] * 5,
        }
    )


def test_band_table_matches_hand_computed_counts(monkeypatch):
    monkeypatch.setattr(config.labeling, "cutoff", 72.0)
    labels = _synthetic_labels()

    table = band_table(labels, [0.3])
    row = table.iloc[0]

    # Smallest 3 distances excluded: 005(1), 010(1), 009(7).
    # Kept: 001,002,003,004 (HAPPY) and 006,007,008 (SAD).
    assert row["n_middle_band"] == 3
    assert row["n_kept_happy"] == 4
    assert row["n_kept_sad"] == 3
    assert row["n_kept_total"] == 7
    assert row["happy_score_min"] == 80
    assert row["happy_score_max"] == 120
    assert row["sad_score_min"] == 24
    assert row["sad_score_max"] == 50


def test_band_table_zero_fraction_keeps_everyone(monkeypatch):
    monkeypatch.setattr(config.labeling, "cutoff", 72.0)
    labels = _synthetic_labels()

    table = band_table(labels, [0.0])
    row = table.iloc[0]

    assert row["n_middle_band"] == 0
    assert row["n_kept_happy"] == 5
    assert row["n_kept_sad"] == 5
    assert row["n_kept_total"] == 10


def test_band_table_multiple_fractions_returns_one_row_each(monkeypatch):
    monkeypatch.setattr(config.labeling, "cutoff", 72.0)
    labels = _synthetic_labels()

    table = band_table(labels, [0.2, 0.3, 0.4])

    assert list(table["fraction"]) == [0.2, 0.3, 0.4]
    # Larger fraction -> larger (or equal) middle band, fewer (or equal) kept.
    assert table["n_middle_band"].is_monotonic_increasing
    assert table["n_kept_total"].is_monotonic_decreasing
