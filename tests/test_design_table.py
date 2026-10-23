"""Tests for src.analysis.design_table.band_table."""

import pandas as pd

from src.analysis.design_table import band_table
from src.utils.config import config


def _synthetic_labels():
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
    assert table["n_middle_band"].is_monotonic_increasing
    assert table["n_kept_total"].is_monotonic_decreasing
