"""Tests for src.analysis.questionnaire_report."""

import hashlib

import pandas as pd
import pytest

from src.analysis.questionnaire_report import (
    alpha_if_item_deleted,
    bootstrap_alpha_ci,
    build_report,
    cronbach_alpha,
)
from src.data.labeler import LabelLoader
from src.utils.config import config


def test_cronbach_alpha_hand_computed_example():
    # item_a = [1,2,3,4], item_b = [2,2,3,5], item_c = [1,3,2,4]
    # Hand-computed: sum(item variances) = 16/3, total variance = 14,
    # alpha = (3/2) * (1 - (16/3)/14) = 13/14.
    items = pd.DataFrame(
        {
            "item_a": [1, 2, 3, 4],
            "item_b": [2, 2, 3, 5],
            "item_c": [1, 3, 2, 4],
        }
    )

    alpha = cronbach_alpha(items)

    assert alpha == pytest.approx(13 / 14, abs=1e-9)


def test_cronbach_alpha_near_one_for_identical_items():
    base = [1, 2, 3, 4, 5, 3, 2, 4]
    items = pd.DataFrame({"a": base, "b": base, "c": base, "d": base})

    alpha = cronbach_alpha(items)

    assert alpha == pytest.approx(1.0, abs=1e-9)


def test_alpha_if_item_deleted_matches_manual_drop():
    items = pd.DataFrame(
        {
            "item_a": [1, 2, 3, 4],
            "item_b": [2, 2, 3, 5],
            "item_c": [1, 3, 2, 4],
        }
    )

    deleted = alpha_if_item_deleted(items)

    assert set(deleted.index) == {"item_a", "item_b", "item_c"}
    assert deleted["item_a"] == pytest.approx(
        cronbach_alpha(items[["item_b", "item_c"]])
    )


def test_bootstrap_alpha_ci_degenerate_for_identical_items():
    base = [1, 2, 3, 4, 5, 3, 2, 4]
    items = pd.DataFrame({"a": base, "b": base, "c": base})

    low, high = bootstrap_alpha_ci(items, n=50, seed=42)

    assert low == pytest.approx(1.0, abs=1e-9)
    assert high == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_alpha_ci_reproducible_with_seed():
    items = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 2, 3, 1, 5],
            "b": [2, 2, 3, 5, 1, 4, 2, 4],
            "c": [1, 3, 2, 4, 3, 2, 1, 5],
        }
    )

    first = bootstrap_alpha_ci(items, n=50, seed=7)
    second = bootstrap_alpha_ci(items, n=50, seed=7)

    assert first == second


def _write_fake_export(path, n_happy=5, n_sad=5):
    """A fake export with real-shaped columns (re, q1..q24, adjusted_total, label).

    Values alternate per-item/per-person (not a flat constant per row) so
    total_score actually varies across participants -- a constant total
    would give zero variance and a NaN alpha, which is a degenerate
    (if harmless) test case worth avoiding.
    """
    rows = []
    pid = 1
    for j in range(n_happy):
        row = {"re": f"P{pid:03d}"}
        for i in range(1, 25):
            row[f"q{i}"] = 4 + ((i + j) % 2)
        row["adjusted_total"] = 110 + j
        row["label"] = "HAPPY"
        rows.append(row)
        pid += 1
    for j in range(n_sad):
        row = {"re": f"P{pid:03d}"}
        for i in range(1, 25):
            row[f"q{i}"] = 1 + ((i + j) % 2)
        row["adjusted_total"] = 30 + j
        row["label"] = "SAD"
        rows.append(row)
        pid += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_report_never_writes_to_export_or_labels_csv(tmp_path, monkeypatch):
    data_root = tmp_path / "data_root"
    output_root = tmp_path / "output_root"
    monkeypatch.setattr(config.paths, "data_root", data_root)
    monkeypatch.setattr(config.paths, "output_root", output_root)
    monkeypatch.setattr(config.labeling, "middle_band_fraction", 0.2)
    monkeypatch.setattr(config.report, "n_bootstrap", 20)

    export_path = config.paths.metadata_dir / "questionnaire_export.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    _write_fake_export(export_path)

    df = LabelLoader.load()
    LabelLoader.write(df)
    labels_path = config.paths.metadata_dir / "labels.csv"

    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    export_hash_before = _hash(export_path)
    labels_hash_before = _hash(labels_path)

    report_path = build_report()

    assert _hash(export_path) == export_hash_before
    assert _hash(labels_path) == labels_hash_before
    assert report_path.is_file()

    text = report_path.read_text(encoding="utf-8")
    assert "alpha" in text.lower()
    assert "nan" not in text.lower()
    assert "HAPPY" in text and "SAD" in text
    assert "Middle-band count" in text

    png_path = config.paths.results_dir / "questionnaire" / "score_distribution.png"
    assert png_path.is_file()


def test_build_report_fails_loudly_on_stale_labels_csv(tmp_path, monkeypatch):
    data_root = tmp_path / "data_root"
    output_root = tmp_path / "output_root"
    monkeypatch.setattr(config.paths, "data_root", data_root)
    monkeypatch.setattr(config.paths, "output_root", output_root)

    export_path = config.paths.metadata_dir / "questionnaire_export.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    _write_fake_export(export_path, n_happy=5, n_sad=5)

    df = LabelLoader.load()
    LabelLoader.write(df)

    # Mutate the export after labels.csv was written, so counts now disagree.
    _write_fake_export(export_path, n_happy=6, n_sad=5)

    with pytest.raises(AssertionError):
        build_report()
