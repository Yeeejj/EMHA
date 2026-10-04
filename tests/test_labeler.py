"""Tests for src.data.labeler.LabelLoader."""

from pathlib import Path

import pandas as pd

from src.data.labeler import LabelLoader
from src.utils.config import config


def _write_fake_export(path: Path, rows: list) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_maps_labels_row_for_row(tmp_path, monkeypatch):
    export = tmp_path / "export.csv"
    _write_fake_export(
        export,
        [
            {"re": "P001", "adjusted_total": 80, "label": "HAPPY"},
            {"re": "P002", "adjusted_total": 60, "label": "SAD"},
            {"re": "P003", "adjusted_total": 72, "label": "HAPPY"},
        ],
    )
    monkeypatch.setattr(config.labeling, "source_csv", export)

    df = LabelLoader.load()

    assert list(df["participant_id"]) == ["001", "002", "003"]
    assert list(df["label"]) == ["HAPPY", "SAD", "HAPPY"]
    assert list(df["total_score"]) == [80.0, 60.0, 72.0]


def test_load_excludes_unlabeled_rows(tmp_path, monkeypatch):
    export = tmp_path / "export.csv"
    _write_fake_export(
        export,
        [
            {"re": "P001", "adjusted_total": 80, "label": "HAPPY"},
            {"re": "P002", "adjusted_total": "", "label": ""},
        ],
    )
    monkeypatch.setattr(config.labeling, "source_csv", export)

    df = LabelLoader.load()

    assert list(df["participant_id"]) == ["001"]


def test_load_reports_id_padding(tmp_path, monkeypatch, capsys):
    export = tmp_path / "export.csv"
    _write_fake_export(export, [{"re": "7", "adjusted_total": 80, "label": "HAPPY"}])
    monkeypatch.setattr(config.labeling, "source_csv", export)

    df = LabelLoader.load()

    assert list(df["participant_id"]) == ["007"]
    assert "007" in capsys.readouterr().out


def test_validate_flags_duplicates():
    df = pd.DataFrame(
        {
            "participant_id": ["001", "001"],
            "total_score": [80.0, 60.0],
            "label": ["HAPPY", "SAD"],
        }
    )

    errors = LabelLoader.validate(df, participants=None)

    assert any("duplicate" in e and "001" in e for e in errors)


def test_validate_flags_malformed_id_bad_label_missing_score():
    df = pd.DataFrame(
        {
            "participant_id": ["1a2", "003"],
            "total_score": [80.0, float("nan")],
            "label": ["MAYBE", "SAD"],
        }
    )

    errors = LabelLoader.validate(df, participants=None)

    assert any("1a2" in e for e in errors)
    assert any("MAYBE" in e for e in errors)
    assert any("003" in e and "score" in e for e in errors)


def test_validate_against_participants_flags_extra_and_missing():
    df = pd.DataFrame(
        {
            "participant_id": ["001", "999"],
            "total_score": [80.0, 60.0],
            "label": ["HAPPY", "SAD"],
        }
    )
    participants = pd.DataFrame(
        {
            "participant_id": ["001", "002"],
            "has_p3_scan": [True, True],
            "has_p4_scan": [True, True],
            "status": ["indexed", "indexed"],
        }
    )

    errors = LabelLoader.validate(df, participants=participants)

    assert any("999" in e and "not in participants" in e for e in errors)
    assert any("002" in e and "no label" in e for e in errors)


def test_write_computes_boundary_distance_and_middle_band(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "data_root")
    monkeypatch.setattr(config.labeling, "cutoff", 72.0)
    monkeypatch.setattr(config.labeling, "middle_band_fraction", 0.5)
    monkeypatch.setattr(config.labeling, "analysis_design", "extreme_groups")

    df = pd.DataFrame(
        {
            "participant_id": ["001", "002", "003", "004"],
            "total_score": [120.0, 24.0, 73.0, 71.0],  # distances: 48, 48, 1, 1
            "label": ["HAPPY", "SAD", "HAPPY", "SAD"],
        }
    )

    out_path = LabelLoader.write(df)
    written = pd.read_csv(out_path, dtype={"participant_id": str})

    assert set(written.columns) == set(
        [
            "participant_id",
            "total_score",
            "label",
            "boundary_distance",
            "in_middle_band",
            "in_primary_analysis",
        ]
    )
    # middle_band_fraction=0.5 of 4 rows = 2 -> the two smallest distances (003, 004)
    band = written.set_index("participant_id")["in_middle_band"].to_dict()
    assert band == {"001": False, "002": False, "003": True, "004": True}

    primary = written.set_index("participant_id")["in_primary_analysis"].to_dict()
    assert primary == {"001": True, "002": True, "003": False, "004": False}

    # labels themselves must be untouched
    assert list(written["label"]) == ["HAPPY", "SAD", "HAPPY", "SAD"]


def test_write_full_sample_marks_everyone_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "data_root")
    monkeypatch.setattr(config.labeling, "analysis_design", "full_sample")

    df = pd.DataFrame(
        {
            "participant_id": ["001", "002"],
            "total_score": [120.0, 24.0],
            "label": ["HAPPY", "SAD"],
        }
    )

    out_path = LabelLoader.write(df)
    written = pd.read_csv(out_path, dtype={"participant_id": str})

    assert written["in_primary_analysis"].all()
