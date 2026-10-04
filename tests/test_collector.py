"""Tests for src.data.collector.ParticipantRegistry.

Builds a small ad hoc file tree per test rather than depending on
src/utils/synthetic.py, which doesn't exist yet.
"""

from pathlib import Path

import pandas as pd

from src.data.collector import ParticipantRegistry
from src.utils.config import config


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_raw_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny raw-3Page/raw-4Page tree.

    Codes:
      001 - scan + crops under both raw3_dir and raw4_dir
      002 - scan + crops under both raw3_dir and raw4_dir
      007 - raw3_dir only (leading-zero code, to catch int-coercion bugs)
    """
    raw3_dir = tmp_path / "raw-3Page"
    raw4_dir = tmp_path / "raw-4Page"

    for code in ("001", "002", "007"):
        _touch(raw3_dir / f"EMHA-P3_DrawingExercise_{code}.png")
        for k in (1, 2, 3, 4):
            _touch(raw3_dir / f"D{k}" / f"EMHA-P3_DrawingExercise_{code}_D{k}.png")

    for code in ("001", "002"):
        _touch(raw4_dir / f"EMHA-P4_WritingExercise_{code}.png")
        _touch(raw4_dir / "W1_LH" / f"EMHA-P4_WritingExercise_{code}_W1_LH.png")
        _touch(raw4_dir / "CS1" / f"EMHA-P4_WritingExercise_{code}_CS1.png")

    return raw3_dir, raw4_dir


def test_build_from_files_produces_expected_participant_list(tmp_path, monkeypatch):
    raw3_dir, raw4_dir = _make_raw_dirs(tmp_path)
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "metadata_root")

    df = ParticipantRegistry.build_from_files(raw3_dir, raw4_dir)

    assert list(df["participant_id"]) == ["001", "002", "007"]
    assert list(df["has_p3_scan"]) == [True, True, True]
    assert list(df["has_p4_scan"]) == [True, True, False]
    assert (df["status"] == "indexed").all()
    assert set(df.columns) == {"participant_id", "has_p3_scan", "has_p4_scan", "status"}

    written = config.paths.metadata_dir / "participants.csv"
    assert written.is_file()


def test_load_preserves_leading_zero_ids(tmp_path, monkeypatch):
    raw3_dir, raw4_dir = _make_raw_dirs(tmp_path)
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "metadata_root")

    ParticipantRegistry.build_from_files(raw3_dir, raw4_dir)
    loaded = ParticipantRegistry.load(config.paths.metadata_dir / "participants.csv")

    assert "007" in set(loaded["participant_id"])


def test_validate_flags_malformed_id():
    df = pd.DataFrame(
        {
            "participant_id": ["001", "1a2"],
            "has_p3_scan": [True, True],
            "has_p4_scan": [True, False],
            "status": ["indexed", "indexed"],
        }
    )

    errors = ParticipantRegistry.validate(df)

    assert any("1a2" in e for e in errors)
    assert not any("001" in e for e in errors)


def test_validate_flags_duplicates():
    df = pd.DataFrame(
        {
            "participant_id": ["001", "001"],
            "has_p3_scan": [True, True],
            "has_p4_scan": [True, True],
            "status": ["indexed", "indexed"],
        }
    )

    errors = ParticipantRegistry.validate(df)

    assert any("duplicate" in e and "001" in e for e in errors)


def test_validate_clean_dataframe_has_no_errors():
    df = pd.DataFrame(
        {
            "participant_id": ["001", "002"],
            "has_p3_scan": [True, True],
            "has_p4_scan": [True, True],
            "status": ["indexed", "indexed"],
        }
    )

    assert ParticipantRegistry.validate(df) == []
