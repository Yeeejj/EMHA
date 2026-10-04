"""Tests for src.data.ingest."""

import hashlib

from PIL import Image

from src.data.ingest import compare_hashes, scan_raw, validate_scans
from src.utils.config import config


def _save(path, size=(100, 130), mode="RGB", dpi=(200, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color=1 if mode == "1" else (255, 255, 255))
    if dpi is not None:
        img.save(path, dpi=dpi)
    else:
        img.save(path)


def _make_clean_fixture(tmp_path):
    raw3 = tmp_path / "raw-3Page"
    raw4 = tmp_path / "raw-4Page"

    # Crops carry no DPI metadata in the real data -- only page scans do.
    for code in ("001", "002"):
        _save(raw3 / f"EMHA-P3_DrawingExercise_{code}.png")
        _save(
            raw3 / "D1" / f"EMHA-P3_DrawingExercise_{code}_D1.png",
            size=(40, 40),
            dpi=None,
        )
        _save(raw4 / f"EMHA-P4_WritingExercise_{code}.png")
        _save(
            raw4 / "CS1" / f"EMHA-P4_WritingExercise_{code}_CS1.png",
            size=(80, 20),
            dpi=None,
        )

    return raw3, raw4


def test_scan_raw_describes_every_file(tmp_path):
    raw3, raw4 = _make_clean_fixture(tmp_path)

    df = scan_raw(raw3, raw4)

    # 2 participants x (1 p3_scan + 1 D1 crop + 1 p4_scan + 1 CS1 crop) = 8
    assert len(df) == 8
    assert set(df["participant_id"]) == {"001", "002"}
    assert set(df["kind"]) == {"p3_scan", "p4_scan", "crop"}
    assert (df["mode"] == "RGB").all()

    scans = df[df["kind"].isin(["p3_scan", "p4_scan"])]
    assert (scans["dpi"].round() == 200).all()
    crops = df[df["kind"] == "crop"]
    assert crops["dpi"].isna().all()


def test_validate_scans_clean_fixture_has_no_problems(tmp_path):
    raw3, raw4 = _make_clean_fixture(tmp_path)
    df = scan_raw(raw3, raw4)

    errors = validate_scans(df)

    assert errors == []


def test_validate_scans_flags_bilevel_and_wrong_name(tmp_path):
    raw3, raw4 = _make_clean_fixture(tmp_path)
    # Deliberately broken: bilevel mode AND a filename that doesn't match
    # the expected EMHA-P3_DrawingExercise_003.png pattern.
    _save(raw3 / "EMHA-P3_DrawingExercise_003_bad.png", mode="1", dpi=(200, 200))
    _save(raw4 / "EMHA-P4_WritingExercise_003.png")

    df = scan_raw(raw3, raw4)
    errors = validate_scans(df)

    assert any("bilevel" in e for e in errors)
    assert any("unexpected file name" in e and "003_bad" in e for e in errors)


def test_validate_scans_flags_missing_p4_scan(tmp_path):
    raw3 = tmp_path / "raw-3Page"
    raw4 = tmp_path / "raw-4Page"
    _save(raw3 / "EMHA-P3_DrawingExercise_001.png")
    # no P4 scan for 001 at all

    df = scan_raw(raw3, raw4)
    errors = validate_scans(df)

    assert any("001" in e and "missing P4 scan" in e for e in errors)


def test_validate_scans_flags_dpi_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ingest, "expected_dpi", 200)
    raw3, raw4 = _make_clean_fixture(tmp_path)
    _save(raw3 / "EMHA-P3_DrawingExercise_003.png", dpi=(150, 150))
    _save(raw4 / "EMHA-P4_WritingExercise_003.png")

    df = scan_raw(raw3, raw4)
    errors = validate_scans(df)

    assert any("DPI" in e and "003" in e for e in errors)


def test_validate_scans_flags_size_outlier(tmp_path):
    raw3, raw4 = _make_clean_fixture(tmp_path)
    # Participant 003's P3 scan is a very different size from 001/002's.
    _save(raw3 / "EMHA-P3_DrawingExercise_003.png", size=(10, 10))
    _save(raw4 / "EMHA-P4_WritingExercise_003.png")

    df = scan_raw(raw3, raw4)
    errors = validate_scans(df)

    assert any("page size off" in e and "003" in e for e in errors)


def test_running_twice_reports_zero_hash_changes(tmp_path):
    raw3, raw4 = _make_clean_fixture(tmp_path)

    first = scan_raw(raw3, raw4)
    second = scan_raw(raw3, raw4)

    assert compare_hashes(first, second) == []


def test_compare_hashes_flags_a_changed_file(tmp_path):
    raw3, raw4 = _make_clean_fixture(tmp_path)
    old = scan_raw(raw3, raw4)

    # Simulate a raw file being modified after the fact.
    _save(raw3 / "EMHA-P3_DrawingExercise_001.png", size=(999, 999))
    new = scan_raw(raw3, raw4)

    changes = compare_hashes(old, new)

    assert len(changes) == 1
    assert "001" in changes[0] and "SHA-256 changed" in changes[0]


def test_scan_raw_writes_nothing_to_raw_dirs(tmp_path):
    raw3, raw4 = _make_clean_fixture(tmp_path)

    def _hashes(root):
        return {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        }

    before = {**_hashes(raw3), **_hashes(raw4)}

    df = scan_raw(raw3, raw4)
    validate_scans(df)

    after = {**_hashes(raw3), **_hashes(raw4)}

    assert before == after
