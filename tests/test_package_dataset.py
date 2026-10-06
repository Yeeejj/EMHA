"""Tests for src.utils.package_dataset on a synthetic root built end to end.

Each test gets a fresh root: synthetic raw crops (shrunk sizes) -> crop
manifest + QC status -> preprocessing, so tests may tamper with files.
"""

import json
import sys

import pandas as pd
import pytest

from src.data import crop_manifest
from src.preprocessing import pipeline
from src.utils import package_dataset as pkgmod
from src.utils.config import config
from src.utils.package_dataset import PackageError, audit_package, build_package
from src.utils.synthetic import make_synthetic_root

N = 10
DIVISOR = 8
MOUNT = config.package.mount_root


def _scaled(sizes):
    return {
        k: (max(8, w // DIVISOR), max(8, h // DIVISOR)) for k, (w, h) in sizes.items()
    }


@pytest.fixture
def root(tmp_path, use_root, monkeypatch):
    monkeypatch.setattr(
        config.crop, "expected_size_px", _scaled(config.crop.expected_size_px)
    )
    monkeypatch.setattr(
        config.preprocessing, "canvas_size", _scaled(config.preprocessing.canvas_size)
    )
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    data = use_root(
        make_synthetic_root(tmp_path / "data", n_participants=N, raw=True, export=True)
    )
    monkeypatch.setattr(sys, "argv", ["crop_manifest", "--apply-qc-status"])
    assert crop_manifest.main() == 0
    pipeline.run()
    return data


def _read(path):
    return pd.read_csv(path, dtype={"participant_id": str})


def _files(pkg, top):
    return sorted(p for p in (pkg / top).rglob("*") if p.is_file())


def test_package_has_no_scans_no_item_columns_and_keeps_layout(root, tmp_path):
    pkg = build_package(None, tmp_path / "dist")
    assert pkg == tmp_path / "dist" / "emha-crops"
    audit_package(pkg)

    # no page scans: every raw file sits in a cell folder; the source has scans
    assert list((root / "raw-3Page").glob("EMHA-P3_DrawingExercise_*.png"))
    for tree in ("raw-3Page", "raw-4Page"):
        assert not [p for p in (pkg / tree).iterdir() if p.is_file()]
    assert len(_files(pkg, "raw-3Page")) + len(_files(pkg, "raw-4Page")) == 24 * N
    assert len(_files(pkg, "processed")) == 24 * N

    # no item-level answers anywhere
    assert not list(pkg.rglob("questionnaire_export.csv"))
    items = set(config.report.item_columns)
    for csv in _files(pkg, "metadata"):
        assert not items & set(_read(csv).columns), csv.name
    assert sorted(p.name for p in _files(pkg, "metadata")) == sorted(
        config.package.metadata_files
    )
    assert "notes" not in _read(pkg / "metadata" / "qc_log.csv").columns

    for name in config.package.verbatim_files:
        assert (pkg / "metadata" / name).read_bytes() == (
            root / "metadata" / name
        ).read_bytes()

    # manifest paths point at packaged files under the Kaggle mount root
    for name, col in (
        ("crop_manifest.csv", "path"),
        ("processed_manifest.csv", "processed_path"),
    ):
        paths = _read(pkg / "metadata" / name)[col]
        assert len(paths) == 24 * N
        for p in paths:
            assert p.startswith(MOUNT + "/")
            assert (pkg / p[len(MOUNT) + 1 :]).is_file()

    meta = json.loads((pkg / "dataset-metadata.json").read_text())
    assert meta["id"] == "tester/emha-crops"
    assert 6 <= len(meta["title"]) <= 50 and len(meta["licenses"]) == 1
    manifest = (pkg / "MANIFEST.txt").read_text()
    for name in ("folds.csv", "labels.csv"):
        assert pkgmod.sha256_of(root / "metadata" / name) in manifest
    assert "kaggle datasets create -p" in manifest and "-r zip" in manifest
    assert f"Participants   : {N}" in manifest


def test_pilot_subset_packages_only_pilot_participants(root, tmp_path, monkeypatch):
    monkeypatch.setattr(config.cv, "pilot_id_range", (1, 4))
    pkg = build_package("pilot", tmp_path / "dist")
    pilot = {f"{i:03d}" for i in range(1, 5)}
    assert {p.name for p in (pkg / "processed").iterdir()} == pilot
    for name in ("crop_manifest.csv", "processed_manifest.csv", "qc_log.csv"):
        assert set(_read(pkg / "metadata" / name)["participant_id"]) == pilot
    assert (pkg / "metadata" / "folds.csv").read_bytes() == (
        root / "metadata" / "folds.csv"
    ).read_bytes()


def _fails(tmp_path, match):
    with pytest.raises(PackageError, match=match):
        build_package(None, tmp_path / "dist")
    assert not (tmp_path / "dist" / "emha-crops").exists()
    assert not (tmp_path / "dist" / ".emha-crops.partial").exists()


@pytest.mark.parametrize("column", ["email", "full_name", "q7"])
def test_forbidden_column_fails(root, tmp_path, column):
    path = root / "metadata" / "participants.csv"
    _read(path).assign(**{column: "x"}).to_csv(path, index=False)
    _fails(tmp_path, "forbidden columns")


def test_notes_with_text_fail(root, tmp_path):
    path = root / "metadata" / "qc_log.csv"
    qc = _read(path)
    qc.loc[0, "notes"] = "reviewer comment"
    qc.to_csv(path, index=False)
    _fails(tmp_path, "notes")


def test_participant_pending_qc_fails(root, tmp_path):
    path = root / "metadata" / "participants.csv"
    parts = _read(path)
    parts.loc[0, "status"] = "indexed"
    parts.to_csv(path, index=False)
    _fails(tmp_path, "not through QC")


def test_missing_processed_crop_fails(root, tmp_path):
    (root / "processed" / "003" / "W2_RH.png").unlink()
    _fails(tmp_path, "completeness")


def test_changed_raw_crop_fails(root, tmp_path):
    crop = next((root / "raw-4Page" / "CS1").glob("*.png"))
    crop.write_bytes(crop.read_bytes() + b"\0")
    _fails(tmp_path, "SHA-256")


def test_existing_output_never_overwritten(root, tmp_path):
    pkg = tmp_path / "dist" / "emha-crops"
    pkg.mkdir(parents=True)
    (pkg / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        build_package(None, tmp_path / "dist")
    assert (pkg / "keep.txt").read_text() == "mine"


def test_tabulation_and_export_never_copied(tmp_path):
    from src.utils.config import _REPO_ROOT

    for bad in (
        _REPO_ROOT / config.package.tabulation_dir / "export.csv",
        tmp_path / "metadata" / "questionnaire_export.csv",
    ):
        with pytest.raises(PackageError, match="never be packaged"):
            pkgmod._check_source(bad)


def test_audit_rejects_a_page_scan(root, tmp_path):
    pkg = build_package(None, tmp_path / "dist")
    scan = next((root / "raw-3Page").glob("EMHA-P3_DrawingExercise_*.png"))
    (pkg / "raw-3Page" / scan.name).write_bytes(scan.read_bytes())
    with pytest.raises(PackageError, match="unexpected file"):
        audit_package(pkg)


def test_username_from_kaggle_json_or_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    with pytest.raises(PackageError, match="username"):
        pkgmod.kaggle_username()
    (tmp_path / "kaggle.json").write_text('{"username": "someone", "key": "secret"}')
    assert pkgmod.kaggle_username() == "someone"


def test_cli(root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["package_dataset", "--out-dir", str(tmp_path / "dist")]
    )
    assert pkgmod.main() == 0
    assert "Package written" in capsys.readouterr().out
    assert pkgmod.main() == 1  # second build refuses to overwrite
    assert "already exists" in capsys.readouterr().out
