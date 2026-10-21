"""scripts/merge_draws.py: merging CNN-path draws cached on two machines."""

import json

import pytest

from scripts.merge_draws import merge

SETTINGS = {"analysis": "primary", "seed": 42, "folds_sha256": "abc"}


def _runs(path, settings, draws):
    path.mkdir(parents=True)
    (path / "run_config.json").write_text(json.dumps(settings, indent=2))
    for k in draws:
        (path / f"perm_{k:04d}.csv").write_text(f"perm\n{k}\n")
    return path


def test_merge_copies_missing_draws_only(tmp_path):
    src = _runs(tmp_path / "kaggle", SETTINGS, range(10, 20))
    dst = _runs(tmp_path / "drive", SETTINGS, [*range(10), 10])
    (dst / "perm_0010.csv").write_text("perm\nkept\n")
    copied = merge(src, dst)
    assert copied == [f"perm_{k:04d}.csv" for k in range(11, 20)]
    assert len(list(dst.glob("perm_*.csv"))) == 20
    assert (dst / "perm_0010.csv").read_text() == "perm\nkept\n"
    assert (dst / "perm_0015.csv").read_text() == "perm\n15\n"
    assert not list(dst.glob("*.tmp"))


def test_merge_refuses_different_settings(tmp_path):
    src = _runs(tmp_path / "kaggle", {**SETTINGS, "folds_sha256": "xyz"}, [10])
    dst = _runs(tmp_path / "drive", SETTINGS, [0])
    with pytest.raises(RuntimeError, match="folds_sha256"):
        merge(src, dst)
    assert not (dst / "perm_0010.csv").exists()


def test_merge_into_empty_folder_copies_the_config(tmp_path):
    src = _runs(tmp_path / "kaggle", SETTINGS, [3])
    dst = tmp_path / "drive" / "seed_42"
    merge(src, dst)
    assert json.loads((dst / "run_config.json").read_text()) == SETTINGS
    assert (dst / "perm_0003.csv").is_file()


def test_dry_run_writes_nothing(tmp_path):
    src = _runs(tmp_path / "kaggle", SETTINGS, [3])
    dst = tmp_path / "drive"
    assert merge(src, dst, dry_run=True) == ["perm_0003.csv"]
    assert not dst.exists()


def test_missing_source_config(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        merge(tmp_path / "empty", tmp_path / "drive")
