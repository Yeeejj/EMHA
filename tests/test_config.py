"""Placeholder test.

Stands in for the real suite until tests/ is built out against the synthetic
fixture (CLAUDE.md Non-Negotiable 7, Build Order stage A). Checks the same
basic invariants smoke_test.py's stage_a_config already does, plus the
PathsConfig contract (env overrides, no directory creation on import,
raw dirs never touched, read-only data_root handled without error).
"""

from pathlib import Path

from src.utils.config import (
    Config,
    PathsConfig,
    config,
    DRAWING_CROPS,
    WORD_CROPS,
    CURSIVE_CROPS,
)


def test_config_imports():
    assert config.project_name == "INSIDE-OUT"


def test_crop_maps_have_expected_counts():
    assert len(DRAWING_CROPS) == 4
    assert len(WORD_CROPS) == 15
    assert len(CURSIVE_CROPS) == 5


def test_paths_config_env_overrides(monkeypatch, tmp_path):
    data_root = tmp_path / "custom_data"
    output_root = tmp_path / "custom_out"
    monkeypatch.setenv("EMHA_DATA_ROOT", str(data_root))
    monkeypatch.setenv("EMHA_OUTPUT_ROOT", str(output_root))

    paths = PathsConfig()

    assert paths.data_root == data_root
    assert paths.output_root == output_root
    assert paths.raw3_dir == data_root / "raw-3Page"
    assert paths.raw4_dir == data_root / "raw-4Page"
    assert paths.metadata_dir == data_root / "metadata"
    assert paths.processed_dir == data_root / "processed"
    assert paths.results_dir == output_root / "results"
    assert paths.models_dir == output_root / "models"


def test_constructing_config_creates_no_directories(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    monkeypatch.setenv("EMHA_DATA_ROOT", str(data_root))
    monkeypatch.setenv("EMHA_OUTPUT_ROOT", str(output_root))

    Config()

    assert not data_root.exists()
    assert not output_root.exists()


def test_ensure_output_dirs_never_touches_raw_dirs(tmp_path):
    cfg = Config(
        paths=PathsConfig(data_root=tmp_path / "data", output_root=tmp_path / "out")
    )

    cfg.ensure_output_dirs()

    assert not cfg.paths.raw3_dir.exists()
    assert not cfg.paths.raw4_dir.exists()
    assert cfg.paths.metadata_dir.is_dir()
    assert cfg.paths.processed_dir.is_dir()
    assert cfg.paths.results_dir.is_dir()
    assert cfg.paths.models_dir.is_dir()


def test_ensure_output_dirs_handles_read_only_data_root(monkeypatch, tmp_path, capsys):
    data_root = tmp_path / "data"
    output_root = tmp_path / "out"
    cfg = Config(paths=PathsConfig(data_root=data_root, output_root=output_root))

    real_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if self == data_root or data_root in self.parents:
            raise PermissionError("simulated read-only data_root")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    cfg.ensure_output_dirs()  # must not raise

    assert not cfg.paths.metadata_dir.exists()
    assert not cfg.paths.processed_dir.exists()
    assert cfg.paths.results_dir.is_dir()
    assert cfg.paths.models_dir.is_dir()
    assert "read-only" in capsys.readouterr().out
