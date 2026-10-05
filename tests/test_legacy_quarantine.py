"""The legacy crop-level training stack must fail loudly, never run.

It splits crops rather than participants (CLAUDE.md Non-Negotiable 1), so
every entry point raises LegacyPipelineError until the Stage E/F replacement
exists. These tests also guarantee the modules still import cleanly.
"""

import re
from pathlib import Path

import pytest

from src.data import dataloader
from src.data.dataloader import LegacyPipelineError
from src.training import cross_validate, evaluator, trainer

SRC = Path(dataloader.__file__).resolve().parents[1]
LEGACY_FILES = (
    SRC / "training" / "cross_validate.py",
    SRC / "training" / "trainer.py",
    SRC / "training" / "evaluator.py",
)


def test_cross_validator_is_disabled():
    cv = cross_validate.CrossValidator.__new__(cross_validate.CrossValidator)
    with pytest.raises(LegacyPipelineError, match="CrossValidator"):
        cv.cross_validate(dataset=None)


def test_run_training_is_disabled():
    with pytest.raises(LegacyPipelineError, match="run_training"):
        trainer.run_training(epochs=1)


def test_run_evaluation_and_cv_are_disabled():
    with pytest.raises(LegacyPipelineError, match="run_evaluation"):
        evaluator.run_evaluation(smoke=True)
    with pytest.raises(LegacyPipelineError, match="_run_cross_validation"):
        evaluator._run_cross_validation(None, n_folds=2, cv_epochs=1)


@pytest.mark.parametrize("module", [trainer, evaluator])
def test_cli_exits_with_error_not_traceback(module, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [module.__name__])
    assert module.main() == 1
    assert "legacy pipeline and is disabled" in capsys.readouterr().out


@pytest.mark.parametrize("path", LEGACY_FILES, ids=lambda p: p.name)
def test_no_suppressed_undefined_names_or_removed_transforms(path):
    text = path.read_text(encoding="utf-8")
    assert "noqa: F821" not in text
    assert not re.search(r"\bget_(train|val)_transform\(", text)
    assert "target_size" not in text or "LEGACY" in text


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_no_src_code_reads_folder_splits(path):
    """Acceptance: no src/ module reads LABELED/ or SPLITS/ folders."""
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"(LABELED|SPLITS)/", text), path
    assert "splits_dir" not in text and "HandwritingDataset" not in text
