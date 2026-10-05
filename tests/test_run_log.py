"""Tests for src.utils.run_log: append-only RUN_LOG.csv, point metrics only."""

import csv
import inspect

import pandas as pd
import pytest

from src.training import run_baselines, run_cnn, run_ensemble, run_hybrid
from src.utils import run_log
from src.utils.config import config
from src.utils.run_log import COLUMNS, RunLogError, log_predictions, log_run

METRICS = {
    "model": "lr_handcrafted",
    "feature_set": "all",
    "n_participants": 10,
    "macro_f1": 0.6,
    "accuracy": 0.7,
    "majority_baseline": 0.5,
}


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path / "data")
    monkeypatch.setattr(config.paths, "output_root", tmp_path)
    return tmp_path


def _rows(root):
    with (root / "results" / "RUN_LOG.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _preds(labels_by_fold, pred=None, middle=()):
    """Participant predictions in the shared schema; fold k holds labels[k]."""
    rows = []
    for fold, labels in enumerate(labels_by_fold):
        for i, label in enumerate(labels):
            pid = f"{fold}{i:02d}"
            rows.append(
                {
                    "analysis": "primary",
                    "model": "m",
                    "feature_set": "fs",
                    "fold": fold,
                    "participant_id": pid,
                    "label": label,
                    "prob_sad": 0.5,
                    "pred": pred or label,
                    "in_middle_band": pid in middle,
                }
            )
    return pd.DataFrame(rows)


# ── append-only ───────────────────────────────────────────────────────────────


def test_two_calls_append_two_rows_and_never_truncate(out_root):
    log_run("smoke", "cmd one", "primary", None, METRICS)
    path = out_root / "results" / "RUN_LOG.csv"
    first = path.read_bytes()
    log_run("pilot", "cmd two", "full", "pilot", {**METRICS, "model": "majority"})
    second = path.read_bytes()

    assert second.startswith(first)  # never truncated or rewritten
    rows = _rows(out_root)
    assert len(rows) == 2
    assert tuple(rows[0]) == COLUMNS
    assert [r["command"] for r in rows] == ["cmd one", "cmd two"]
    assert [r["stage"] for r in rows] == ["smoke", "pilot"]
    assert [r["subset"] for r in rows] == [run_log.ALL_PARTICIPANTS, "pilot"]
    assert rows[1]["model"] == "majority"
    assert path.read_text(encoding="utf-8").count("timestamp,") == 1  # one header


def test_row_records_commit_and_utc_timestamp(out_root, monkeypatch):
    monkeypatch.setattr(run_log, "git_commit", lambda: "abc123")
    log_run("full", "cmd", "primary", None, METRICS)
    row = _rows(out_root)[0]
    assert row["git_commit"] == "abc123"
    assert row["timestamp"].endswith("+00:00")
    assert float(row["macro_f1"]) == 0.6 and row["n_participants"] == "10"


def test_existing_file_with_other_header_is_refused_untouched(out_root):
    path = out_root / "results" / "RUN_LOG.csv"
    path.parent.mkdir(parents=True)
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RunLogError):
        log_run("smoke", "cmd", "primary", None, METRICS)
    assert path.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_bad_stage_or_metrics_rejected_before_writing(out_root):
    with pytest.raises(ValueError, match="stage"):
        log_run("final", "cmd", "primary", None, METRICS)
    with pytest.raises(ValueError, match="missing"):
        log_run("smoke", "cmd", "primary", None, {"model": "m"})
    with pytest.raises(ValueError, match="unknown"):
        log_run("smoke", "cmd", "primary", None, {**METRICS, "roc_auc": 0.5})
    assert not (out_root / "results" / "RUN_LOG.csv").exists()


# ── CIs: empty from runners, filled only by the H1 evaluator ─────────────────


def test_runner_rows_leave_ci_empty(out_root):
    preds = _preds([["SAD", "HAPPY"]] * config.cv.n_splits)
    assert log_predictions("smoke", "cmd", "primary", None, preds) == 1
    row = _rows(out_root)[0]
    assert row["macro_f1_ci_low"] == "" and row["macro_f1_ci_high"] == ""


def test_evaluator_rows_can_fill_ci(out_root):
    ci = {"macro_f1_ci_low": 0.41, "macro_f1_ci_high": 0.77}
    log_run(
        "full", "python -m src.training.evaluator", "primary", None, {**METRICS, **ci}
    )
    row = _rows(out_root)[0]
    assert float(row["macro_f1_ci_low"]) == 0.41
    assert float(row["macro_f1_ci_high"]) == 0.77


@pytest.mark.parametrize("module", [run_baselines, run_cnn, run_hybrid, run_ensemble])
def test_runners_never_compute_or_pass_a_ci(module):
    source = inspect.getsource(module)
    for word in ("ci_low", "ci_high", "bootstrap", "log_run("):
        assert word not in source, f"{module.__name__} mentions {word}"
    assert "log_predictions(" in source


# ── point metrics ─────────────────────────────────────────────────────────────


def test_point_metrics_exclude_middle_band():
    labels = [["SAD", "HAPPY", "SAD"]] * config.cv.n_splits
    preds = _preds(labels, middle={"000"})
    (m,) = run_log.point_metrics(preds)
    assert m["n_participants"] == 3 * config.cv.n_splits - 1
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_majority_baseline_uses_training_folds_only():
    # fold 0 is all SAD, the other folds all HAPPY: fold 0's training majority
    # is HAPPY (wrong for all of fold 0); every other fold's training set is
    # split 3 SAD / 9 HAPPY, so HAPPY is right for them.
    n = config.cv.n_splits
    labels = [["SAD"] * 3] + [["HAPPY"] * 3] * (n - 1)
    test = _preds(labels)
    assert run_log.majority_baseline(test) == pytest.approx((n - 1) / n)


def test_majority_tie_goes_to_sad():
    n = config.cv.n_splits
    test = _preds([["SAD", "HAPPY"]] * n)
    # every training set is exactly half SAD -> predict SAD -> half right
    assert run_log.majority_baseline(test) == pytest.approx(0.5)


def test_partial_folds_are_not_logged(out_root, capsys):
    preds = _preds([["SAD", "HAPPY"]] * (config.cv.n_splits - 1))
    assert log_predictions("smoke", "cmd", "primary", None, preds) == 0
    assert "skipped" in capsys.readouterr().out
    assert not (out_root / "results" / "RUN_LOG.csv").exists()


# ── stage and command ─────────────────────────────────────────────────────────


def test_default_stage(out_root, synthetic_root, monkeypatch):
    assert run_log.default_stage(None) == "full"
    assert run_log.default_stage("pilot") == "pilot"
    monkeypatch.setattr(config.paths, "data_root", synthetic_root)
    assert run_log.default_stage("pilot") == "smoke"
    assert run_log.resolve_stage("full", None) == "full"  # explicit wins
    with pytest.raises(ValueError):
        run_log.resolve_stage("final", None)


def test_command_line():
    cmd = run_log.command_line(
        "src.training.run_cnn",
        analysis="primary",
        subset=None,
        folds=[0, 1],
        shuffle_labels=True,
        predict_middle=False,
    )
    assert cmd == (
        "python -m src.training.run_cnn --analysis primary --folds 0,1 "
        "--shuffle-labels"
    )


def test_git_commit_unknown_when_git_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(run_log.subprocess, "run", fail)
    assert run_log.git_commit() == "unknown"
    monkeypatch.undo()
    assert len(run_log.git_commit()) == 40  # this checkout's HEAD
