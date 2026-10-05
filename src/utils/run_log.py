"""
Run log — one append-only CSV row per (run, model, feature_set).

results/RUN_LOG.csv is only ever appended to: the header is written once when
the file is new or empty, and an existing file whose header differs is
refused (never rewritten). Columns, in order:

    timestamp, git_commit, stage, command, analysis, subset, model,
    feature_set, n_participants, macro_f1, macro_f1_ci_low, macro_f1_ci_high,
    accuracy, majority_baseline

Runners (run_baselines, run_cnn, run_hybrid, run_ensemble) call
log_predictions() at the end of run(): point estimates only, on the pooled
out-of-fold participants outside the middle band, with the two CI columns
left empty. Runners never compute CIs; only the H1 evaluator does, and it
appends its own rows through log_run() with macro_f1_ci_low/high filled.

majority_baseline follows EVALUATION_PROTOCOL.md section 2: in each outer
fold, predict the majority label of that fold's training participants (the
other folds' participants), pooled over folds; ties go to SAD, as for every
prediction at AggregateConfig.threshold. A (model, feature_set) whose
predictions do not cover all CVConfig.n_splits folds (a partial --folds run)
is not logged.

stage is smoke | pilot | full. default_stage() gives smoke on a synthetic
data root, pilot for subset "pilot", and full otherwise.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.utils.config import config
from src.utils.protocol_guard import is_synthetic_root

LOG_NAME = "RUN_LOG.csv"
STAGES = ("smoke", "pilot", "full")
ALL_PARTICIPANTS = "full"  # subset column value when no subset is given
CI_KEYS = ("macro_f1_ci_low", "macro_f1_ci_high")
METRIC_KEYS = (
    "model",
    "feature_set",
    "n_participants",
    "macro_f1",
    "accuracy",
    "majority_baseline",
)
COLUMNS = (
    "timestamp",
    "git_commit",
    "stage",
    "command",
    "analysis",
    "subset",
    "model",
    "feature_set",
    "n_participants",
    "macro_f1",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "accuracy",
    "majority_baseline",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class RunLogError(RuntimeError):
    """The run log could not be appended to safely."""


def log_path(cfg=config) -> Path:
    return cfg.paths.results_dir / LOG_NAME


def git_commit(repo_root: Path = _REPO_ROOT) -> str:
    """HEAD commit hash, or "unknown" outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def default_stage(subset: str | None, cfg=config) -> str:
    if is_synthetic_root(cfg.paths.data_root):
        return "smoke"
    return "pilot" if subset == "pilot" else "full"


def resolve_stage(stage: str | None, subset: str | None, cfg=config) -> str:
    """The given stage, or default_stage(); runners call it before any work."""
    stage = stage or default_stage(subset, cfg)
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return stage


def command_line(module: str, **flags) -> str:
    """`python -m module --flag value ...`; True adds a bare flag, None/False skip."""
    parts = ["python", "-m", module]
    for name, value in flags.items():
        if value is None or value is False:
            continue
        parts.append("--" + name.replace("_", "-"))
        if value is not True:
            parts.append(
                ",".join(map(str, value)) if isinstance(value, list) else str(value)
            )
    return " ".join(parts)


def log_run(
    stage: str, command: str, analysis: str, subset: str | None, metrics: dict
) -> None:
    """Append one row; metrics holds METRIC_KEYS and, from H1 only, CI_KEYS."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    missing = [k for k in METRIC_KEYS if k not in metrics]
    unknown = set(metrics) - set(METRIC_KEYS) - set(CI_KEYS)
    if missing or unknown:
        raise ValueError(f"metrics missing {missing}, unknown {sorted(unknown)}")
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "stage": stage,
        "command": command,
        "analysis": analysis,
        "subset": subset or ALL_PARTICIPANTS,
        **{k: metrics[k] for k in METRIC_KEYS},
        **{k: metrics.get(k) for k in CI_KEYS},
    }
    _append(log_path(), row)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    if not new:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if tuple(header) != COLUMNS:
            raise RunLogError(
                f"{path} has columns {header}, expected {list(COLUMNS)}; "
                "refusing to append (the log is never rewritten)"
            )
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: "" if row[k] is None else row[k] for k in COLUMNS})


def _is_sad(labels: pd.Series) -> np.ndarray:
    return (labels == "SAD").to_numpy()


def majority_baseline(test: pd.DataFrame, cfg=config) -> float:
    """Pooled accuracy of the training-fold majority label (ties -> SAD)."""
    hits = []
    for fold, part in test.groupby("fold"):
        train = test[test["fold"] != fold]
        sad_share = float(_is_sad(train["label"]).mean())
        hits.append(_is_sad(part["label"]) == (sad_share >= cfg.aggregate.threshold))
    return float(np.concatenate(hits).mean())


def point_metrics(predictions: pd.DataFrame, cfg=config) -> list:
    """Point metrics per complete (model, feature_set); never a CI."""
    rows = []
    folds = set(range(cfg.cv.n_splits))
    for (model, fs), part in predictions.groupby(["model", "feature_set"], sort=False):
        test = part[~part["in_middle_band"].astype(bool)]
        if set(test["fold"].astype(int)) != folds:
            print(f"run log: {model}/{fs} skipped (not all {len(folds)} folds)")
            continue
        y, p = _is_sad(test["label"]), _is_sad(test["pred"])
        rows.append(
            {
                "model": model,
                "feature_set": fs,
                "n_participants": int(test["participant_id"].nunique()),
                "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y, p)),
                "majority_baseline": majority_baseline(test, cfg),
            }
        )
    return rows


def log_predictions(
    stage: str | None,
    command: str,
    analysis: str,
    subset: str | None,
    predictions: pd.DataFrame,
    cfg=config,
) -> int:
    """Log a runner's predictions (CI columns empty); return rows written."""
    stage = resolve_stage(stage, subset, cfg)
    rows = point_metrics(predictions, cfg)
    for metrics in rows:
        log_run(stage, command, analysis, subset, metrics)
    return len(rows)
