"""Tests for src.training.run_ensemble with hand-made component files."""

import pandas as pd
import pytest

from src.training import run_ensemble
from src.training.run_baselines import PRED_COLUMNS
from src.training.run_ensemble import combine
from src.utils.protocol_guard import SYNTHETIC_MARKER, ProtocolNotFrozenError

PIDS = ["001", "002", "003"]
LABELS = {"001": "SAD", "002": "HAPPY", "003": "SAD"}
PROBS = {
    ("lr_handcrafted", "all"): [0.9, 0.2, 0.3],
    ("lr_embeddings", "concat"): [0.6, 0.4, 0.5],
    ("cnn_hmm_fused", "fused"): [0.3, 0.3, 0.4],
}


def _frame(model, fs, probs, pids=PIDS, middle=(), analysis="primary"):
    rows = [
        {
            "analysis": analysis,
            "model": model,
            "feature_set": fs,
            "fold": 0 if pid in ("001", "002") else 1,
            "participant_id": pid,
            "label": LABELS.get(pid, "HAPPY"),
            "prob_sad": p,
            "pred": "SAD" if p >= 0.5 else "HAPPY",
            "in_middle_band": pid in middle,
        }
        for pid, p in zip(pids, probs)
    ]
    return pd.DataFrame(rows, columns=list(PRED_COLUMNS))


def _frames():
    return [_frame(m, f, p) for (m, f), p in PROBS.items()]


def _write_root(tmp_path, monkeypatch, frames):
    root = tmp_path / "root"
    (root / "metadata").mkdir(parents=True)
    (root / SYNTHETIC_MARKER).write_text("test")
    base = pd.concat(frames[:2] + [_frame("lr_handwriting", "concat", [0.99] * 3)])
    hybrid = pd.concat(
        [frames[2], _frame("cnn_hmm", "word", [0.1] * 3)], ignore_index=True
    )
    for sub, df in (("baselines", base), ("hybrid", hybrid)):
        d = root / "results" / sub / "primary"
        d.mkdir(parents=True)
        df.to_csv(d / "predictions.csv", index=False)
    monkeypatch.setattr(run_ensemble.config.paths, "data_root", root)
    monkeypatch.setattr(run_ensemble.config.paths, "output_root", root)
    return root


def test_three_files_give_expected_averages(tmp_path, monkeypatch):
    root = _write_root(tmp_path, monkeypatch, _frames())
    path = run_ensemble.run("primary")
    assert path == root / "results" / "ensemble" / "primary" / "predictions.csv"
    raw = pd.read_csv(path, dtype={"participant_id": str})
    assert tuple(raw.columns) == PRED_COLUMNS
    out = raw.set_index("participant_id")
    expected = {"001": (0.9 + 0.6 + 0.3) / 3, "002": 0.3, "003": 0.4}
    for pid, p in expected.items():
        assert out.loc[pid, "prob_sad"] == pytest.approx(p)
    assert out.loc["001", "pred"] == "SAD" and out.loc["002", "pred"] == "HAPPY"
    assert set(out["model"]) == {"ensemble"} and set(out["feature_set"]) == {"all"}
    assert out.loc["003", "label"] == "SAD" and out.loc["003", "fold"] == 1


def test_appends_one_run_log_row_with_empty_ci(tmp_path, monkeypatch):
    n = run_ensemble.config.cv.n_splits
    pids = [f"{i:03d}" for i in range(1, 2 * n + 1)]
    frames = [
        _frame(
            m, f, [0.6 if i % 2 else 0.4 for i in range(len(pids))], pids=pids
        ).assign(fold=[i % n for i in range(len(pids))])
        for m, f in PROBS
    ]
    root = _write_root(tmp_path, monkeypatch, frames)
    run_ensemble.run("primary")
    log = pd.read_csv(root / "results" / "RUN_LOG.csv", dtype=str)
    assert len(log) == 1
    row = log.iloc[0]
    assert (row["model"], row["feature_set"], row["stage"]) == (
        "ensemble",
        "all",
        "smoke",
    )
    assert row["command"] == (
        "python -m src.training.run_ensemble --analysis primary --stage smoke"
    )
    assert int(row["n_participants"]) == len(pids)
    assert pd.isna(row["macro_f1_ci_low"]) and pd.isna(row["macro_f1_ci_high"])


def test_partial_folds_not_logged(tmp_path, monkeypatch):
    root = _write_root(tmp_path, monkeypatch, _frames())  # folds 0 and 1 only
    run_ensemble.run("primary")
    assert not (root / "results" / "RUN_LOG.csv").exists()


def test_exploratory_and_other_rows_ignored(tmp_path, monkeypatch):
    _write_root(tmp_path, monkeypatch, _frames())
    out = pd.read_csv(run_ensemble.run("primary"))
    assert out["prob_sad"].max() < 0.99  # lr_handwriting (0.99) never averaged


def test_threshold_exactly_half_is_sad():
    frames = [_frame(m, f, [0.5, 0.5, 0.5]) for m, f in PROBS]
    assert set(combine(frames)["pred"]) == {"SAD"}


def test_missing_participant_raises(tmp_path, monkeypatch):
    frames = _frames()
    frames[1] = frames[1][frames[1]["participant_id"] != "002"]
    _write_root(tmp_path, monkeypatch, frames)
    with pytest.raises(ValueError, match="misses 1 test-fold participant"):
        run_ensemble.run("primary")


def test_fold_mismatch_counts_as_missing():
    frames = _frames()
    frames[2] = frames[2].assign(
        fold=frames[2]["fold"].where(frames[2]["participant_id"] != "001", 1)
    )
    with pytest.raises(ValueError, match="misses"):
        combine(frames)


def test_middle_band_rows_only_when_all_components_have_them():
    with_mid = [
        _frame(m, f, p + [0.7], pids=PIDS + ["004"], middle=("004",))
        for (m, f), p in PROBS.items()
    ]
    out = combine(with_mid)
    assert out.loc[out["participant_id"] == "004", "in_middle_band"].tolist() == [True]
    partial = with_mid[:2] + [_frames()[2]]  # hybrid run without --predict-middle
    out = combine(partial)
    assert not out["in_middle_band"].any() and len(out) == 3


def test_components_are_fixed():
    frames = _frames()
    with pytest.raises(ValueError, match="components are fixed"):
        combine(frames[:2])
    with pytest.raises(ValueError, match="exploratory"):
        combine(frames[:2] + [_frame("lr_handwriting", "concat", [0.5] * 3)])
    with pytest.raises(ValueError, match="components are fixed"):
        combine(frames[:2] + [_frame("cnn_hmm", "word", [0.5] * 3)])
    with pytest.raises(ValueError, match="given twice"):
        combine(frames + [frames[0]])


def test_label_disagreement_raises():
    frames = _frames()
    frames[0] = frames[0].assign(label="HAPPY")
    with pytest.raises(ValueError, match="disagree on label"):
        combine(frames)


def test_missing_component_file_and_guard(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "metadata").mkdir(parents=True)
    (root / SYNTHETIC_MARKER).write_text("test")
    monkeypatch.setattr(run_ensemble.config.paths, "data_root", root)
    monkeypatch.setattr(run_ensemble.config.paths, "output_root", root)
    with pytest.raises(FileNotFoundError, match="baselines"):
        run_ensemble.run("primary")
    (root / SYNTHETIC_MARKER).unlink()
    monkeypatch.setattr(
        "src.utils.protocol_guard.protocol_frozen", lambda *a, **k: False
    )
    with pytest.raises(ProtocolNotFrozenError):
        run_ensemble.run("primary")
