"""Tests for src.training.evaluator on hand-made prediction files."""

import csv
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.training import evaluator
from src.training.evaluator import (
    METRICS,
    bootstrap_ci,
    evaluate,
    load_predictions,
    majority_hits,
    metrics,
    paired_diff_ci,
)
from src.training.run_baselines import PRED_COLUMNS
from src.utils import protocol_guard, run_log
from src.utils.config import config
from src.utils.protocol_guard import SYNTHETIC_MARKER, ProtocolNotFrozenError

N_FOLDS = config.cv.n_splits
THR = config.aggregate.threshold


def _frame(labels, probs, model="m", fs="all", folds=None, middle=(), **kw):
    """Prediction frame in the shared schema; pred follows the threshold."""
    n = len(labels)
    folds = folds if folds is not None else [i % N_FOLDS for i in range(n)]
    return pd.DataFrame(
        {
            "analysis": kw.get("analysis", "primary"),
            "model": model,
            "feature_set": fs,
            "fold": folds,
            "participant_id": kw.get("pids", [f"{i:03d}" for i in range(1, n + 1)]),
            "label": labels,
            "prob_sad": probs,
            "pred": ["SAD" if p >= THR else "HAPPY" for p in probs],
            "in_middle_band": [i in middle for i in range(n)],
        },
        columns=list(PRED_COLUMNS),
    )


def _majority_frame(labels, folds, **kw):
    """The majority baseline model: prob_sad = training-fold SAD share."""
    labels, folds = np.asarray(labels), np.asarray(folds)
    probs = [float(np.mean(labels[folds != f] == "SAD")) for f in folds]
    return _frame(list(labels), probs, model="majority", folds=list(folds), **kw)


# ── metrics ──────────────────────────────────────────────────────────────────


def test_metrics_hand_computed():
    # TP 2 (.9, .7), FN 2 (.3, .2), FP 1 (.8), TN 1 (.1)
    df = _frame(
        ["SAD", "SAD", "SAD", "SAD", "HAPPY", "HAPPY"], [0.9, 0.7, 0.3, 0.2, 0.8, 0.1]
    )
    m = metrics(df)
    assert m["accuracy"] == pytest.approx(3 / 6)
    assert m["precision_sad"] == pytest.approx(2 / 3)
    assert m["recall_sad"] == pytest.approx(2 / 4)
    assert m["precision_happy"] == pytest.approx(1 / 3)
    assert m["recall_happy"] == pytest.approx(1 / 2)
    # F1 SAD = 2*2/(4+1+2) = 4/7, F1 HAPPY = 2*1/(2+2+1) = 2/5
    assert m["macro_f1"] == pytest.approx((4 / 7 + 2 / 5) / 2)
    assert m["balanced_accuracy"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    # SAD > HAPPY pairs: .9>.8,.1  .7>.1  .3>.1  .2>.1 -> 5 of 8
    assert m["roc_auc"] == pytest.approx(5 / 8)
    assert set(m) == set(METRICS)


def test_single_class_auc_is_nan():
    m = metrics(_frame(["SAD", "SAD"], [0.9, 0.2]))
    assert math.isnan(m["roc_auc"]) and math.isnan(m["balanced_accuracy"])


def test_vectorized_resamples_match_sklearn():
    rng = np.random.default_rng(0)
    n = 30
    labels = ["SAD" if v else "HAPPY" for v in rng.random(n) < 0.4]
    df = evaluator._sorted(_frame(labels, rng.random(n).round(2).tolist()))
    idx = rng.integers(0, n, size=(25, n))
    fast = evaluator._resampled_metrics(df, idx)
    for r, rows in enumerate(idx):
        slow = metrics(df.iloc[rows])
        for m in METRICS:
            if math.isnan(slow[m]):
                assert math.isnan(fast[m][r])
            else:
                assert fast[m][r] == pytest.approx(slow[m]), m


def test_majority_model_macro_f1_matches_analytic_value():
    # 24 SAD / 16 HAPPY, folds balanced: every training fold is 60% SAD, so
    # the majority model predicts SAD for everyone. F1_SAD = 2s/(2s+h),
    # F1_HAPPY = 0 -> macro-F1 = s / (2s + h) = s / (n + s).
    s, h = 24, 16
    labels = ["SAD"] * s + ["HAPPY"] * h
    folds = [i % N_FOLDS for i in range(s)] + [i % N_FOLDS for i in range(h)]
    df = _majority_frame(labels, folds)
    assert set(df["pred"]) == {"SAD"}
    result = evaluate(df)
    row = result.comparison.iloc[0]
    assert row["macro_f1"] == pytest.approx(s / (s + h + s))
    assert row["accuracy"] == pytest.approx(s / (s + h))
    assert row["accuracy"] == pytest.approx(row["majority_baseline"])


def test_majority_hits_agree_with_run_log():
    rng = np.random.default_rng(3)
    labels = ["SAD" if v else "HAPPY" for v in rng.random(40) < 0.45]
    df = _frame(labels, [0.5] * 40)
    assert majority_hits(df).mean() == pytest.approx(run_log.majority_baseline(df))


# ── bootstrap ────────────────────────────────────────────────────────────────


def _noisy(n=60, seed=1, model="m", fs="all", flip=0.3):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.5
    score = np.where(y, 0.65, 0.35) + rng.normal(0, flip, n)
    probs = np.clip(score, 0, 1).round(3)
    labels = ["SAD" if v else "HAPPY" for v in y]
    return _frame(labels, probs.tolist(), model=model, fs=fs)


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate():
    df = _noisy()
    lo, hi = bootstrap_ci(df, "macro_f1", 500, 42)
    assert (lo, hi) == bootstrap_ci(
        df.sample(frac=1, random_state=0), "macro_f1", 500, 42
    )
    assert lo <= metrics(df)["macro_f1"] <= hi
    assert (lo, hi) != bootstrap_ci(df, "macro_f1", 500, 43)
    with pytest.raises(ValueError, match="metric"):
        bootstrap_ci(df, "f1", 10, 0)


def test_paired_difference():
    a = _noisy(flip=0.1)
    same = paired_diff_ci(a, a.assign(model="copy"), "macro_f1", 300, 42)
    assert same == (0.0, 0.0, 0.0)
    b = a.assign(prob_sad=0.6, pred="SAD", model="always_sad")
    mean, lo, hi = paired_diff_ci(a, b, "macro_f1", 300, 42)
    assert lo <= mean <= hi and lo > 0
    with pytest.raises(ValueError, match="participant sets differ"):
        paired_diff_ci(a, b.iloc[1:], "macro_f1", 10, 0)


# ── validation and analysis sets ─────────────────────────────────────────────


def test_validate_rejects_bad_files():
    good = _frame(["SAD", "HAPPY"], [0.7, 0.2])
    v = evaluator.validate
    with pytest.raises(ValueError, match="more than one prediction"):
        v(pd.concat([good, good]), "primary", "x")
    with pytest.raises(ValueError, match="not HAPPY/SAD"):
        v(good.assign(label=["SAD", "sad"]), "primary", "x")
    with pytest.raises(ValueError, match="pred disagrees"):
        v(good.assign(pred="SAD"), "primary", "x")
    with pytest.raises(ValueError, match="analysis"):
        v(good, "full", "x")
    with pytest.raises(ValueError, match="missing columns"):
        v(good.drop(columns=["fold"]), "primary", "x")


def test_primary_drops_middle_band_full_keeps_all():
    labels = ["SAD", "HAPPY"] * N_FOLDS + ["SAD"]
    probs = [0.9, 0.1] * N_FOLDS + [0.2]
    prim = _frame(labels, probs, middle={len(labels) - 1})
    result = evaluate(prim)
    assert result.comparison.iloc[0]["n_participants"] == 2 * N_FOLDS
    assert result.comparison.iloc[0]["macro_f1"] == 1.0
    full = prim.assign(analysis="full")
    assert evaluate(full).comparison.iloc[0]["n_participants"] == 2 * N_FOLDS + 1


def test_incomplete_folds_are_skipped():
    df = _frame(["SAD", "HAPPY"] * 4, [0.9, 0.1] * 4, folds=[0, 1] * 4)
    result = evaluate(df)
    assert result.comparison.empty and "need all" in result.skipped[0]


# ── run ──────────────────────────────────────────────────────────────────────


def _root_with_predictions(tmp_path, monkeypatch, n=40):
    root = tmp_path / "root"
    (root / "metadata").mkdir(parents=True)
    (root / SYNTHETIC_MARKER).write_text("test")
    monkeypatch.setattr(config.paths, "data_root", root)
    monkeypatch.setattr(config.paths, "output_root", root)
    monkeypatch.setattr(config.evaluation, "n_bootstrap", 200)

    rng = np.random.default_rng(7)
    y = rng.random(n + 4) < 0.5
    labels = ["SAD" if v else "HAPPY" for v in y]
    folds = [i % N_FOLDS for i in range(n + 4)]
    middle = set(range(n, n + 4))

    def probs(noise, seed):
        r = np.random.default_rng(seed)
        return np.clip(np.where(y, 0.6, 0.4) + r.normal(0, noise, n + 4), 0, 1).round(3)

    def model(name, fs, noise, seed, analysis="primary"):
        return _frame(
            labels,
            probs(noise, seed).tolist(),
            name,
            fs,
            folds,
            middle,
            analysis=analysis,
        )

    hand = model("lr_handcrafted", "all", 0.2, 1)
    emb = model("lr_embeddings", "concat", 0.3, 2)
    hmm = model("cnn_hmm_fused", "fused", 0.3, 3)
    ens_prob = (hand["prob_sad"] + emb["prob_sad"] + hmm["prob_sad"]) / 3
    ens = hand.assign(
        model="ensemble",
        prob_sad=ens_prob,
        pred=np.where(ens_prob >= THR, "SAD", "HAPPY"),
    )
    maj = _majority_frame(labels, folds, middle=middle)
    full_hand = model("lr_handcrafted", "all", 0.2, 4, "full").assign(
        in_middle_band=False
    )
    full_maj = _majority_frame(labels, folds, analysis="full")
    files = {
        "baselines/primary": pd.concat([maj, hand, emb]),
        "hybrid/primary": hmm,
        "ensemble/primary": ens,
        "baselines/full": pd.concat([full_maj, full_hand]),
    }
    for sub, df in files.items():
        d = root / "results" / sub
        d.mkdir(parents=True)
        df.to_csv(d / "predictions.csv", index=False)
    return root


def test_run_writes_outputs_and_fills_run_log_ci(tmp_path, monkeypatch):
    root = _root_with_predictions(tmp_path, monkeypatch)
    path = evaluator.run(stage="smoke")
    out = root / "results" / "final"
    assert path == out / "model_comparison.csv"
    comp = pd.read_csv(path)
    assert list(comp["analysis"].unique()) == ["primary", "full"]
    prim = comp[comp["analysis"] == "primary"].set_index("model")
    assert set(prim.index) == {
        "majority",
        "lr_handcrafted",
        "lr_embeddings",
        "cnn_hmm_fused",
        "ensemble",
    }
    assert (prim["n_participants"] == 40).all()  # middle band excluded
    assert math.isnan(prim.loc["lr_handcrafted", "delta_macro_f1_vs_reference"])
    assert prim["delta_macro_f1_ci_low"].drop("lr_handcrafted").notna().all()
    for name in (
        "model_comparison.txt",
        "per_fold_metrics.csv",
        "confusion_matrices.csv",
        "confusion_matrices_primary.png",
        "roc_curves_primary.png",
        "roc_curves_full.png",
        "reliability_ensemble_primary.png",
    ):
        assert (out / name).is_file(), name
    assert not (out / "reliability_ensemble_full.png").exists()
    per_fold = pd.read_csv(out / "per_fold_metrics.csv")
    assert len(per_fold) == len(comp) * N_FOLDS
    text = (out / "model_comparison.txt").read_text(encoding="utf-8")
    assert "majority 0." in text and "cnn" in text  # missing cnn files listed

    with (root / "results" / "RUN_LOG.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(comp)
    assert all(r["macro_f1_ci_low"] and r["macro_f1_ci_high"] for r in rows)
    assert all(
        r["command"].startswith("python -m src.training.evaluator") for r in rows
    )
    # the logged macro-F1 is the pooled sklearn value
    hand = pd.read_csv(root / "results" / "baselines" / "primary" / "predictions.csv")
    hand = hand[(hand["model"] == "lr_handcrafted") & ~hand["in_middle_band"]]
    expected = f1_score(hand["label"], hand["pred"], average="macro")
    assert prim.loc["lr_handcrafted", "macro_f1"] == pytest.approx(expected, abs=1e-6)


def test_run_is_identical_across_two_runs(tmp_path, monkeypatch):
    root = _root_with_predictions(tmp_path, monkeypatch)
    out = root / "results" / "final"
    evaluator.run(stage="smoke")
    first = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    evaluator.run(stage="smoke")
    second = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    assert first.keys() == second.keys() and len(first) >= 8
    for name in first:
        assert first[name] == second[name], name


def test_load_predictions_rejects_label_disagreement(tmp_path, monkeypatch):
    root = _root_with_predictions(tmp_path, monkeypatch)
    path = root / "results" / "hybrid" / "primary" / "predictions.csv"
    df = pd.read_csv(path, dtype={"participant_id": str})
    df.loc[0, "label"] = "HAPPY" if df.loc[0, "label"] == "SAD" else "SAD"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree on the label"):
        load_predictions()


def test_refuses_real_root_before_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    monkeypatch.setattr(config.paths, "output_root", tmp_path)
    monkeypatch.setattr(protocol_guard, "protocol_frozen", lambda: False)
    with pytest.raises(ProtocolNotFrozenError, match="evaluator"):
        evaluator.run(stage="smoke")


def test_cli_reports_missing_predictions(tmp_path, monkeypatch, capsys):
    root = tmp_path / "empty"
    root.mkdir()
    (root / SYNTHETIC_MARKER).write_text("test")
    monkeypatch.setattr(config.paths, "data_root", root)
    monkeypatch.setattr(config.paths, "output_root", root)
    monkeypatch.setattr("sys.argv", ["evaluator", "--stage", "smoke"])
    assert evaluator.main() == 1
    assert "no predictions.csv" in capsys.readouterr().out
