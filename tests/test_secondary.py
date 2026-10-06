"""Tests for src.analysis.secondary on a hand-made synthetic prediction root."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from src.analysis import secondary
from src.training import evaluator, run_cnn, run_ensemble, run_hybrid
from src.training.run_baselines import PRED_COLUMNS
from src.utils.config import config
from src.utils.protocol_guard import SYNTHETIC_MARKER

N_FOLDS = config.cv.n_splits
THR = config.aggregate.threshold
N, N_MID, N_FAILED = 50, 6, 4
CROPS = {"drawing": 4, "word": 3, "cursive": 2}


def _pids(n, start=1):
    return [f"{i:03d}" for i in range(start, start + n)]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    rng = np.random.default_rng(11)
    root = tmp_path_factory.mktemp("secondary") / "root"
    (root / "metadata").mkdir(parents=True)
    (root / SYNTHETIC_MARKER).write_text("test")

    prim, mid, failed = _pids(N), _pids(N_MID, N + 1), _pids(N_FAILED, N + N_MID + 1)
    everyone = prim + mid + failed
    y = {p: ("SAD" if v else "HAPPY") for p, v in zip(everyone, rng.random(99) < 0.5)}
    fold = {p: i % N_FOLDS for i, p in enumerate(prim + mid)}
    pd.DataFrame(
        {"participant_id": everyone, "label": [y[p] for p in everyone]}
    ).to_csv(root / "metadata" / "labels.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": everyone,
            "status": ["qc_passed"] * (N + N_MID) + ["qc_failed"] * N_FAILED,
        }
    ).to_csv(root / "metadata" / "participants.csv", index=False)

    def frame(model, fs, noise, ids=prim, analysis="primary", middle=()):
        signal = np.array([0.62 if y[p] == "SAD" else 0.38 for p in ids])
        prob = np.clip(signal + rng.normal(0, noise, len(ids)), 0, 1).round(4)
        return pd.DataFrame(
            {
                "analysis": analysis,
                "model": model,
                "feature_set": fs,
                "fold": [fold[p] for p in ids],
                "participant_id": ids,
                "label": [y[p] for p in ids],
                "prob_sad": prob,
                "pred": np.where(prob >= THR, "SAD", "HAPPY"),
                "in_middle_band": [p in middle for p in ids],
            }
        )[list(PRED_COLUMNS)]

    def crops(model, families, noise, ids=prim):
        rows = []
        for p in ids:
            for fam in families:
                for c in range(CROPS[fam]):
                    base = 0.6 if y[p] == "SAD" else 0.4
                    rows.append(
                        {
                            "analysis": "primary",
                            "model": model,
                            "fold": fold[p],
                            "participant_id": p,
                            "task_family": fam,
                            "cell": f"{fam}{c}",
                            "label": y[p],
                            "prob_sad": float(
                                np.clip(base + rng.normal(0, noise), 0, 1).round(4)
                            ),
                            "in_middle_band": False,
                        }
                    )
        return pd.DataFrame(rows)

    def write(sub, df, name="predictions.csv"):
        d = root / "results" / sub
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / name, index=False)

    hand_all = pd.concat(
        [frame("lr_handcrafted", "all", 0.2, prim + mid, middle=set(mid))]
    )
    lr = [hand_all, frame("majority", "all", 0.0).assign(prob_sad=0.5, pred="SAD")]
    lr += [frame("lr_handcrafted", f, 0.3) for f in CROPS]
    lr += [frame("lr_embeddings", f, 0.3) for f in list(CROPS) + ["concat"]]
    lr.append(frame("lr_handwriting", "concat", 0.25))
    write("baselines/primary", pd.concat(lr))
    coefs = pd.DataFrame(
        [
            {
                "analysis": "primary",
                "model": "lr_handcrafted",
                "feature_set": "all",
                "fold": k,
                "feature": f"f{j:02d}",
                "coef_std": (j - 7) * 0.1 + 0.01 * k,
                "C": 1.0,
            }
            for k in range(N_FOLDS)
            for j in range(15)
        ]
    )
    write("baselines/primary", coefs, "lr_coefficients.csv")
    write(
        "baselines/full",
        pd.concat(
            [
                frame("lr_handcrafted", "all", 0.2, prim + mid, "full"),
                frame("majority", "all", 0.0, prim + mid, "full").assign(
                    prob_sad=0.5, pred="SAD"
                ),
            ]
        ),
    )
    for sub, noise in (("cnn/primary", 0.25), ("cnn/primary_simple", 0.35)):
        c = crops("cnn_head", CROPS, noise)
        write(sub, c, "crop_predictions.csv")
        write(sub, run_cnn._participant_predictions(c, "primary"))
    h = crops("cnn_hmm", ("word", "cursive"), 0.25)
    write("hybrid/primary", h, "crop_predictions.csv")
    hybrid = run_hybrid._predictions(h, "primary")
    write("hybrid/primary", hybrid)
    base = pd.concat(lr)
    parts = [
        base[(base["model"] == "lr_handcrafted") & (base["feature_set"] == "all")],
        base[(base["model"] == "lr_embeddings") & (base["feature_set"] == "concat")],
        hybrid[hybrid["model"] == "cnn_hmm_fused"],
    ]
    write("ensemble/primary", run_ensemble.combine(parts))
    return root


@pytest.fixture
def at_root(root, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", root)
    monkeypatch.setattr(config.paths, "output_root", root)
    monkeypatch.setattr(config.labeling, "output_csv", None)
    monkeypatch.setattr(config.evaluation, "n_bootstrap", 200)
    return root


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── acceptance ────────────────────────────────────────────────────────────────


def test_runs_and_never_writes_model_comparison(at_root, monkeypatch):
    evaluator.run(stage="smoke")
    out = at_root / "results" / "final"
    guarded = sorted(out.glob("model_comparison*"))
    assert len(guarded) == 2
    before = {p.name: (_digest(p), p.stat().st_mtime_ns) for p in guarded}

    titles = []
    real_save = secondary._save

    def spy(fig, path, title, cfg=config):
        titles.append((path.name, title))
        real_save(fig, path, title, cfg)

    monkeypatch.setattr(secondary, "_save", spy)
    path = secondary.run()

    assert {p.name: (_digest(p), p.stat().st_mtime_ns) for p in guarded} == before
    assert sorted(out.glob("model_comparison*")) == guarded
    assert path == out / "secondary.txt"
    text = path.read_text(encoding="utf-8")
    for heading in ("1. Macro-F1 per task family", "2a.", "2b.", "3. Ablations"):
        assert heading in text
    assert "EXPLORATORY" in text and "lr_handwriting/concat" in text
    names = {name for name, _ in titles}
    assert {
        "secondary_family_primary.png",
        "secondary_ablations_primary.png",
        "secondary_top_features.png",
    } <= names
    assert all(t.startswith(("Secondary analysis", "Exploratory")) for _, t in titles)
    for name in names:
        assert (out / name).is_file()


# ── sections ─────────────────────────────────────────────────────────────────


def _rows(at_root):
    secondary.run()
    return pd.read_csv(at_root / "results" / "final" / "secondary_metrics.csv")


def test_family_table_covers_every_family_and_combined(at_root):
    rows = _rows(at_root)
    fam = rows[(rows["block"] == "family") & (rows["analysis"] == "primary")]
    resnet = fam[fam["row_label"] == "cnn_head (ResNet18)"]
    assert set(resnet["variant"]) == {"drawing", "word", "cursive", "combined"}
    hmm = fam[fam["row_label"] == "cnn_hmm"]
    assert set(hmm["variant"]) == {"word", "cursive", "combined"}
    simple = fam[fam["row_label"] == "cnn_head (simple CNN)"]
    assert set(simple["model"]) == {"cnn_head_simple", "cnn_head_simple_fused"}
    assert (fam["macro_f1_ci_low"] <= fam["macro_f1"]).all()
    assert (fam["macro_f1"] <= fam["macro_f1_ci_high"]).all()


def test_middle_band_only_for_predict_middle_models(at_root):
    rows = _rows(at_root)
    mid = rows[rows["block"] == "middle_band"]
    assert list(zip(mid["model"], mid["feature_set"])) == [("lr_handcrafted", "all")]
    assert mid.iloc[0]["n_participants"] == N_MID
    # primary-band rows are never in the middle-band block
    prim = evaluator.load_predictions("primary")
    m = prim[prim["in_middle_band"]]
    assert mid.iloc[0]["accuracy"] == pytest.approx(evaluator.metrics(m)["accuracy"])


def test_ablations_are_paired_against_named_references(at_root):
    rows = _rows(at_root)
    ens = rows[rows["block"] == "ensemble"]
    assert set(ens["model"]) == {"lr_handcrafted", "lr_embeddings", "cnn_hmm_fused"}
    assert set(ens["reference"]) == {"ensemble/all"}
    prim = evaluator.analysis_frame(evaluator.load_predictions("primary"), "primary")

    def group(m, f):
        return prim[(prim["model"] == m) & (prim["feature_set"] == f)]

    d, lo, hi = evaluator.paired_diff_ci(
        group("lr_handcrafted", "all"), group("ensemble", "all"), "macro_f1", 200, 42
    )
    r = ens[ens["model"] == "lr_handcrafted"].iloc[0]
    assert (r["delta_macro_f1"], r["delta_ci_low"], r["delta_ci_high"]) == (
        pytest.approx(d, abs=1e-6),
        pytest.approx(lo, abs=1e-6),
        pytest.approx(hi, abs=1e-6),
    )
    agg = rows[rows["block"] == "aggregation"]
    assert set(agg["variant"]) == {"mean_logit", "majority_vote"}
    assert {"cnn_head_fused", "cnn_hmm_fused"} <= set(agg["model"])
    emb = rows[rows["block"] == "embeddings"]
    assert emb["exploratory"].all() and set(emb["reference"]) == {
        "lr_embeddings/concat"
    }


def test_mean_prob_from_crops_matches_the_runner(at_root):
    crops = secondary._crops("cnn", "primary")
    mine = secondary.aggregate_with(
        crops, "mean_prob", "cnn_head", "cnn_head_fused", tuple(CROPS)
    )
    stored = pd.read_csv(
        at_root / "results" / "cnn" / "primary" / "predictions.csv",
        dtype={"participant_id": str},
    )
    key = ["model", "feature_set", "participant_id"]
    merged = mine.merge(stored, on=key, suffixes=("", "_stored"))
    assert len(merged) == len(stored)
    np.testing.assert_allclose(merged["prob_sad"], merged["prob_sad_stored"])


def test_majority_vote_aggregation_by_hand():
    crops = pd.DataFrame(
        {
            "analysis": "primary",
            "model": "cnn_head",
            "fold": 0,
            "participant_id": ["001"] * 5,
            "task_family": ["word"] * 3 + ["cursive"] * 2,
            "label": "SAD",
            "prob_sad": [0.9, 0.6, 0.2, 0.4, 0.3],
            "in_middle_band": False,
        }
    )
    out = secondary.aggregate_with(
        crops, "majority_vote", "cnn_head", "fused_m", ("word", "cursive")
    ).set_index("feature_set")
    assert out.loc["word", "prob_sad"] == pytest.approx(2 / 3)
    assert out.loc["cursive", "prob_sad"] == pytest.approx(0.0)
    assert out.loc["fused", "prob_sad"] == pytest.approx(1 / 3)  # equal weights
    assert out.loc["fused", "pred"] == "HAPPY"


def test_top_features_ranked_by_absolute_mean(at_root):
    top = secondary.top_features()
    assert len(top) == config.evaluation.top_features
    # coef = (j - 7) * 0.1 + 0.01 * fold -> mean (j - 7) * 0.1 + 0.02
    assert top.iloc[0]["feature"] == "f14"
    assert top.iloc[0]["mean"] == pytest.approx(0.72)
    assert top.iloc[1]["feature"] == "f00"
    assert top.iloc[1]["mean"] == pytest.approx(-0.68)
    assert top["std"].iloc[0] == pytest.approx(
        np.std([0, 0.01, 0.02, 0.03, 0.04], ddof=1)
    )
    assert (top["n_folds"] == N_FOLDS).all()


def test_qc_balance_chi_square(at_root):
    qc = secondary.qc_balance()
    table = qc["table"]
    assert table.loc["excluded"].sum() == N_FAILED
    assert table.loc["qc_passed"].sum() == N + N_MID
    chi2, p, _, _ = chi2_contingency(table.to_numpy())
    assert qc["chi2"] == pytest.approx(chi2) and qc["p"] == pytest.approx(p)
    assert qc["statuses"] == {"qc_failed": N_FAILED}


def test_qc_balance_none_when_nobody_excluded(at_root, tmp_path, monkeypatch):
    meta = tmp_path / "metadata"
    meta.mkdir()
    pd.DataFrame({"participant_id": ["001", "002"], "label": ["SAD", "HAPPY"]}).to_csv(
        meta / "labels.csv", index=False
    )
    pd.DataFrame({"participant_id": ["001", "002"], "status": "qc_passed"}).to_csv(
        meta / "participants.csv", index=False
    )
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    assert secondary.qc_balance() is None
