"""Tests for v5 signal boost (exploratory, Deviation 8). Synthetic data only."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import embeddings_v5
from src.features.embeddings import FAMILIES, load_backbone
from src.training import evaluator, run_baselines
from src.training import run_signal_v5 as v5
from src.utils import synthetic
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

N_PARTICIPANTS = 80
N_DRAWS = 3
STYLE_DIM, DINO_DIM = 896, 768
SMALL_CANVAS = {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)}
FAST = {
    ("baselines", "C_grid"): (1.0,),
    ("baselines", "inner_cv"): 2,
    ("baselines", "pca_components"): 4,
    ("signal_v5", "n_permutations"): N_DRAWS,
    ("signal_v5", "inner_cv"): 3,
    ("signal_v5", "ridge_alphas"): (1.0, 100.0),
    ("evaluation", "n_bootstrap"): 50,
}


def _fast(mp):
    for (section, key), value in FAST.items():
        mp.setattr(getattr(config, section), key, value)


def _write_v5_blocks(root, signal=True, seed=3):
    """Stand-in style/dinov2 caches with the synthetic layout."""
    labels = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    rng = np.random.default_rng(seed)
    emb_dir = root / "results" / "embeddings"
    for block, dim in (("style", STYLE_DIM), ("dinov2", DINO_DIM)):
        for family in FAMILIES:
            arrays = synthetic._embeddings(labels, rng, family, dim, signal, 3, 1.0)
            np.savez_compressed(emb_dir / f"{block}_{family}.npz", **arrays)


def _write_references(root):
    run_baselines.run("primary")
    base = pd.read_csv(
        root / "results" / "baselines" / "primary" / "predictions.csv",
        dtype={"participant_id": str},
    )
    rows = base[(base["model"] == "lr_handcrafted") & (base["feature_set"] == "all")]
    hybrid = root / "results" / "hybrid" / "primary" / "predictions.csv"
    hybrid.parent.mkdir(parents=True, exist_ok=True)
    rows.assign(model="cnn_hmm_fused", feature_set="fused").to_csv(hybrid, index=False)


def _new_root(path, signal=True):
    root = make_synthetic_root(path, n_participants=N_PARTICIPANTS, signal=signal)
    _write_v5_blocks(root, signal=signal)
    return root


def _hashes(root) -> dict:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def v5_root(tmp_path, monkeypatch, use_root):
    _fast(monkeypatch)
    return use_root(_new_root(tmp_path / "root"))


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    """Every v5 step run once on a root holding v1 stand-ins."""
    with pytest.MonkeyPatch.context() as mp:
        _fast(mp)
        root = _new_root(tmp_path_factory.mktemp("v5") / "root")
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        _write_references(root)
        before = _hashes(root)
        summary = v5.preflight(config)
        v5.step_fit(config)
        v5.step_evaluate(config)
        v5.step_curve(config)
        v5.step_permutation(config)
        v5.step_report(config)
        yield root, before, summary


def _read(path):
    return pd.read_csv(path, dtype={"participant_id": str})


# ---------------------------------------------------------------- config


def test_v5_settings_leave_the_run_fingerprint_unchanged(monkeypatch):
    from src.pipeline_runner import config_fingerprint

    before = config_fingerprint()
    monkeypatch.setattr(config.signal_v5, "strict_quantile", 0.3)
    assert config_fingerprint() == before


@pytest.mark.parametrize(
    "key, value",
    [
        ("headline", ("ridge_v5", "nope")),
        ("headline", ("svm_v5", "all_v5")),
        ("strict_quantile", 0.5),
        ("extremity_distances", (3.0, 0.0)),
        ("feature_sets", (("a", ("resnet18",)), ("a", ("style",)))),
    ],
)
def test_check_config_rejects_bad_settings(monkeypatch, key, value):
    monkeypatch.setattr(config.signal_v5, key, value)
    with pytest.raises(ValueError):
        v5.check_config(config)


# ---------------------------------------------------------------- groups, folds


def test_strict_groups_come_from_scores_only(v5_root):
    ctx = v5.context(config)
    low, high = v5.strict_groups(ctx, config)
    scores = ctx.labels.loc[ctx.full, "total_score"]
    q = config.signal_v5.strict_quantile
    assert set(low) == set(scores.index[scores <= scores.quantile(q, "lower")])
    assert set(high) == set(scores.index[scores >= scores.quantile(1 - q, "higher")])
    assert (ctx.labels.loc[low, "label"] == "SAD").all()
    assert (ctx.labels.loc[high, "label"] == "HAPPY").all()
    assert not (set(low) | set(high)) & ctx.middle()


def test_fold_sets_never_overlap_and_cover_everyone(v5_root):
    ctx = v5.context(config)
    seen = []
    for fold in range(config.cv.n_splits):
        s = v5.fold_sets(fold, ctx)
        assert not set(s["train_full"]) & set(s["test_full"])
        assert not set(s["train_primary"]) & ctx.middle()
        assert set(s["train_primary"]) <= set(s["train_full"])
        seen += s["test_full"]
    assert sorted(seen) == sorted(ctx.full)


# ---------------------------------------------------------------- leakage


def test_ridge_training_side_ignores_test_features(v5_root):
    ctx = v5.context(config)
    X = v5.feature_table("all_v5", ctx, config)
    sets = v5.fold_sets(1, ctx)
    test = sets["test_full"]
    noisy = X.copy()
    noisy.loc[test] = np.random.default_rng(5).normal(size=(len(test), X.shape[1])) * 50
    a = v5.ridge_fold(X, sets, ctx, 1, config)
    b = v5.ridge_fold(noisy, sets, ctx, 1, config)
    assert np.allclose(a.Z_train, b.Z_train)
    for (za_tr, za_va), (zb_tr, zb_va) in zip(a.inner, b.inner):
        assert np.allclose(za_tr, zb_tr) and np.allclose(za_va, zb_va)
    score = ctx.column("total_score", a.train_ids).astype(float)
    sad = ctx.is_sad(a.train_ids, config)
    fa, fb = v5.fit_ridge(a, score, sad, config), v5.fit_ridge(b, score, sad, config)
    assert fa["alpha"] == fb["alpha"] and fa["inner_mse"] == fb["inner_mse"]
    assert not np.allclose(fa["prob"], fb["prob"])


def test_lr_training_side_ignores_test_features(v5_root):
    ctx = v5.context(config)
    X = v5.feature_table("all_v5", ctx, config)
    sets = v5.fold_sets(2, ctx)
    test = sets["test_full"]
    noisy = X.copy()
    noisy.loc[test] = np.random.default_rng(6).normal(size=(len(test), X.shape[1])) * 50
    y = ctx.is_sad(sets["train_primary"], config).astype(int)
    pa = run_baselines.fit_lr(X.loc[sets["train_primary"]], y, config, pca=True, seed=2)
    pb = run_baselines.fit_lr(
        noisy.loc[sets["train_primary"]], y, config, pca=True, seed=2
    )
    assert np.allclose(pa.named_steps["lr"].coef_, pb.named_steps["lr"].coef_)


def test_cached_projection_matches_a_plain_pipeline(v5_root):
    ctx = v5.context(config)
    X = v5.feature_table("all_v5", ctx, config)
    sets = v5.fold_sets(0, ctx)
    rf = v5.ridge_fold(X, sets, ctx, 0, config)
    score = ctx.column("total_score", rf.train_ids).astype(float)
    fit = v5.fit_ridge(rf, score, ctx.is_sad(rf.train_ids, config), config)
    pipe = Pipeline(
        [
            ("s", StandardScaler()),
            ("p", PCA(n_components=rf.Z_train.shape[1], svd_solver="full")),
            ("r", Ridge(alpha=fit["alpha"])),
        ]
    ).fit(X.loc[rf.train_ids].to_numpy(), score)
    assert np.allclose(pipe.predict(X.loc[rf.test_ids].to_numpy()), fit["test_score"])


def test_calibrator_uses_only_training_participants(v5_root):
    ctx = v5.context(config)
    X = v5.feature_table("all_v5", ctx, config)
    sets = v5.fold_sets(3, ctx)
    rf = v5.ridge_fold(X, sets, ctx, 3, config)
    covered = np.concatenate([va for _, va in rf.splits])
    assert sorted(covered) == list(range(len(rf.train_ids)))
    assert not set(rf.train_ids) & set(rf.test_ids)


def test_permutation_shuffles_pairs_and_keeps_test_labels(v5_root):
    ctx = v5.context(config)
    folds = v5.headline_folds(ctx, config)
    rf = folds[0]
    score, sad = v5.shuffled_targets(rf, ctx, 0, 0, config)
    true_pairs = sorted(
        zip(
            ctx.column("total_score", rf.train_ids),
            ctx.is_sad(rf.train_ids, config),
        )
    )
    assert sorted(zip(score, sad)) == true_pairs
    assert not np.array_equal(score, ctx.column("total_score", rf.train_ids))
    rows = v5.draw(0, folds, ctx, config)
    truth = ctx.labels.loc[rows["participant_id"], "label"].to_numpy()
    assert (rows["label"].to_numpy() == truth).all()


def test_holm_matches_hand_computation():
    adj = v5.holm({"H1": 0.03, "H2": 0.01})
    assert adj["H2"] == pytest.approx(0.02)
    assert adj["H1"] == pytest.approx(0.03)
    assert v5.holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


# ---------------------------------------------------------------- full run


def test_preflight_reports_sets(run_root):
    _, _, summary = run_root
    assert summary["n_full"] == N_PARTICIPANTS
    assert summary["n_primary"] + summary["n_middle_band"] == N_PARTICIPANTS
    assert summary["strict_low_sad"] > 0 and summary["strict_high_happy"] > 0


def test_predictions_cover_every_participant_once(run_root):
    root, _, _ = run_root
    preds = _read(v5.output_dir(config) / "predictions.csv")
    labels = _read(root / "metadata" / "labels.csv").set_index("participant_id")
    groups = preds.groupby(["model", "feature_set"])
    assert len(groups) == len(config.signal_v5.models) * len(
        config.signal_v5.feature_sets
    )
    for _, g in groups:
        assert sorted(g["participant_id"]) == sorted(labels.index)
        assert set(g["fold"]) == set(range(config.cv.n_splits))
        mid = labels.loc[g["participant_id"], "in_middle_band"].to_numpy()
        assert (g["in_middle_band"].to_numpy() == mid).all()


def test_headlines_and_report(run_root):
    metrics = pd.read_csv(v5.output_dir(config) / "metrics.csv").set_index(
        ["model", "feature_set"]
    )
    for key in v5.headlines(config).values():
        assert metrics.loc[key, "macro_f1"] > 0.6
    h1 = metrics.loc[v5.headlines(config)["H1"]]
    h2 = metrics.loc[v5.headlines(config)["H2"]]
    assert h2["n_participants"] < h1["n_participants"]
    text = v5.report_path(config).read_text(encoding="utf-8").splitlines()
    assert text[1] == v5.REPORT_HEADER
    assert any("Pre-declared headline H2" in line for line in text)
    perm = json.loads((v5.output_dir(config) / "permutation.json").read_text())
    assert perm["n"] == N_DRAWS
    for tag in ("H1", "H2"):
        assert 0 < perm[tag]["p_macro_f1"] <= 1
        assert perm[tag]["p_macro_f1_holm"] >= perm[tag]["p_macro_f1"]


def test_curve_distances_and_sizes(run_root):
    curve = pd.read_csv(v5.output_dir(config) / "extremity_curve.csv")
    for _, part in curve.groupby(["model", "feature_set"]):
        assert list(part["d"]) == list(config.signal_v5.extremity_distances)
        assert part["n_participants"].is_monotonic_decreasing
    first = curve.iloc[0]
    assert first["n_participants"] == N_PARTICIPANTS


def test_v5_writes_only_new_files_under_v5_paths(run_root):
    root, before, _ = run_root
    after = _hashes(root)
    for path, digest in before.items():
        assert after[path] == digest, f"v5 changed {path}"
    allowed = (
        "results/signal_v5/",
        "results/final/permutation_runs/primary_v5/",
        "results/final/EXPLORATORY_v5.txt",
        "results/final/fig_v5_",
    )
    new = set(after) - set(before)
    assert new and all(p.startswith(allowed) for p in new), sorted(new)


def test_rerun_with_other_settings_is_refused(run_root, monkeypatch):
    _fast(monkeypatch)
    monkeypatch.setattr(config.signal_v5, "strict_quantile", 0.2)
    with pytest.raises(RuntimeError, match="different settings"):
        v5.step_evaluate(config)


# ---------------------------------------------------------------- features


def _write_crops(root, family, n, seed=0):
    w, h = SMALL_CANVAS[family]
    rng = np.random.default_rng(seed)
    paths = []
    for k in range(n):
        img = np.full((h, w), 0, dtype=np.uint8)
        img[h // 4 : h // 2, 2 + k : 10 + k] = rng.integers(150, 255)
        path = root / f"{k:03d}_{family}.png"
        from PIL import Image

        Image.fromarray(img).save(path)
        paths.append(path)
    return paths


class _FakeViT(torch.nn.Module):
    embed_dim = 384

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, self.embed_dim, 14, stride=14)
        self.seen = []

    def forward_features(self, x):
        self.seen.append(tuple(x.shape))
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return {"x_norm_clstoken": tokens.mean(1), "x_norm_patchtokens": tokens}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
    monkeypatch.setattr(config.embedding, "weights", None)
    monkeypatch.setattr(config.signal_v5, "batch_size", 3)


def test_style_features_shape_and_determinism(tmp_path, small):
    paths = _write_crops(tmp_path, "word", 4)
    torch.manual_seed(0)
    model = load_backbone("cpu", weights=None)
    a = embeddings_v5.style_features(paths, "word", "cpu", model)
    b = embeddings_v5.style_features(paths, "word", "cpu", model)
    assert a.shape == (4, STYLE_DIM) and np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert not model.layer1._forward_hooks


@pytest.mark.parametrize("h, w", [(300, 900), (900, 300), (434, 434), (50, 70)])
def test_dinov2_size_keeps_aspect_and_patch_multiple(h, w):
    (rh, rw), (ph, pw) = embeddings_v5.dinov2_size(h, w, config)
    assert max(rh, rw) == config.signal_v5.dinov2_long_side
    assert abs(rh / rw - h / w) < 0.01
    assert ph % 14 == 0 and pw % 14 == 0 and ph >= rh and pw >= rw


def test_dinov2_features_shape_and_white_padding(tmp_path, small):
    paths = _write_crops(tmp_path, "cursive", 4)
    fake = _FakeViT().eval()
    out = embeddings_v5.dinov2_features(paths, "cpu", fake, config)
    assert out.shape == (4, 2 * _FakeViT.embed_dim)
    _, (ph, pw) = embeddings_v5.dinov2_size(32, 64, config)
    assert fake.seen[0][2:] == (ph, pw)
    x = embeddings_v5.dinov2_input(paths[0], config)
    white = (1.0 - torch.tensor(embeddings_v5.IMAGENET_MEAN)) / torch.tensor(
        embeddings_v5.IMAGENET_STD
    )
    assert torch.allclose(x[:, -1, -1], white)


def test_run_writes_caches_and_reuses_them(tmp_path, small, monkeypatch, use_root):
    root = use_root(
        make_synthetic_root(tmp_path / "img", n_participants=10, images=True)
    )
    monkeypatch.setattr(embeddings_v5, "load_dinov2", lambda d, c=config: _FakeViT())
    monkeypatch.setattr(
        embeddings_v5, "load_backbone", lambda d: load_backbone(d, weights=None)
    )
    written = embeddings_v5.run(("style", "dinov2"), device="cpu")
    assert len(written) == 6
    with np.load(embeddings_v5.output_path("style", "word", config)) as z:
        assert z["embeddings"].shape[1] == STYLE_DIM
        assert len(set(z["participant_id"])) == 10
    with np.load(embeddings_v5.output_path("dinov2", "drawing", config)) as z:
        assert "weights_sha256" in z.files
    stamp = embeddings_v5.output_path("dinov2", "word", config).stat().st_mtime_ns
    monkeypatch.setattr(
        embeddings_v5, "load_dinov2", lambda *a: pytest.fail("should reuse cache")
    )
    embeddings_v5.run(("dinov2",), device="cpu")
    assert (
        embeddings_v5.output_path("dinov2", "word", config).stat().st_mtime_ns == stamp
    )
    assert root.exists()


def test_evaluator_accepts_v5_predictions(run_root):
    preds = _read(v5.output_dir(config) / "predictions.csv")
    evaluator.validate(preds, "primary", "v5", config)
