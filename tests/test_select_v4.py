"""Tests for v4 exercise selection (exploratory, Deviation 7)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data.crop_manifest import ALL_CELLS
from src.features.handcrafted import ALL_FEATURES, features_for_family
from src.training import run_baselines
from src.training import run_per_task_v3 as v3
from src.training import run_select_v4 as v4
from src.training.run_baselines import PRED_COLUMNS
from src.training.splits import fold_ids
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

N_PARTICIPANTS = 60
N_DRAWS = 2
FAST = {
    ("baselines", "C_grid"): (1.0,),
    ("baselines", "inner_cv"): 2,
    ("baselines", "pca_components"): 4,
    ("select_v4", "rank_inner_folds"): 2,
    ("select_v4", "n_permutations"): N_DRAWS,
    ("per_task_v3", "n_cnn_permutations"): N_DRAWS,
    ("evaluation", "n_bootstrap"): 50,
}


def _fast(mp):
    for (section, key), value in FAST.items():
        mp.setattr(getattr(config, section), key, value)


def _write_handcrafted_crop(root, labels):
    rng = np.random.default_rng(0)
    rows = []
    for pid in labels["participant_id"]:
        for cell in ALL_CELLS:
            family = v3.family_of(cell)
            row = {"participant_id": pid, "cell": cell, "task_family": family}
            row.update({f: np.nan for f in ALL_FEATURES})
            row.update({f: rng.normal() for f in features_for_family(family)})
            row["imputed"] = ""
            rows.append(row)
    path = root / "results" / "features" / "handcrafted_crop.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def _per_cell_rows(labels, folds, ids, seed, models=("cnn_head_v3", "cnn_hmm_v3")):
    """Stand-in v3 per-exercise test rows (random prob_sad, true labels)."""
    rng = np.random.default_rng(seed)
    frames = []
    for model in models:
        for cell in ALL_CELLS:
            for fold in range(config.cv.n_splits):
                _, test = fold_ids(folds, fold, ids)
                prob = rng.uniform(size=len(test))
                frames.append(
                    v3._participant_rows(model, cell, fold, test, prob, labels, config)
                )
    return pd.concat(frames, ignore_index=True)[list(PRED_COLUMNS)]


def _write_v3_and_references(root):
    labels, ids, folds = v3.context(config)
    out = v3.output_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    _per_cell_rows(labels, folds, ids, seed=1).to_csv(
        out / "predictions.csv", index=False
    )
    for perm in range(N_DRAWS):
        path = v3.draw_path(perm, config.training.seed, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        _per_cell_rows(labels, folds, ids, seed=100 + perm).assign(perm=perm).to_csv(
            path, index=False
        )
    run_baselines.run("primary")
    base = pd.read_csv(
        root / "results" / "baselines" / "primary" / "predictions.csv",
        dtype={"participant_id": str},
    )
    rows = base[(base["model"] == "lr_handcrafted") & (base["feature_set"] == "all")]
    hybrid = root / "results" / "hybrid" / "primary" / "predictions.csv"
    hybrid.parent.mkdir(parents=True, exist_ok=True)
    rows.assign(model="cnn_hmm_fused", feature_set="fused").to_csv(hybrid, index=False)


def _hashes(root) -> dict:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def _new_root(path):
    root = make_synthetic_root(path, n_participants=N_PARTICIPANTS, signal=True)
    labels = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    _write_handcrafted_crop(root, labels)
    return root


@pytest.fixture
def v4_root(tmp_path, monkeypatch, use_root):
    _fast(monkeypatch)
    root = use_root(_new_root(tmp_path / "root"))
    _write_v3_and_references(root)
    return root


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    """Every v4 step run once on a root holding v1 and v3 stand-ins."""
    with pytest.MonkeyPatch.context() as mp:
        _fast(mp)
        root = _new_root(tmp_path_factory.mktemp("v4") / "root")
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        _write_v3_and_references(root)
        before = _hashes(root)
        v4.step_rank(config)
        v4.step_combine(config)
        v4.step_evaluate(config)
        v4.step_draws(config)
        v4.step_permutation(config)
        v4.step_report(config)
        yield root, before


def _read(path):
    return pd.read_csv(path, dtype={"participant_id": str})


# ---------------------------------------------------------------- config


def test_v4_settings_leave_the_run_fingerprint_unchanged(monkeypatch):
    from src.pipeline_runner import config_fingerprint

    before = config_fingerprint()
    monkeypatch.setattr(config.select_v4, "headline_k", 3)
    assert config_fingerprint() == before


@pytest.mark.parametrize(
    "key, value",
    [
        ("k_grid", (1, 3, 6, 12)),
        ("k_grid", (6, 3, 24)),
        ("headline_k", 24),
        ("headline_k", 5),
        ("headline_model", "cnn_hmm_v3"),
    ],
)
def test_check_config_rejects_bad_settings(monkeypatch, key, value):
    monkeypatch.setattr(config.select_v4, key, value)
    with pytest.raises(ValueError):
        v4.check_config(config)


def test_preflight_passes_on_clean_root(v4_root, capsys):
    summary = v4.preflight(config)
    assert summary["n_primary"] > 0
    assert summary["v3_draws_present"] == f"{N_DRAWS}/{N_DRAWS}"
    assert "headline" in capsys.readouterr().out


def test_preflight_rejects_incomplete_v3_predictions(v4_root):
    path = v3.output_dir(config) / "predictions.csv"
    df = _read(path)
    df.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="test participants"):
        v4.preflight(config)


# ---------------------------------------------------------------- leakage


def _scramble_test(tables, labels, test, seed=7):
    """Copies with the test participants' features and labels replaced."""
    rng = np.random.default_rng(seed)
    new_tables = {}
    for kind, per_cell in tables.items():
        new_tables[kind] = {}
        for cell, X in per_cell.items():
            X = X.copy()
            X.loc[test] = (rng.normal(size=(len(test), X.shape[1])) * 50).astype(
                X.dtypes.iloc[0]
            )
            new_tables[kind][cell] = X
    new_labels = labels.copy()
    mask = new_labels["participant_id"].isin(test)
    col = config.labeling.label_column
    new_labels.loc[mask, col] = np.where(
        new_labels.loc[mask, col] == "SAD", "HAPPY", "SAD"
    )
    return new_tables, new_labels


@pytest.mark.parametrize("shuffle_seed", [None, 12345])
def test_ranking_ignores_test_participants(v4_root, shuffle_seed):
    labels, ids, folds = v3.context(config)
    tables = v4.load_tables(config)
    _, test = fold_ids(folds, 0, ids)
    a = v4.rank_fold(0, ids, labels, folds, tables, config, shuffle_seed)
    t2, l2 = _scramble_test(tables, labels, test)
    b = v4.rank_fold(0, ids, l2, folds, t2, config, shuffle_seed)
    pd.testing.assert_frame_equal(a, b)
    assert sorted(a["rank"]) == list(range(1, len(ALL_CELLS) + 1))


def test_lr_rows_ignore_test_labels_and_report_true_labels(v4_root):
    labels, ids, folds = v3.context(config)
    tables = v4.load_tables(config)
    _, test = fold_ids(folds, 1, ids)
    cells = ["D1", "W2_LH", "CS4"]
    kind = config.select_v4.lr_models[0][1]
    a = v4.lr_rows("m", kind, 1, cells, 3, ids, labels, folds, tables, config)
    _, flipped = _scramble_test(tables, labels, test)
    b = v4.lr_rows("m", kind, 1, cells, 3, ids, flipped, folds, tables, config)
    np.testing.assert_array_equal(a["prob_sad"], b["prob_sad"])
    lab = labels.set_index("participant_id")[config.labeling.label_column]
    assert list(a["label"]) == list(lab.loc[a["participant_id"]])
    assert set(a["participant_id"]) == set(test)
    assert set(a["feature_set"]) == {"top3"}


def test_ranking_null_shuffle_touches_only_training_labels(v4_root, monkeypatch):
    labels, ids, folds = v3.context(config)
    tables = v4.load_tables(config)
    train, test = fold_ids(folds, 2, ids)
    real, seen = v4.shuffled_labels, {}

    def spy(labels_, ids_, seed):
        seen["ids"] = sorted(ids_)
        return real(labels_, ids_, seed)

    monkeypatch.setattr(v4, "shuffled_labels", spy)
    v4.rank_fold(2, ids, labels, folds, tables, config, shuffle_seed=3)
    assert seen["ids"] == sorted(train)
    assert not set(seen["ids"]) & set(test)


# ---------------------------------------------------------------- full run


def test_v4_changes_no_existing_file(run_root):
    root, before = run_root
    after = _hashes(root)
    changed = [p for p in before if after.get(p) != before[p]]
    assert changed == []
    allowed = ("results/select_v4/", "results/final/")
    new = [p for p in after if p not in before]
    assert new and all(p.startswith(allowed) for p in new), new


def test_predictions_cover_each_participant_once_per_group(run_root):
    preds = _read(v4.output_dir(config) / "predictions.csv")
    _, ids, _ = v3.context(config)
    groups = preds.groupby(["model", "feature_set"])
    assert len(groups) == len(v4.v4_models(config)) * len(config.select_v4.k_grid)
    for _, g in groups:
        assert sorted(g["participant_id"]) == sorted(ids)


def test_selection_is_nested_and_has_k_cells(run_root):
    selected = pd.read_csv(v4.output_dir(config) / "selected_cells.csv")
    ks = config.select_v4.k_grid
    for fold in range(config.cv.n_splits):
        previous = []
        for k in ks:
            kept = v4.kept_cells(selected, k, fold)
            assert len(kept) == len(set(kept)) == k
            assert kept[: len(previous)] == previous
            previous = kept


def test_top24_cnn_rows_equal_v3_all24_fusion(run_root):
    preds = _read(v4.output_dir(config) / "predictions.csv")
    per_cell = _read(v3.output_dir(config) / "predictions.csv")
    full = v3.fuse(per_cell, "cnn_hmm_v3", "x", config)
    full = full[full["feature_set"] == "all24"].set_index("participant_id")
    mine = preds[(preds["model"] == "cnn_hmm_v4") & (preds["feature_set"] == "top24")]
    mine = mine.set_index("participant_id")
    np.testing.assert_allclose(
        mine["prob_sad"], full.loc[mine.index, "prob_sad"], atol=1e-12
    )


def test_topk_cnn_rows_are_the_mean_of_the_kept_cells(run_root):
    out = v4.output_dir(config)
    preds = _read(out / "predictions.csv")
    selected = pd.read_csv(out / "selected_cells.csv")
    per_cell = _read(v3.output_dir(config) / "predictions.csv")
    k = config.select_v4.headline_k
    for fold in range(config.cv.n_splits):
        kept = v4.kept_cells(selected, k, fold)
        rows = per_cell[
            (per_cell["model"] == "cnn_head_v3") & (per_cell["fold"] == fold)
        ]
        want = (
            rows[rows["feature_set"].isin(kept)]
            .groupby("participant_id")["prob_sad"]
            .mean()
        )
        got = preds[
            (preds["model"] == "cnn_head_v4")
            & (preds["feature_set"] == f"top{k}")
            & (preds["fold"] == fold)
        ].set_index("participant_id")["prob_sad"]
        np.testing.assert_allclose(got, want.loc[got.index], atol=1e-12)


def test_selection_frequency_matches_ranking(run_root):
    out = v4.output_dir(config)
    ranking = pd.read_csv(out / "ranking.csv")
    freq = pd.read_csv(out / "selection_frequency.csv").set_index("cell")
    k = config.select_v4.headline_k
    col = f"folds_in_top{k}"
    assert freq[col].sum() == k * config.cv.n_splits
    for cell, part in ranking.groupby("cell"):
        assert freq.loc[cell, col] == (part["rank"] <= k).sum()


def test_evaluate_writes_removal_and_v1_comparisons(run_root):
    paired = pd.read_csv(v4.output_dir(config) / "paired.csv")
    removal = paired[paired["comparison"] == "removal"]
    assert len(removal) == len(v4.v4_models(config))
    assert removal["delta_macro_f1"].notna().all()
    v1 = paired[paired["comparison"] == "headline_vs_v1"]
    assert v1["delta_macro_f1"].notna().all()


def test_draws_carry_true_labels_and_every_model(run_root):
    labels, _, _ = v3.context(config)
    lab = labels.set_index("participant_id")[config.labeling.label_column]
    for perm in range(N_DRAWS):
        frame = _read(v4.draw_path(perm, config.training.seed, config))
        assert set(frame["model"]) == set(v4.v4_models(config))
        assert list(frame["label"]) == list(lab.loc[frame["participant_id"]])


def test_permutation_and_report(run_root):
    out = v4.output_dir(config)
    perm = json.loads((out / "permutation.json").read_text())
    assert perm["headline"] == "/".join(v4.headline(config))
    assert 1 / (N_DRAWS + 1) <= perm["p_value"] <= 1
    text = v4.report_path(config).read_text()
    assert v4.REPORT_HEADER in text and "Pre-declared v4 test" in text
    for name in ("k_curve.png", "selection.png", "permutation_null.png"):
        assert v4.figure_path(name, config).is_file()


def test_rerun_refuses_changed_settings(run_root, monkeypatch):
    monkeypatch.setattr(config.select_v4, "headline_k", 3)
    with pytest.raises(RuntimeError, match="Deviation 7"):
        v4.step_combine(config)


def test_rerun_resumes_ranking(run_root, capsys):
    before = (v4.output_dir(config) / "ranking.csv").read_bytes()
    v4.step_rank(config)
    assert "resume: rank fold 0" in capsys.readouterr().out
    assert (v4.output_dir(config) / "ranking.csv").read_bytes() == before
