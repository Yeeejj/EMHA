"""Tests for v3 per-exercise training (exploratory, Deviation 6)."""

import numpy as np
import pandas as pd
import pytest

from src.data.crop_manifest import ALL_CELLS, CURSIVE_CELLS, DRAWING_CELLS, WORD_CELLS
from src.data.dataloader import load_tables
from src.features.handcrafted import ALL_FEATURES, features_for_family
from src.training import run_per_task_v3 as v3
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

SMALL_CANVAS = {"drawing": (32, 48), "word": (48, 32), "cursive": (64, 32)}
N_PARTICIPANTS = 60
IMPUTED_EVERY = 7


def test_cells_match_crop_manifest():
    assert config.per_task_v3.cells == ALL_CELLS


def test_v3_settings_leave_the_run_fingerprint_unchanged(monkeypatch):
    from src.pipeline_runner import config_fingerprint

    before = config_fingerprint()
    monkeypatch.setattr(config.per_task_v3, "n_lr_permutations", 7)
    assert config_fingerprint() == before


def _write_handcrafted_crop(root, labels):
    """results/features/handcrafted_crop.csv, one row per (participant, cell)."""
    rng = np.random.default_rng(0)
    rows = []
    for i, pid in enumerate(labels["participant_id"]):
        for cell in ALL_CELLS:
            family = v3.family_of(cell)
            row = {"participant_id": pid, "cell": cell, "task_family": family}
            row.update({f: np.nan for f in ALL_FEATURES})
            row.update({f: rng.normal() for f in features_for_family(family)})
            row["imputed"] = "slant_deg" if i % IMPUTED_EVERY == 0 else ""
            rows.append(row)
    path = root / "results" / "features" / "handcrafted_crop.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def v3_root(tmp_path, monkeypatch, use_root):
    monkeypatch.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
    root = make_synthetic_root(
        tmp_path / "root", n_participants=N_PARTICIPANTS, signal=True, images=True
    )
    use_root(root)
    labels = pd.read_csv(
        root / "metadata" / "labels.csv", dtype={"participant_id": str}
    )
    _write_handcrafted_crop(root, labels)
    return root


def test_family_of_maps_every_cell():
    assert [v3.family_of(c) for c in DRAWING_CELLS] == ["drawing"] * 4
    assert [v3.family_of(c) for c in WORD_CELLS] == ["word"] * 15
    assert [v3.family_of(c) for c in CURSIVE_CELLS] == ["cursive"] * 5
    with pytest.raises(ValueError, match="unknown cell"):
        v3.family_of("X1")


def test_cell_manifest_one_row_per_participant(v3_root):
    _, ids, _ = v3.context(config)
    manifest, _, dropped = load_tables(config)
    rows = v3.cell_manifest(manifest, "W2_RH", ids, dropped)
    assert list(rows["participant_id"]) == sorted(ids)
    assert set(rows["cell"]) == {"W2_RH"}

    target = manifest[
        (manifest["cell"] == "W2_RH") & manifest["participant_id"].eq(ids[0])
    ]
    with pytest.raises(ValueError, match="repeated"):
        v3.cell_manifest(pd.concat([manifest, target]), "W2_RH", ids, dropped)
    with pytest.raises(ValueError, match="missing"):
        v3.cell_manifest(manifest.drop(target.index), "W2_RH", ids, dropped)


def test_fold_splits_shared_by_every_cell(v3_root):
    labels, ids, folds = v3.context(config)
    manifest, _, dropped = load_tables(config)
    tested = []
    for fold in range(config.cv.n_splits):
        inner_train, inner_val, test = v3.fold_splits(fold, ids, labels, folds)
        sets = [set(inner_train), set(inner_val), set(test)]
        assert all(sets)
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert sets[0] | sets[1] | sets[2] == set(ids)
        for cell in ALL_CELLS:
            assert v3.fold_splits(fold, ids, labels, folds) == (
                inner_train,
                inner_val,
                test,
            )
            for part in (inner_train, inner_val, test):
                rows = v3.cell_manifest(manifest, cell, part, dropped)
                assert list(rows["participant_id"]) == sorted(part)
        tested += test
    assert sorted(tested) == sorted(ids)


def test_preflight_passes_on_clean_root(v3_root, capsys):
    summary = v3.preflight(config)
    _, ids, _ = v3.context(config)
    assert summary["synthetic_root"] is True
    assert summary["n_primary"] == len(ids)
    assert summary["n_happy"] + summary["n_sad"] == len(ids)
    assert summary["n_cells"] == len(ALL_CELLS)
    assert summary["n_dropped_primary_crops"] == 0
    assert len(summary["inner_splits"]) == config.cv.n_splits
    assert sum(s["n_test"] for s in summary["inner_splits"]) == len(ids)
    expected = len(range(0, N_PARTICIPANTS, IMPUTED_EVERY))
    assert set(summary["imputed_crops_per_cell"].values()) == {expected}
    assert '"n_primary"' in capsys.readouterr().out


def test_preflight_rejects_dropped_primary_crop(v3_root):
    _, ids, _ = v3.context(config)
    pd.DataFrame(
        {"participant_id": [ids[0]], "cell": ["CS3"], "dropped": [True]}
    ).to_csv(v3_root / "metadata" / "qc_log.csv", index=False)
    with pytest.raises(ValueError, match="dropped crops"):
        v3.preflight(config)


def test_preflight_rejects_missing_embedding_row(v3_root):
    _, ids, _ = v3.context(config)
    path = v3.embedding_path("word", config)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    keep = ~((arrays["participant_id"] == ids[0]) & (arrays["cell"] == "W4_UC"))
    for key in ("embeddings", "participant_id", "cell"):
        arrays[key] = arrays[key][keep]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="embedding W4_UC"):
        v3.preflight(config)


def test_preflight_rejects_nan_handcrafted_value(v3_root):
    _, ids, _ = v3.context(config)
    path = v3.handcrafted_path(config)
    table = pd.read_csv(path, dtype={"participant_id": str})
    hit = (table["participant_id"] == ids[0]) & (table["cell"] == "D2")
    table.loc[hit, features_for_family("drawing")[0]] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="D2: NaN handcrafted"):
        v3.preflight(config)


LR_CELL = "W3_UC"
LR_FOLD = 1
FEW_CELLS = ("D2", "W3_UC", "CS1")


def _randomize_test_rows(cell, test):
    """Overwrite the test participants' features of one cell with noise."""
    rng = np.random.default_rng(123)
    path = v3.handcrafted_path(config)
    table = pd.read_csv(path, dtype={"participant_id": str})
    hit = table["participant_id"].isin(test) & (table["cell"] == cell)
    feats = list(features_for_family(v3.family_of(cell)))
    table.loc[hit, feats] = rng.normal(5.0, 3.0, size=(int(hit.sum()), len(feats)))
    table.to_csv(path, index=False)

    emb_path = v3.embedding_path(v3.family_of(cell), config)
    with np.load(emb_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    rows = np.isin(arrays["participant_id"], test) & (arrays["cell"] == cell)
    arrays["embeddings"][rows] = rng.normal(
        5.0, 3.0, size=(rows.sum(), arrays["embeddings"].shape[1])
    )
    np.savez_compressed(emb_path, **arrays)


def _fitted_state(pipe):
    state = {
        "scaler_mean": pipe["scaler"].mean_,
        "scaler_scale": pipe["scaler"].scale_,
        "coef": pipe["lr"].coef_,
        "intercept": pipe["lr"].intercept_,
        "C": np.array([pipe["lr"].C]),
    }
    if "pca" in pipe.named_steps:
        state["pca"] = pipe["pca"].components_
    return state


def test_lr_fit_ignores_test_participants(v3_root):
    labels, ids, folds = v3.context(config)
    before = {}
    for kind in config.per_task_v3.lr_kinds:
        pipe, X, _, test = v3.fit_lr_cell(
            kind, LR_CELL, LR_FOLD, ids, labels, folds, config
        )
        preds = v3.lr_cell(kind, LR_CELL, LR_FOLD, ids, labels, folds, config)
        before[kind] = (_fitted_state(pipe), preds, test)
    _randomize_test_rows(LR_CELL, before[config.per_task_v3.lr_kinds[0]][2])

    for kind in config.per_task_v3.lr_kinds:
        state, preds, test = before[kind]
        pipe, _, _, test_after = v3.fit_lr_cell(
            kind, LR_CELL, LR_FOLD, ids, labels, folds, config
        )
        assert test_after == test
        after = _fitted_state(pipe)
        assert set(after) == set(state)
        for key in state:
            assert np.allclose(after[key], state[key]), (kind, key)
        new = v3.lr_cell(kind, LR_CELL, LR_FOLD, ids, labels, folds, config)
        assert list(new["participant_id"]) == list(preds["participant_id"])
        assert not np.allclose(new["prob_sad"], preds["prob_sad"])


def _read_parts():
    parts = sorted((v3.output_dir(config) / "parts").glob("*.csv"))
    return parts, pd.concat(
        [pd.read_csv(p, dtype={"participant_id": str}) for p in parts],
        ignore_index=True,
    )


def test_lr_parts_cover_each_participant_once(v3_root):
    v3.step_lr(config)
    _, ids, folds = v3.context(config)
    parts, preds = _read_parts()
    v3cfg = config.per_task_v3
    assert len(parts) == len(v3cfg.lr_kinds) * len(ALL_CELLS) * config.cv.n_splits
    assert tuple(preds.columns) == v3.PRED_COLUMNS
    assert set(preds["analysis"]) == {"primary"}
    assert not preds["in_middle_band"].any()
    fold_of = folds.set_index(folds["participant_id"].astype(str))["outer_fold"]
    for (kind, cell), group in preds.groupby(["model", "feature_set"]):
        assert kind in v3cfg.lr_kinds and cell in ALL_CELLS
        assert not group["participant_id"].duplicated().any()
        assert sorted(group["participant_id"]) == sorted(ids)
        assert set(group["fold"]) == set(range(config.cv.n_splits))
        expected = fold_of.loc[group["participant_id"]].to_numpy()
        assert (group["fold"].to_numpy() == expected).all()
    assert preds.groupby("participant_id")["fold"].nunique().max() == 1


def test_lr_rerun_resumes_and_refuses_changed_settings(v3_root, monkeypatch, capsys):
    monkeypatch.setattr(config.per_task_v3, "cells", FEW_CELLS)
    v3.step_lr(config)
    capsys.readouterr()
    calls = []
    monkeypatch.setattr(v3, "fit_lr", lambda *a, **k: calls.append(a))
    v3.step_lr(config)
    out = capsys.readouterr().out
    assert calls == []
    assert f"resume: lr_handcrafted_v3 {FEW_CELLS[0]} fold 0" in out
    assert out.count("resume:") == 2 * len(FEW_CELLS) * config.cv.n_splits
    assert "macro" not in out.lower() and "accuracy" not in out.lower()

    monkeypatch.setattr(config.training, "seed", config.training.seed + 1)
    with pytest.raises(RuntimeError, match="different settings"):
        v3.step_lr(config)
