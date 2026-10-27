"""Tests for v3 per-exercise training (exploratory, Deviation 6)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.crop_manifest import ALL_CELLS, CURSIVE_CELLS, DRAWING_CELLS, WORD_CELLS
from src.data.dataloader import load_tables
from src.features.handcrafted import ALL_FEATURES, features_for_family
from src.models.hmm import HMMClassifier
from src.training import run_baselines
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


V3_CELLS = ("D1", "W1_LH", "CS1")
STAND_INS = (
    "models/hmm/primary/word/fold_0.joblib",
    "models/cnn/primary/word/fold_0.pth",
    "results/final/RESULTS.txt",
    "results/final/model_comparison.csv",
    "results/final/permutation_runs/primary/seed_42/perm_0.csv",
    "models/hmm_v2/primary/fold_0.joblib",
    "results/final/permutation_runs/primary_v2/seed_42/perm_0.csv",
    "results/final/EXPLORATORY_v2.txt",
)
V3_PREFIXES = (
    "results/per_task_v3/primary/",
    "models/cnn_v3/primary/",
    "models/hmm_v3/primary/",
    "results/final/EXPLORATORY_v3.txt",
    "results/final/fig_v3_",
)
REFERENCE_GROUPS = {
    "cnn": (("cnn_head_fused", "fused"), ("cnn_head", "word"), ("cnn_head", "cursive")),
    "hybrid": (("cnn_hmm_fused", "fused"), ("cnn_hmm", "word"), ("cnn_hmm", "cursive")),
    "hybrid_v2": (("cnn_hmm_v2", "task_order"),),
}


def _hashes(root) -> dict:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def _write_references(root):
    """Stand-in v1/v2 reference predictions: relabelled lr_handcrafted/all."""
    run_baselines.run("primary")
    base = pd.read_csv(
        root / "results" / "baselines" / "primary" / "predictions.csv",
        dtype={"participant_id": str},
    )
    rows = base[(base["model"] == "lr_handcrafted") & (base["feature_set"] == "all")]
    for source, groups in REFERENCE_GROUPS.items():
        path = root / "results" / source / "primary" / "predictions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([rows.assign(model=m, feature_set=fs) for m, fs in groups]).to_csv(
            path, index=False
        )


@pytest.fixture(scope="module")
def trained_root(tmp_path_factory):
    """lr, cnn and hmm run once on a root holding v1 and v2 stand-ins."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config.preprocessing, "canvas_size", SMALL_CANVAS)
        mp.setattr(config.training, "epochs", 1)
        mp.setattr(config.training, "batch_size", 16)
        mp.setattr(config.data, "num_workers", 0)
        mp.setattr(config.cnn, "use_pretrained", False)
        mp.setattr(config.hmm, "n_restarts", 1)
        mp.setattr(config.hmm, "n_iter", 10)
        mp.setattr(config.hybrid, "selection_restarts", 1)
        mp.setattr(config.hybrid, "n_states_grid", (2,))
        mp.setattr(config.hybrid, "pca_grid", (4,))
        mp.setattr(config.runner, "hmm_n_jobs", 1)
        mp.setattr(config.per_task_v3, "cells", V3_CELLS)
        mp.setattr(config.per_task_v3, "n_cnn_permutations", 0)
        mp.setattr(config.evaluation, "n_bootstrap", 50)
        root = make_synthetic_root(
            tmp_path_factory.mktemp("v3") / "root",
            n_participants=40,
            signal=True,
            images=True,
        )
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        labels = pd.read_csv(
            root / "metadata" / "labels.csv", dtype={"participant_id": str}
        )
        _write_handcrafted_crop(root, labels)
        for rel in STAND_INS:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"stand-in file {rel}\n")
        _write_references(root)
        before = _hashes(root)
        v3.step_lr(config)
        v3.step_cnn(config, device="cpu")
        v3.step_hmm(config, device="cpu")
        yield root, before


def _context():
    labels, ids, folds = v3.context(config)
    manifest, _, dropped = load_tables(config)
    return labels, ids, folds, manifest, dropped


def _read(path):
    return pd.read_csv(path, dtype={"participant_id": str})


def test_v3_steps_change_no_v1_or_v2_file(trained_root):
    root, before = trained_root
    after = _hashes(root)
    assert [p for p in before if after.get(p) != before[p]] == []
    created = sorted(set(after) - set(before))
    assert created
    assert [p for p in created if not p.startswith(V3_PREFIXES)] == []


def test_v3_checkpoints_record_fold_splits(trained_root):
    labels, ids, folds = v3.context(config)
    for cell in config.per_task_v3.cells:
        for fold in range(config.cv.n_splits):
            state = torch.load(v3.cnn_ckpt_path(cell, fold, config), map_location="cpu")
            inner_train, inner_val, test = v3.fold_splits(fold, ids, labels, folds)
            assert state["inner_train_ids"] == inner_train, (cell, fold)
            assert state["inner_val_ids"] == inner_val, (cell, fold)
            assert state["test_ids"] == test, (cell, fold)
            assert state["middle_ids"] == []
            assert (state["cell"], state["fold"]) == (cell, fold)
            assert state["family"] == v3.family_of(cell)


def test_cnn_and_hmm_parts_cover_test_and_inner_val(trained_root):
    labels, ids, folds = v3.context(config)
    true = labels.set_index("participant_id")["label"]
    v3cfg = config.per_task_v3
    for model in (v3cfg.cnn_model, v3cfg.hmm_model):
        for cell in v3cfg.cells:
            tested = []
            for fold in range(config.cv.n_splits):
                _, inner_val, test = v3.fold_splits(fold, ids, labels, folds)
                rows = _read(v3.part_path(model, cell, fold, config))
                val = _read(v3.val_part_path(model, cell, fold, config))
                for frame, want in ((rows, test), (val, inner_val)):
                    assert tuple(frame.columns) == v3.PRED_COLUMNS
                    assert list(frame["participant_id"]) == sorted(want)
                    assert set(frame["model"]) == {model}
                    assert set(frame["feature_set"]) == {cell}
                    assert set(frame["fold"]) == {fold}
                    assert not frame["in_middle_band"].any()
                    assert frame["prob_sad"].between(0, 1).all()
                    expected = true.loc[frame["participant_id"]].to_numpy()
                    assert (frame["label"].to_numpy() == expected).all()
                tested += list(rows["participant_id"])
            assert sorted(tested) == sorted(ids)
    for cell in v3cfg.cells:
        for fold in range(config.cv.n_splits):
            record = v3.selection_path(cell, fold, config).read_text()
            assert f'"cell": "{cell}"' in record


def test_cnn_and_hmm_rerun_resumes_without_metrics(trained_root, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(v3, "fit_cnn_cell", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(v3, "fit_hmm_cell", lambda *a, **k: calls.append(a))
    v3.step_cnn(config, device="cpu")
    v3.step_hmm(config, device="cpu")
    out = capsys.readouterr().out
    assert calls == []
    n_units = len(config.per_task_v3.cells) * config.cv.n_splits
    assert out.count("resume: cnn ") == n_units
    assert out.count("resume: hmm ") == n_units
    assert "resume: cnn D1 fold 0" in out
    for word in ("macro", "accuracy", "f1", "auc"):
        assert word not in out.lower()
    with pytest.raises(ValueError, match="unknown cells"):
        v3.step_cnn(config, cells=["W2_RH"], device="cpu")
    with pytest.raises(ValueError, match="out of range"):
        v3.step_hmm(config, folds=[config.cv.n_splits], device="cpu")


def test_hmm_refuses_without_cnn_checkpoint(trained_root, tmp_path, monkeypatch):
    labels, ids, folds, manifest, dropped = _context()
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="run --step cnn first"):
        v3.fit_hmm_cell(
            "D1", 0, ids, labels, folds, manifest, dropped, "cpu", missing, config
        )
    monkeypatch.setattr(v3, "cnn_ckpt_path", lambda *a, **k: missing)
    monkeypatch.setattr(v3, "hmm_model_path", lambda *a, **k: tmp_path / "x.joblib")
    with pytest.raises(FileNotFoundError, match="run --step cnn first"):
        v3.step_hmm(config, cells=["D1"], folds=[0], device="cpu")


def _hmm_state(clf) -> dict:
    state = {
        "pca_components": clf.pca.components_,
        "pca_mean": clf.pca.mean_,
        "platt_coef": clf.calibrator.coef_,
        "platt_intercept": clf.calibrator.intercept_,
        "log_prior_ratio": np.array([clf.log_prior_ratio]),
    }
    if clf.scaler is not None:
        state["scaler_mean"] = clf.scaler.mean_
        state["scaler_scale"] = clf.scaler.scale_
    for name, model in clf.models.items():
        for attr in ("startprob_", "transmat_", "means_", "covars_"):
            state[f"{name}_{attr}"] = getattr(model, attr)
    return state


def _noisy_manifest(manifest, cell, test, out_dir):
    """manifest with the test participants' crops of cell replaced by noise."""
    rng = np.random.default_rng(7)
    noisy = manifest.copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    hit = noisy["participant_id"].isin(test) & (noisy["cell"] == cell)
    assert hit.sum() == len(test)
    for idx in noisy.index[hit]:
        shape = np.array(Image.open(noisy.at[idx, "processed_path"])).shape
        dst = out_dir / f"{noisy.at[idx, 'participant_id']}_{cell}.png"
        Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8)).save(dst)
        noisy.at[idx, "processed_path"] = str(dst)
    return noisy


def test_cnn_and_hmm_fits_ignore_test_participants(trained_root, tmp_path):
    cell, fold = "CS1", 1
    labels, ids, folds, manifest, dropped = _context()
    _, _, test = v3.fold_splits(fold, ids, labels, folds)
    noisy = _noisy_manifest(manifest, cell, test, tmp_path / "noise")
    ckpt = tmp_path / "models" / "cnn.pth"
    hmm_file = tmp_path / "models" / "hmm.joblib"
    v3cfg = config.per_task_v3

    cnn_test, cnn_val = v3.fit_cnn_cell(
        cell, fold, ids, labels, folds, noisy, dropped, "cpu", ckpt, config
    )
    ref = torch.load(v3.cnn_ckpt_path(cell, fold, config), map_location="cpu")
    new = torch.load(ckpt, map_location="cpu")
    assert set(ref["model_state_dict"]) == set(new["model_state_dict"])
    for key, value in ref["model_state_dict"].items():
        assert torch.allclose(value.float(), new["model_state_dict"][key].float()), key

    hmm_test, hmm_val, _ = v3.fit_hmm_cell(
        cell,
        fold,
        ids,
        labels,
        folds,
        noisy,
        dropped,
        "cpu",
        ckpt,
        config,
        save_path=hmm_file,
    )
    before = _hmm_state(HMMClassifier().load(v3.hmm_model_path(cell, fold, config)))
    after = _hmm_state(HMMClassifier().load(hmm_file))
    assert set(before) == set(after)
    for key in before:
        assert np.allclose(before[key], after[key]), key

    for model, new_test, new_val in (
        (v3cfg.cnn_model, cnn_test, cnn_val),
        (v3cfg.hmm_model, hmm_test, hmm_val),
    ):
        old_test = _read(v3.part_path(model, cell, fold, config))
        old_val = _read(v3.val_part_path(model, cell, fold, config))
        assert list(new_test["participant_id"]) == list(old_test["participant_id"])
        assert not np.allclose(new_test["prob_sad"], old_test["prob_sad"]), model
        assert np.allclose(new_val["prob_sad"], old_val["prob_sad"]), model


def test_cpu_rerun_gives_identical_prediction_bytes(trained_root, tmp_path):
    cell, fold = "D1", 2
    labels, ids, folds, manifest, dropped = _context()
    v3cfg = config.per_task_v3
    ckpt = tmp_path / "cnn.pth"
    cnn_test, cnn_val = v3.fit_cnn_cell(
        cell, fold, ids, labels, folds, manifest, dropped, "cpu", ckpt, config
    )
    hmm_test, hmm_val, _ = v3.fit_hmm_cell(
        cell, fold, ids, labels, folds, manifest, dropped, "cpu", ckpt, config
    )
    for model, frame, path_of in (
        (v3cfg.cnn_model, cnn_test, v3.part_path),
        (v3cfg.cnn_model, cnn_val, v3.val_part_path),
        (v3cfg.hmm_model, hmm_test, v3.part_path),
        (v3cfg.hmm_model, hmm_val, v3.val_part_path),
    ):
        reference = path_of(model, cell, fold, config)
        rerun = tmp_path / Path(reference).name
        v3.write_part(frame, rerun)
        assert rerun.read_bytes() == reference.read_bytes(), rerun.name


@pytest.fixture(scope="module")
def combined_root(trained_root):
    """combine, evaluate and report run once on the trained root."""
    root, _ = trained_root
    v3.step_combine(config)
    v3.step_evaluate(config)
    v3.step_report(config)
    return root


def _predictions():
    return _read(v3.output_dir(config) / "predictions.csv")


def test_cell_selection_ignores_test_labels(combined_root, tmp_path, monkeypatch):
    src = v3.output_dir(config)
    reference = (src / "selected_cell.csv").read_bytes()
    original = _predictions()
    monkeypatch.setattr(config.paths, "output_root", tmp_path)
    dst = v3.output_dir(config)
    (dst / "parts").mkdir(parents=True)
    flip = {"HAPPY": "SAD", "SAD": "HAPPY"}
    for path in (src / "parts").glob("*.csv"):
        frame = _read(path)
        if not path.name.endswith(".val.csv"):
            frame["label"] = frame["label"].map(flip)
        frame.to_csv(dst / "parts" / path.name, index=False)
    v3.step_combine(config)
    assert (dst / "selected_cell.csv").read_bytes() == reference
    flipped = _read(dst / "predictions.csv")
    assert (flipped["label"] != original["label"]).all()


def test_fuse_is_the_mean_of_per_cell_prob_sad(combined_root):
    preds = _predictions()
    v3cfg = config.per_task_v3
    rng = np.random.default_rng(3)
    sets = v3.fuse_cells(config)
    assert set(sets) == {name for name, _ in v3cfg.fuse_sets}
    for per_cell, fused in (
        (v3cfg.cnn_model, v3cfg.fused_cnn_model),
        (v3cfg.hmm_model, v3cfg.fused_hmm_model),
    ):
        for name, cells in sets.items():
            group = preds[(preds["model"] == fused) & (preds["feature_set"] == name)]
            for pid in rng.choice(group["participant_id"].to_numpy(), 3, False):
                rows = preds[
                    (preds["model"] == per_cell)
                    & (preds["participant_id"] == pid)
                    & preds["feature_set"].isin(cells)
                ]
                assert sorted(rows["feature_set"]) == sorted(cells)
                got = group.loc[group["participant_id"] == pid, "prob_sad"].item()
                assert got == pytest.approx(rows["prob_sad"].mean())


def test_fuse_cells_refuses_an_empty_set(monkeypatch):
    monkeypatch.setattr(config.per_task_v3, "cells", ("D1", "CS1"))
    with pytest.raises(ValueError, match="'word'"):
        v3.fuse_cells(config)


def test_predictions_cover_each_participant_once_per_group(combined_root):
    preds = _predictions()
    labels, ids, folds = v3.context(config)
    v3cfg = config.per_task_v3
    fold_of = folds.set_index(folds["participant_id"].astype(str))["outer_fold"]
    groups = preds.groupby(["model", "feature_set"])
    n_cells = len(v3cfg.cells)
    assert len(groups) == 4 * n_cells + 2 * len(v3cfg.fuse_sets) + 1
    assert tuple(preds.columns) == v3.PRED_COLUMNS
    for (model, fs), group in groups:
        assert model in v3.v3_models(config), model
        assert sorted(group["participant_id"]) == sorted(ids), (model, fs)
        assert set(group["fold"]) == set(range(config.cv.n_splits))
        expected = fold_of.loc[group["participant_id"]].to_numpy()
        assert (group["fold"].to_numpy() == expected).all(), (model, fs)
    keys = ["model", "feature_set", "fold", "participant_id"]
    assert preds[keys].equals(preds.sort_values(keys)[keys].reset_index(drop=True))
    selected = _read(v3.output_dir(config) / "selected_cell.csv")
    assert list(selected["fold"]) == list(range(config.cv.n_splits))
    assert set(selected["cell"]) <= set(v3cfg.cells)
    inner = _read(v3.output_dir(config) / "inner_val_predictions.csv")
    assert set(inner["model"]) == {v3cfg.cnn_model, v3cfg.hmm_model}


def test_per_task_table_keeps_form_order(combined_root):
    table = pd.read_csv(v3.output_dir(config) / "per_task_table.csv")
    assert list(table["cell"]) == list(config.per_task_v3.cells)
    assert list(table["family"]) == ["drawing", "word", "cursive"]
    assert list(table["task"]) == ["circles", "content", "cursive 1"]
    assert list(table["style"].fillna("")) == ["", "LH", ""]
    for model in v3.per_cell_models(config):
        for m in v3.TABLE_METRICS:
            for suffix in ("", "_ci_low", "_ci_high"):
                assert f"{model}_{m}{suffix}" in table.columns
    assert "imputed_crops" in table.columns and "majority_baseline" in table.columns


def test_evaluate_refuses_while_d2_undecided(combined_root, monkeypatch):
    monkeypatch.setattr(config.per_task_v3, "n_cnn_permutations", -1)
    with pytest.raises(RuntimeError, match="D2 undecided: do plan step 7.1 first"):
        v3.step_evaluate(config)


def test_paired_marks_missing_v1_group(combined_root):
    paired = pd.read_csv(v3.output_dir(config) / "paired.csv", keep_default_na=False)
    v3cfg = config.per_task_v3
    headline = paired[paired["comparison"] == "headline"]
    assert list(headline["reference"]) == [
        f"{m}/{fs}" for m, fs in v3cfg.paired_references
    ]
    assert (headline["note"] == "").all()
    family = paired[paired["comparison"] == "family"]
    assert len(family) == len(v3cfg.family_references)
    missing = family[family["model"] == "cnn_head_v3_fused/drawing"]
    assert list(missing["note"]) == [v3.NO_V1]
    assert list(missing["delta_macro_f1"]) == [""]
    assert (family[family["model"] != "cnn_head_v3_fused/drawing"]["note"] == "").all()
    metrics = pd.read_csv(v3.output_dir(config) / "metrics.csv")
    assert set(metrics["model"]) <= set(v3.v3_models(config))


def test_report_header_figures_and_permutation_status(combined_root, capsys):
    text = v3.report_path(config).read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "EXPLORATORY, POST HOC (Deviation 6). Not part of the primary claim; "
        "the primary model remains CNN-HMM v1 (cnn_hmm_fused/fused)."
    )
    assert "not run (D2 = 0)" in text
    assert v3.NO_V1 in text
    assert "about 217" in text and "about 39" in text
    for name in ("per_task_forest.png", "per_task_heatmap.png"):
        path = v3.figure_path(name, config)
        assert path.is_file() and path.name.startswith("fig_v3_")
    v3.step_evaluate(config)
    out = capsys.readouterr().out
    assert all(line.startswith("Written: ") for line in out.splitlines())
