"""
CNN-HMM, the titled method, on layer3 sequences of the F4 checkpoints — Stage F.

For each task family (word, cursive; drawing only with --include-drawing,
reported as secondary and never fused) and outer fold k:

  1. Load the fine-tuned CNN checkpoint models/cnn/<run>/<family>/fold_<k>.pth
     (no retraining). Rebuild the outer split and the inner split with the
     same seed as run_cnn (TrainingConfig.seed + k) and assert that the
     inner-train / inner-val / test IDs equal those stored in the checkpoint.
  2. Extract layer3 column sequences (eval mode, eval transform) for the
     inner-train, inner-val, and outer test (+ middle-band) crops.
  3. Select the HMM setting (HybridConfig n_states x pca x topology grids)
     by inner-val participant macro-F1 (uncalibrated sigmoid of the
     decision score, aggregated with mean_prob), each candidate fit on
     inner-train with HybridConfig.selection_restarts restarts.
  4. Refit the winner on inner-train with HMMConfig.n_restarts, Platt-
     calibrate it on inner-val crop scores, predict the outer test crops,
     and aggregate per participant (mean_prob).

Outer test and middle-band participants are never used for fitting,
selection, or calibration; assert_no_leakage runs on the outer and inner
splits of every fold.

LIMITATION: the HMMs are fit on features from images the CNN itself was
trained on (inner-train), so those features fit the training labels better
than unseen data would. Selecting the HMM setting and calibrating on the
inner-val participants -- whose images the CNN did not train on (they were
used only for its early stopping) -- mitigates this; selection and
calibration share that set, which makes the calibration mildly optimistic.

Outputs (run name <analysis>[_pilot], from the matching run_cnn run):

    models/hmm/<run>/<family>/fold_<k>.joblib
    results/hybrid/<run>/parts/<family>_fold<k>.csv     crop predictions
    results/hybrid/<run>/parts/<family>_fold<k>.json    selection record
    results/hybrid/<run>/crop_predictions.csv
    results/hybrid/<run>/predictions.csv   "cnn_hmm" (feature_set = family),
                                           "cnn_hmm_fused" ("fused": word +
                                           cursive, equal weights)
    results/hybrid/<run>/selection.json
    results/hybrid/<run>/run_config.json   resume refuses a mismatch

Resume as in run_cnn: finished (family, fold) pairs are skipped.

    python -m src.training.run_hybrid [--analysis primary|full]
        [--subset pilot] [--include-drawing] [--predict-middle] [--device cuda]
        [--stage smoke|pilot|full]

Each run appends point metrics to results/RUN_LOG.csv (src/utils/run_log.py).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataloader import CropDataset, labels_csv_path, load_tables
from src.data.transforms import get_eval_transform
from src.features.embeddings import resolve_device
from src.models.hmm import HMMClassifier
from src.models.hybrid import cnn_checkpoint_path, crop_sequences, hmm_path, load_cnn
from src.training.aggregate import aggregate_crops, fuse_task_families
from src.training.run_baselines import PRED_COLUMNS
from src.training.run_cnn import check_resume, run_name, shuffled_labels
from src.training.splits import (
    SUBSETS,
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    folds_path,
    inner_split,
    load_folds,
    middle_band_ids,
    restrict_to_subset,
)
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic
from src.utils.run_log import STAGES, command_line, log_predictions, resolve_stage
from src.utils.seed import set_seed

MODEL = "cnn_hmm"
FUSED_MODEL = "cnn_hmm_fused"


@dataclass
class SequenceSet:
    """Crop sequences of one participant set: meta rows align with seqs."""

    meta: pd.DataFrame
    seqs: list

    @property
    def ids(self) -> set:
        return set(self.meta["participant_id"].astype(str))

    @property
    def labels(self) -> np.ndarray:
        return self.meta["label_index"].to_numpy()


def extract_set(cnn, manifest, labels, ids, family, dropped, device) -> SequenceSet:
    ds = CropDataset(
        manifest,
        labels,
        ids,
        [family],
        transform=get_eval_transform(config, family),
        dropped=dropped,
    )
    loader = DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    meta, seqs = crop_sequences(cnn.to(device), loader, device)
    return SequenceSet(meta, seqs)


def _participant_macro_f1(meta: pd.DataFrame, prob_sad: np.ndarray) -> float:
    crops = meta[["participant_id", "task_family", "label"]].assign(prob_sad=prob_sad)
    people = aggregate_crops(crops, config.aggregate.method)
    return float(
        f1_score(people["label"], people["pred"], average="macro", zero_division=0)
    )


def _grid() -> list:
    h = config.hybrid
    return [
        {"n_states": n, "pca_components": p, "topology": t}
        for n, p, t in itertools.product(h.n_states_grid, h.pca_grid, h.topology_grid)
    ]


def select_hmm(train: SequenceSet, val: SequenceSet, seed: int) -> tuple:
    """(winning setting, table of all candidates) by inner-val participant F1.

    Candidates are fit on `train` only and scored on `val` only. Ties keep
    the first setting in grid order.
    """
    table, best = [], None
    for setting in _grid():
        cfg = replace(
            config.hmm, n_restarts=config.hybrid.selection_restarts, **setting
        )
        try:
            clf = HMMClassifier(cfg, seed).fit(train.seqs, train.labels)
            scores = clf.decision_scores(val.seqs)
            f1 = _participant_macro_f1(val.meta, 1.0 / (1.0 + np.exp(-scores)))
            table.append({**setting, "inner_val_macro_f1": f1})
        except (ValueError, RuntimeError) as exc:
            table.append({**setting, "inner_val_macro_f1": None, "error": str(exc)})
            continue
        if best is None or f1 > best["inner_val_macro_f1"]:
            best = {**setting, "inner_val_macro_f1": f1}
    if best is None:
        errors = sorted({row["error"] for row in table if "error" in row})
        raise RuntimeError(f"no HMM setting could be fit: {errors[:3]}")
    return best, table


def fit_final(train: SequenceSet, val: SequenceSet, setting: dict, seed: int):
    """Refit the winner on train with full restarts; Platt-calibrate on val."""
    params = {k: setting[k] for k in ("n_states", "pca_components", "topology")}
    clf = HMMClassifier(replace(config.hmm, **params), seed)
    clf.fit(train.seqs, train.labels)
    clf.fit_calibrator(clf.decision_scores(val.seqs), val.labels)
    return clf


def _check_split(state: dict, inner_train, inner_val, test, ckpt: Path) -> None:
    stored = {k: state.get(k) for k in ("inner_train_ids", "inner_val_ids", "test_ids")}
    if any(v is None for v in stored.values()):
        raise RuntimeError(f"{ckpt} has no recorded split IDs; re-run run_cnn")
    rebuilt = {
        "inner_train_ids": list(inner_train),
        "inner_val_ids": list(inner_val),
        "test_ids": list(test),
    }
    for key, ids in rebuilt.items():
        if sorted(map(str, stored[key])) != sorted(map(str, ids)):
            raise RuntimeError(
                f"{ckpt}: {key} differ from the split rebuilt here; the CNN run "
                "used different folds, labels, seed, or analysis set"
            )


def _fold(
    family,
    fold,
    ids,
    middle_ids,
    folds,
    labels,
    manifest,
    dropped,
    name,
    device,
    ckpt: Path | None = None,
    shuffle_seed: int | None = None,
    save: bool = True,
) -> tuple:
    """CNN-HMM for one (family, fold): (eval crop predictions, selection).

    By default the CNN checkpoint of run `name` is used, the HMMs are fit on
    the true labels, and the winner is saved. src.analysis.permutation
    passes its own checkpoint, a shuffle_seed (outer-train labels permuted
    among themselves, as in run_cnn) and save=False. Eval crops always keep
    their true labels.
    """
    train, test = fold_ids(folds, fold, ids)
    middle = fold_ids(folds, fold, middle_ids)[1] if middle_ids else []
    assert_no_leakage(train, test, middle)
    seed = config.training.seed + fold
    inner_train, inner_val = inner_split(
        train, labels, config.cv.inner_val_fraction, seed
    )
    assert_no_leakage(inner_train, inner_val, test, middle)

    ckpt = ckpt or cnn_checkpoint_path(config, name, family, fold)
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found; run src.training.run_cnn first")
    cnn, state = load_cnn(ckpt)
    _check_split(state, inner_train, inner_val, test, ckpt)

    fit_labels = (
        labels if shuffle_seed is None else shuffled_labels(labels, train, shuffle_seed)
    )
    set_seed(seed)
    sets = {
        role: extract_set(cnn, manifest, role_labels, role_ids, family, dropped, device)
        for role, role_ids, role_labels in (
            ("inner_train", inner_train, fit_labels),
            ("inner_val", inner_val, fit_labels),
            ("eval", test + middle, labels),
        )
    }
    best, table = select_hmm(sets["inner_train"], sets["inner_val"], seed)
    clf = fit_final(sets["inner_train"], sets["inner_val"], best, seed)
    if save:
        clf.save(hmm_path(config, name, family, fold))

    sad_col = list(clf.classes_).index(config.data.label_to_index["SAD"])
    crops = sets["eval"].meta.drop(columns=["label_index"]).copy()
    crops["prob_sad"] = clf.predict_proba(sets["eval"].seqs)[:, sad_col]
    crops["fold"] = fold
    crops["in_middle_band"] = crops["participant_id"].isin(set(middle))
    selection = {
        "family": family,
        "fold": fold,
        "secondary": family not in config.hybrid.families,
        "chosen": best,
        "grid": table,
        "refit_restarts": config.hmm.n_restarts,
        "restart_log": clf.restart_log,
        "cnn_checkpoint": str(ckpt),
    }
    return crops, selection


def _predictions(crops: pd.DataFrame, analysis: str) -> pd.DataFrame:
    per_family = [
        aggregate_crops(
            part.drop(columns=["cell"]).assign(model=MODEL, feature_set=family)
        )
        for (family, _), part in crops.groupby(["task_family", "fold"], sort=False)
    ]
    preds = pd.concat(per_family, ignore_index=True)
    fused = []
    fused_families = set(config.hybrid.families)
    for _, part in preds.groupby("fold"):
        part = part[part["feature_set"].isin(fused_families)]
        if set(part["feature_set"]) != fused_families:
            continue
        long = part[
            [
                "participant_id",
                "feature_set",
                "prob_sad",
                "fold",
                "label",
                "in_middle_band",
                "analysis",
            ]
        ].rename(columns={"feature_set": "task_family"})
        fused.append(
            fuse_task_families(long.assign(model=FUSED_MODEL)).assign(
                feature_set="fused"
            )
        )
    if fused:
        preds = pd.concat([preds] + fused, ignore_index=True)
    preds["analysis"] = analysis
    return (
        preds[list(PRED_COLUMNS)]
        .sort_values(["model", "feature_set", "fold", "participant_id"])
        .reset_index(drop=True)
    )


def _settings(analysis, subset, include_drawing, predict_middle, name) -> dict:
    cnn_run_config = config.paths.results_dir / "cnn" / name / "run_config.json"
    return {
        "analysis": analysis,
        "subset": subset,
        "include_drawing": include_drawing,
        "predict_middle": predict_middle,
        "hybrid": json.loads(json.dumps(asdict(config.hybrid))),
        "hmm": json.loads(json.dumps(asdict(config.hmm))),
        "seed": config.training.seed,
        "inner_val_fraction": config.cv.inner_val_fraction,
        "folds_sha256": hashlib.sha256(folds_path(config).read_bytes()).hexdigest(),
        "cnn_run_config_sha256": (
            hashlib.sha256(cnn_run_config.read_bytes()).hexdigest()
            if cnn_run_config.is_file()
            else None
        ),
    }


def run(
    analysis: str = "primary",
    subset: str | None = None,
    include_drawing: bool = False,
    predict_middle: bool = False,
    device: str = "auto",
    stage: str | None = None,
) -> Path:
    """CNN-HMM over all folds; return the predictions.csv path."""
    require_frozen_or_synthetic(config, "run_hybrid")
    stage = resolve_stage(stage, subset)
    if predict_middle and analysis != "primary":
        raise ValueError("--predict-middle applies to the primary analysis only")
    device = resolve_device(device)
    families = list(config.hybrid.families)
    if include_drawing:
        families.append(config.hybrid.secondary_family)

    name = run_name(analysis, subset, "resnet18")
    out_dir = config.paths.results_dir / config.hybrid.output_subdir / name
    check_resume(
        out_dir, _settings(analysis, subset, include_drawing, predict_middle, name)
    )
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    meta = config.paths.metadata_dir
    labels = pd.read_csv(labels_csv_path(config), dtype={"participant_id": str})
    participants = pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
    folds = load_folds(config)
    ids = restrict_to_subset(analysis_ids(labels, participants, analysis), subset)
    if not ids:
        raise ValueError(f"no participants in analysis {analysis!r} / {subset}")
    middle_ids = middle_band_ids(labels, participants, subset) if predict_middle else []
    manifest, _, dropped = load_tables(config)

    for family in families:
        for fold in range(config.cv.n_splits):
            part = parts_dir / f"{family}_fold{fold}.csv"
            sel = parts_dir / f"{family}_fold{fold}.json"
            if (
                part.is_file()
                and sel.is_file()
                and hmm_path(config, name, family, fold).is_file()
            ):
                print(f"resume: {family} fold {fold} already done")
                continue
            print(f"CNN-HMM: {family} fold {fold} ({device})")
            crops, selection = _fold(
                family,
                fold,
                ids,
                middle_ids,
                folds,
                labels,
                manifest,
                dropped,
                name,
                device,
            )
            crops = crops.assign(analysis=analysis, model=MODEL)
            for path, write in (
                (sel, lambda p: p.write_text(json.dumps(selection, indent=2))),
                (part, lambda p: crops.to_csv(p, index=False)),
            ):
                tmp = path.with_suffix(path.suffix + ".tmp")
                write(tmp)
                tmp.replace(path)

    crop_parts = sorted(parts_dir.glob("*_fold*.csv"))
    crops = pd.concat(
        [pd.read_csv(p, dtype={"participant_id": str}) for p in crop_parts],
        ignore_index=True,
    )
    crops.to_csv(out_dir / "crop_predictions.csv", index=False)
    selections = [json.loads(p.read_text()) for p in sorted(parts_dir.glob("*.json"))]
    (out_dir / "selection.json").write_text(json.dumps(selections, indent=2))
    preds = _predictions(crops, analysis)
    pred_path = out_dir / "predictions.csv"
    preds.to_csv(pred_path, index=False)
    print(f"Written: {out_dir} ({len(crop_parts)} family-fold parts)")
    command = command_line(
        "src.training.run_hybrid",
        analysis=analysis,
        subset=subset,
        include_drawing=include_drawing,
        predict_middle=predict_middle,
        stage=stage,
    )
    log_predictions(stage, command, analysis, subset, preds)
    return pred_path


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN-HMM on layer3 sequences.")
    parser.add_argument("--analysis", choices=("primary", "full"), default="primary")
    parser.add_argument("--subset", choices=SUBSETS, default=None)
    parser.add_argument("--include-drawing", action="store_true")
    parser.add_argument("--predict-middle", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", choices=STAGES, default=None)
    args = parser.parse_args()
    try:
        run(
            args.analysis,
            args.subset,
            args.include_drawing,
            args.predict_middle,
            args.device,
            args.stage,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
