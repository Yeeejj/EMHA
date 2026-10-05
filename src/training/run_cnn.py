"""
Fine-tuned CNN per task family across the frozen outer folds — Stage F.

For each task family (drawing, word, cursive) and outer fold k:

    train, test = fold_ids(folds, k, analysis ids)        assert_no_leakage
    inner_train, inner_val = inner_split(train, 15%, seed + k)
                                                          assert_no_leakage
    build_loaders(inner_train, inner_val, family) -> Trainer.fit
        (AdamW lr_backbone/lr_head, cosine, balanced CE, early stopping on
         inner-val participant macro-F1; best epoch restored)
    predict the test-fold crops (+ middle-band crops with --predict-middle)
    -> aggregate_crops(mean_prob) per participant

Outputs, under a per-run folder (run name <analysis>[_pilot][_simple]):

    models/cnn/<run>/<family>/fold_<k>.pth
    results/cnn/<run>/parts/<family>_fold<k>.csv    crop predictions, one file
                                                    per finished (family, fold)
    results/cnn/<run>/crop_predictions.csv
    results/cnn/<run>/predictions.csv   model "cnn_head", feature_set = family;
                                        model "cnn_head_fused", feature_set
                                        "fused" = fuse_task_families (equal
                                        weights) where all three families
                                        are done for that fold
    results/cnn/<run>/run_config.json   settings; resume refuses a mismatch

Resume (Kaggle sessions): a (family, fold) pair whose part file and
checkpoint both exist is skipped; parts are written atomically.

--shuffle-labels: null run. Within every outer fold the labels of the
outer-train participants are permuted among them (participant level, seed
TrainingConfig.seed + fold); test participants keep their true labels.
Writes to results/cnn_shuffled/ and models/cnn_shuffled/; expect chance.

--backbone simple: the 4-block CNN ablation (run name gains "_simple").

Refuses to run on real data before the protocol-frozen tag.

    python -m src.training.run_cnn [--task-family word]
        [--analysis primary|full] [--subset pilot] [--folds 0,1]
        [--backbone resnet18|simple] [--shuffle-labels] [--predict-middle]
        [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataloader import (
    CropDataset,
    build_loaders,
    labels_csv_path,
    load_tables,
)
from src.data.transforms import get_eval_transform
from src.features.embeddings import resolve_device
from src.models.cnn import EmotionCNN
from src.training.aggregate import aggregate_crops, fuse_task_families
from src.training.run_baselines import PRED_COLUMNS
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
from src.training.trainer import Trainer
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic
from src.utils.seed import set_seed

FAMILIES = ("drawing", "word", "cursive")
BACKBONES = ("resnet18", "simple")
MODEL = "cnn_head"
FUSED_MODEL = "cnn_head_fused"
CROP_COLUMNS = (
    "analysis",
    "model",
    "fold",
    "participant_id",
    "task_family",
    "cell",
    "label",
    "prob_sad",
    "in_middle_band",
)


def run_name(analysis: str, subset: str | None, backbone: str) -> str:
    name = analysis if subset is None else f"{analysis}_{subset}"
    return name if backbone == "resnet18" else f"{name}_{backbone}"


def _dirs(name: str, shuffle_labels: bool) -> tuple:
    kind = "cnn_shuffled" if shuffle_labels else "cnn"
    return config.paths.results_dir / kind / name, config.paths.models_dir / kind / name


def _run_config(analysis, subset, backbone, shuffle_labels, predict_middle, cnn_cfg):
    tcfg = config.training
    return {
        "analysis": analysis,
        "subset": subset,
        "backbone": backbone,
        "shuffle_labels": shuffle_labels,
        "predict_middle": predict_middle,
        "cnn": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in asdict(cnn_cfg).items()
        },
        "training": {
            k: getattr(tcfg, k)
            for k in (
                "batch_size",
                "epochs",
                "lr_head",
                "lr_backbone",
                "weight_decay",
                "class_weight",
                "patience",
                "min_delta",
                "seed",
            )
        },
        "cv": {
            "n_splits": config.cv.n_splits,
            "inner_val_fraction": config.cv.inner_val_fraction,
        },
        "aggregate": config.aggregate.method,
        "folds_sha256": hashlib.sha256(folds_path(config).read_bytes()).hexdigest(),
    }


def check_resume(out_dir: Path, settings: dict) -> None:
    path = out_dir / "run_config.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous != json.loads(json.dumps(settings)):
            changed = sorted(k for k in settings if previous.get(k) != settings[k])
            raise RuntimeError(
                f"{out_dir} holds a run with different settings ({changed}); "
                "use a different run or clear it deliberately before resuming."
            )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2))


def shuffled_labels(labels: pd.DataFrame, ids: list, seed: int) -> pd.DataFrame:
    """labels with the label column permuted among `ids` (participant level)."""
    col = config.labeling.label_column
    out = labels.copy()
    out["participant_id"] = out["participant_id"].astype(str)
    mask = out["participant_id"].isin(set(map(str, ids)))
    rng = np.random.default_rng(seed)
    out.loc[mask, col] = rng.permutation(out.loc[mask, col].to_numpy())
    return out


def _eval_loader(manifest, labels, ids, family, dropped) -> DataLoader:
    ds = CropDataset(
        manifest,
        labels,
        ids,
        [family],
        transform=get_eval_transform(config, family),
        dropped=dropped,
    )
    return DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )


def _fit_fold(
    family,
    fold,
    ids,
    middle_ids,
    folds,
    labels,
    manifest,
    dropped,
    cnn_cfg,
    shuffle,
    device,
    ckpt_path,
) -> pd.DataFrame:
    train, test = fold_ids(folds, fold, ids)
    middle = fold_ids(folds, fold, middle_ids)[1] if middle_ids else []
    assert_no_leakage(train, test, middle)
    seed = config.training.seed + fold
    inner_train, inner_val = inner_split(
        train, labels, config.cv.inner_val_fraction, seed
    )
    assert_no_leakage(inner_train, inner_val, test, middle)

    fit_labels = shuffled_labels(labels, train, seed) if shuffle else labels
    set_seed(seed)
    train_loader, val_loader = build_loaders(
        inner_train, inner_val, family, config, labels=fit_labels
    )
    model = EmotionCNN(cnn_cfg)
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)
    trainer.save(
        ckpt_path,
        {
            "family": family,
            "fold": fold,
            "cnn": asdict(cnn_cfg),
            # recorded so run_hybrid can assert it rebuilds the same split
            "inner_train_ids": list(inner_train),
            "inner_val_ids": list(inner_val),
            "test_ids": list(test),
            "middle_ids": list(middle),
        },
    )

    crops = trainer.predict_crops(
        _eval_loader(manifest, labels, test + middle, family, dropped)
    )
    crops["fold"] = fold
    crops["in_middle_band"] = crops["participant_id"].isin(set(middle))
    return crops


def _participant_predictions(crops: pd.DataFrame, analysis: str) -> pd.DataFrame:
    per_family = []
    for (family, _fold), part in crops.groupby(["task_family", "fold"], sort=False):
        agg = aggregate_crops(
            part.drop(columns=["cell"]).assign(model=MODEL, feature_set=family)
        )
        per_family.append(agg)
    preds = pd.concat(per_family, ignore_index=True)

    fused = []
    for fold, part in preds.groupby("fold"):
        if set(part["feature_set"]) != set(FAMILIES):
            continue  # fuse only when all three families are done
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


def run(
    task_family: str | None = None,
    analysis: str = "primary",
    subset: str | None = None,
    folds: list | None = None,
    backbone: str = "resnet18",
    shuffle_labels: bool = False,
    predict_middle: bool = False,
    device: str = "auto",
) -> Path:
    """Train/predict every requested (family, fold); return predictions.csv path."""
    require_frozen_or_synthetic(config, "run_cnn")
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}")
    if task_family is not None and task_family not in FAMILIES:
        raise ValueError(f"task_family must be one of {FAMILIES}")
    if predict_middle and analysis != "primary":
        raise ValueError("--predict-middle applies to the primary analysis only")
    families = [task_family] if task_family else list(FAMILIES)
    fold_list = list(range(config.cv.n_splits)) if folds is None else list(folds)
    device = resolve_device(device)

    cnn_cfg = replace(config.cnn, backbone=backbone)
    name = run_name(analysis, subset, backbone)
    out_dir, model_dir = _dirs(name, shuffle_labels)
    check_resume(
        out_dir,
        _run_config(
            analysis, subset, backbone, shuffle_labels, predict_middle, cnn_cfg
        ),
    )
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    meta = config.paths.metadata_dir
    labels = pd.read_csv(labels_csv_path(config), dtype={"participant_id": str})
    participants = pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
    fold_table = load_folds(config)
    ids = restrict_to_subset(analysis_ids(labels, participants, analysis), subset)
    if not ids:
        raise ValueError(f"no participants in analysis {analysis!r} / {subset}")
    middle_ids = middle_band_ids(labels, participants, subset) if predict_middle else []
    manifest, _, dropped = load_tables(config)

    for family in families:
        for fold in fold_list:
            part_path = parts_dir / f"{family}_fold{fold}.csv"
            ckpt = model_dir / family / f"fold_{fold}.pth"
            if part_path.is_file() and ckpt.is_file():
                print(f"resume: {family} fold {fold} already done")
                continue
            print(f"training: {family} fold {fold} ({device})")
            crops = _fit_fold(
                family,
                fold,
                ids,
                middle_ids,
                fold_table,
                labels,
                manifest,
                dropped,
                cnn_cfg,
                shuffle_labels,
                device,
                ckpt,
            )
            crops = crops.assign(analysis=analysis, model=MODEL)[list(CROP_COLUMNS)]
            tmp = part_path.with_suffix(".tmp")
            crops.to_csv(tmp, index=False)
            tmp.replace(part_path)

    parts = sorted(parts_dir.glob("*_fold*.csv"))
    crops = pd.concat(
        [pd.read_csv(p, dtype={"participant_id": str}) for p in parts],
        ignore_index=True,
    )
    crops.to_csv(out_dir / "crop_predictions.csv", index=False)
    preds = _participant_predictions(crops, analysis)
    pred_path = out_dir / "predictions.csv"
    preds.to_csv(pred_path, index=False)
    print(f"Written: {out_dir} ({len(parts)} family-fold parts)")
    return pred_path


def _parse_folds(text: str | None) -> list | None:
    if not text:
        return None
    return [int(x) for x in text.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuned CNN per task family.")
    parser.add_argument("--task-family", choices=FAMILIES, default=None)
    parser.add_argument("--analysis", choices=("primary", "full"), default="primary")
    parser.add_argument("--subset", choices=SUBSETS, default=None)
    parser.add_argument("--folds", default=None, help="comma-separated, e.g. 0,1")
    parser.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--predict-middle", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        run(
            task_family=args.task_family,
            analysis=args.analysis,
            subset=args.subset,
            folds=_parse_folds(args.folds),
            backbone=args.backbone,
            shuffle_labels=args.shuffle_labels,
            predict_middle=args.predict_middle,
            device=args.device,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
