"""
Crop dataset and loaders built from the processed manifest and labels — Stage E.

CropDataset joins DATASET/metadata/processed_manifest.csv with labels.csv
on participant_id, keeps only the requested participants and task families,
and excludes crops marked dropped in qc_log.csv. Participant IDs, never crop
indices, decide membership, so a participant's crops always land on one side
of a split. Images are read as grayscale (RGB files converted in memory).

build_loaders makes one train and one validation DataLoader for a single
task family (canvas sizes differ by family, so families never share a
batch), with the compliant transforms from src/data/transforms.py.

LegacyPipelineError is raised by the quarantined legacy training entry
points in src/training (crop-level splits); see legacy_pipeline_error.
"""

from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.transforms import get_eval_transform, get_train_transform
from src.preprocessing.pipeline import _load_qc_dropped
from src.training.splits import assert_no_leakage
from src.utils.config import config

MANIFEST_COLUMNS = ("participant_id", "cell", "task_family", "processed_path")


class LegacyPipelineError(RuntimeError):
    """Raised by entry points of the legacy, non-compliant training stack."""


def legacy_pipeline_error(entry: str) -> LegacyPipelineError:
    """The error every quarantined legacy entry point raises."""
    return LegacyPipelineError(
        f"{entry} is part of the legacy pipeline and is disabled: it splits "
        "crops (not participants), with no participant grouping or "
        "assert_no_leakage (CLAUDE.md Non-Negotiable 1). Use the Stage E/F "
        "replacement (folds.csv, src/training/splits.py, run_*.py)."
    )


class CropDataset(Dataset):
    """Processed crops of the given participants, with their labels.

    manifest: processed_manifest.csv rows (participant_id, cell, task_family,
        processed_path). labels: labels.csv rows (participant_id and the
        config.labeling.label_column). participant_ids: who to include; every
        ID must have crops and a label. task_families: families to keep, or
        None for all. transform: applied to the grayscale PIL image (default
        ToTensor). dropped: {(participant_id, cell)} to exclude (qc_log.csv).

    Items are dicts: image (tensor), label (int via label_to_index),
    participant_id, task_family, cell. The joined rows are in self.table,
    sorted by participant_id then cell.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        labels: pd.DataFrame,
        participant_ids: list,
        task_families: list | None,
        transform=None,
        dropped: set | None = None,
        label_column: str | None = None,
        label_to_index: dict | None = None,
    ):
        label_column = label_column or config.labeling.label_column
        self.label_to_index = label_to_index or config.data.label_to_index
        self.transform = transform

        missing_cols = set(MANIFEST_COLUMNS) - set(manifest.columns)
        if missing_cols:
            raise ValueError(f"manifest lacks columns: {sorted(missing_cols)}")
        if {"participant_id", label_column} - set(labels.columns):
            raise ValueError(f"labels need participant_id and {label_column!r}")

        ids = {str(pid) for pid in participant_ids}
        crops = manifest.loc[:, list(MANIFEST_COLUMNS)].copy()
        crops["participant_id"] = crops["participant_id"].astype(str)
        unknown = ids - set(crops["participant_id"])
        if unknown:
            raise ValueError(f"participants with no crops: {sorted(unknown)}")

        crops = crops[crops["participant_id"].isin(ids)]
        if task_families is not None:
            crops = crops[crops["task_family"].isin(task_families)]
        if dropped:
            keep = [
                (pid, cell) not in dropped
                for pid, cell in zip(crops["participant_id"], crops["cell"])
            ]
            crops = crops[keep]

        lab = labels.loc[:, ["participant_id", label_column]].copy()
        lab["participant_id"] = lab["participant_id"].astype(str)
        if lab["participant_id"].duplicated().any():
            raise ValueError("labels has duplicate participant_id rows")
        unlabelled = ids - set(lab["participant_id"])
        if unlabelled:
            raise ValueError(f"participants with no label: {sorted(unlabelled)}")

        table = crops.merge(lab, on="participant_id", how="left")
        bad = set(table[label_column]) - set(self.label_to_index)
        if bad:
            raise ValueError(f"labels not in label_to_index: {sorted(map(str, bad))}")
        table["label"] = table[label_column]
        table["label_index"] = table[label_column].map(self.label_to_index)
        self.table = table.sort_values(["participant_id", "cell"]).reset_index(
            drop=True
        )

    @property
    def participant_ids(self) -> set:
        return set(self.table["participant_id"])

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict:
        row = self.table.iloc[idx]
        with Image.open(row["processed_path"]) as img:
            image = img.convert("L")
        image = (
            self.transform(image) if self.transform else transforms.ToTensor()(image)
        )
        return {
            "image": image,
            "label": int(row["label_index"]),
            "participant_id": row["participant_id"],
            "task_family": row["task_family"],
            "cell": row["cell"],
        }


def labels_csv_path(cfg):
    """labels.csv path: LabelingConfig.output_csv, else metadata_dir/labels.csv."""
    return cfg.labeling.output_csv or cfg.paths.metadata_dir / "labels.csv"


def load_tables(cfg) -> tuple:
    """(processed manifest, labels, dropped set) read from the data root."""
    meta = cfg.paths.metadata_dir
    manifest = pd.read_csv(
        meta / "processed_manifest.csv", dtype={"participant_id": str}
    )
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    return manifest, labels, _load_qc_dropped(meta)


def build_loaders(train_ids: list, val_ids: list, task_family: str, cfg) -> tuple:
    """(train DataLoader, validation DataLoader) for one task family.

    Calls assert_no_leakage, so a participant in both ID lists raises
    LeakageError. Train uses the augmenting transform and a seeded shuffle;
    validation uses the eval transform, unshuffled.
    """
    assert_no_leakage(train_ids, test_ids=[], val_ids=val_ids)

    manifest, labels, dropped = load_tables(cfg)
    families = [task_family]
    train_ds = CropDataset(
        manifest,
        labels,
        train_ids,
        families,
        transform=get_train_transform(cfg, task_family),
        dropped=dropped,
    )
    val_ds = CropDataset(
        manifest,
        labels,
        val_ids,
        families,
        transform=get_eval_transform(cfg, task_family),
        dropped=dropped,
    )
    generator = torch.Generator().manual_seed(cfg.training.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    return train_loader, val_loader
