"""
Training Module — Phase 9 — LEGACY ENTRY POINT QUARANTINED.

run_training (and the CLI) raise LegacyPipelineError: it trained on a
crop-level folder split, not participant-level folds, and its body has been
removed. Trainer.fit is the protocol-exact CNN fine-tuning loop used by
src/training/run_cnn.py; the feature helpers are kept for reuse.

Trains CNN (ResNet18 backbone), CNN→HMM pipeline, and LR baseline.

Primary metric: F1 macro (early stopping and logging).
Secondary: accuracy.

"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataloader import (
    CropDataset,
    LegacyPipelineError,
    legacy_pipeline_error,
)
from src.models.cnn import EmotionCNN
from src.training.aggregate import aggregate_crops

LOG_FIELDS = [
    "run_ts",
    "epoch",
    "model",
    "train_loss",
    "train_f1_macro",
    "val_loss",
    "val_f1_macro",
]


# ─── handcrafted features (LR baseline) ──────────────────────────────────────


def _image_features(img_path: str) -> np.ndarray:
    """Return [mean_intensity, pixel_density, slant_angle] for one image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(3, dtype=np.float32)

    mean_intensity = float(img.mean()) / 255.0
    pixel_density = float((img < 128).sum()) / float(img.size)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slant = 0.0
    if contours:
        pts = np.concatenate(contours)
        if len(pts) >= 5:
            angle = cv2.minAreaRect(pts)[-1]
            if angle < -45:
                angle += 90.0
            slant = angle / 90.0  # normalise to [-0.5, 0.5]

    return np.array([mean_intensity, pixel_density, slant], dtype=np.float32)


def gather_lr_features(
    dataset: CropDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3-D handcrafted features from every crop in a dataset."""
    feats = np.array(
        [_image_features(path) for path in dataset.table["processed_path"]],
        dtype=np.float32,
    )
    labels = dataset.table["label_index"].to_numpy()
    return feats, labels


# ─── sequence features (CNN → HMM) ───────────────────────────────────────────


def extract_sequences(
    cnn: EmotionCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Extract spatial sequence features for HMM training / inference.

    Returns:
        features : (total_timesteps, num_features)
        labels   : (n_samples,)
        lengths  : per-sample sequence lengths
    """
    cnn.eval()
    all_seqs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_lengths: list[int] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            seq = cnn.extractor.extract_spatial_features(images)  # (B, T, F)
            for i in range(seq.shape[0]):
                s = seq[i].cpu().numpy()
                all_seqs.append(s)
                all_lengths.append(s.shape[0])
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_seqs, axis=0),
        np.concatenate(all_labels),
        all_lengths,
    )


# ─── Trainer (CNN) ───────────────────────────────────────────────────────────


class Trainer:
    """Fine-tunes an EmotionCNN as specified in EVALUATION_PROTOCOL.md section 5.

    fit(train_loader, val_loader):
      * AdamW with two parameter groups: trainable backbone parameters at
        TrainingConfig.lr_backbone, head (extractor.fc + classifier) at
        lr_head; weight_decay from TrainingConfig. The "simple" backbone is
        trained from scratch, so its parameters use lr_head.
      * Cosine annealing over TrainingConfig.epochs.
      * Class-balanced cross-entropy (class_weight="balanced": inverse
        frequency of the training crops' labels), else unweighted.
      * After every epoch, the validation crops are aggregated per
        participant (aggregate_crops, mean_prob) and scored by participant
        macro-F1; the best epoch (improvement > min_delta) is kept, and
        training stops after `patience` epochs without improvement. The best
        state is restored at the end.

    Loaders yield CropDataset dict batches (image, label, participant_id,
    task_family, cell). The validation loader must hold inner-validation
    participants only.
    """

    def __init__(self, model: EmotionCNN, cfg=None, device: str | None = None):
        from src.features.embeddings import resolve_device
        from src.utils.config import config

        self.cfg = cfg or config
        self.device = torch.device(resolve_device(device))
        self.model = model.to(self.device)
        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.best_score: float | None = None

    def _optimizer(self) -> optim.Optimizer:
        tcfg = self.cfg.training
        extractor = self.model.extractor
        head = list(extractor.fc.parameters()) + list(
            self.model.classifier.parameters()
        )
        backbone = [p for p in extractor.backbone.parameters() if p.requires_grad]
        lr_backbone = (
            tcfg.lr_head if self.cfg.cnn.backbone == "simple" else tcfg.lr_backbone
        )
        groups = [{"params": head, "lr": tcfg.lr_head}]
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone})
        return optim.AdamW(groups, weight_decay=tcfg.weight_decay)

    def _criterion(self, loader: DataLoader) -> nn.Module:
        if self.cfg.training.class_weight != "balanced":
            return nn.CrossEntropyLoss()
        labels = loader.dataset.table["label_index"].to_numpy()
        counts = np.bincount(labels, minlength=self.cfg.cnn.num_classes).astype(float)
        weights = len(labels) / (len(counts) * np.maximum(counts, 1.0))
        return nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

    def train_epoch(self, loader, optimizer, criterion) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            n += images.size(0)
        return total / max(n, 1)

    def predict_crops(self, loader: DataLoader) -> pd.DataFrame:
        """Crop-level prob_sad for every crop in loader (eval mode, no grad)."""
        sad = self.cfg.data.label_to_index["SAD"]
        index_to_label = {v: k for k, v in self.cfg.data.label_to_index.items()}
        self.model.eval()
        rows = []
        with torch.no_grad():
            for batch in loader:
                probs = torch.softmax(self.model(batch["image"].to(self.device)), dim=1)
                for i, prob in enumerate(probs[:, sad].cpu().tolist()):
                    rows.append(
                        {
                            "participant_id": batch["participant_id"][i],
                            "task_family": batch["task_family"][i],
                            "cell": batch["cell"][i],
                            "label": index_to_label[int(batch["label"][i])],
                            "prob_sad": prob,
                        }
                    )
        return pd.DataFrame(rows)

    def participant_macro_f1(self, loader: DataLoader) -> float:
        crops = self.predict_crops(loader)
        people = aggregate_crops(crops, self.cfg.aggregate.method)
        return float(
            f1_score(people["label"], people["pred"], average="macro", zero_division=0)
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        """Train with early stopping on inner-val participant macro-F1."""
        tcfg = self.cfg.training
        optimizer = self._optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(tcfg.epochs, 1)
        )
        criterion = self._criterion(train_loader)
        best_state, waited = None, 0
        for epoch in range(1, tcfg.epochs + 1):
            loss = self.train_epoch(train_loader, optimizer, criterion)
            scheduler.step()
            score = self.participant_macro_f1(val_loader)
            self.history.append(
                {"epoch": epoch, "train_loss": loss, "val_participant_macro_f1": score}
            )
            if self.best_score is None or score > self.best_score + tcfg.min_delta:
                self.best_score, self.best_epoch, waited = score, epoch, 0
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
            else:
                waited += 1
                if waited >= tcfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def save(self, path: Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_val_participant_macro_f1": self.best_score,
                "history": self.history,
                **(extra or {}),
            },
            path,
        )


# ─── log helpers ─────────────────────────────────────────────────────────────


def _log_row(
    run_ts: str,
    epoch: int | str,
    model: str,
    train_loss: float | str = "",
    train_f1: float | str = "",
    val_loss: float | str = "",
    val_f1: float | str = "",
) -> dict:
    def _fmt(v: float | str) -> str:
        return f"{v:.6f}" if isinstance(v, float) else v

    return {
        "run_ts": run_ts,
        "epoch": epoch,
        "model": model,
        "train_loss": _fmt(train_loss),
        "train_f1_macro": _fmt(train_f1),
        "val_loss": _fmt(val_loss),
        "val_f1_macro": _fmt(val_f1),
    }


def _write_log(log_path: Path, rows: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ─── main orchestration ───────────────────────────────────────────────────────


def run_training(epochs: int | None = None) -> None:
    """
    Full Phase 9 training run:
      1. Train CNN  (ResNet18 or custom, per config)
      2. Extract CNN sequences → train HMM
      3. Compute handcrafted features → train LR baseline
      4. Write results/training_log.csv

    LEGACY (disabled): raises LegacyPipelineError.
    """
    raise legacy_pipeline_error("run_training")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 9: train CNN-HMM and LR baseline."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config epochs (e.g. --epochs 2 for a smoke test).",
    )
    args = parser.parse_args()
    try:
        run_training(epochs=args.epochs)
    except LegacyPipelineError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
