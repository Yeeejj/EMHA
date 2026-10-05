"""
Training Module — Phase 9 — LEGACY ENTRY POINT QUARANTINED.

run_training (and the CLI) raise LegacyPipelineError: it trained on a
crop-level folder split, not participant-level folds, and its body has been
removed. The Trainer class (CropDataset dict batches) and feature helpers
are kept for reuse by the Stage F run_*.py scripts.

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
    """
    Trains the CNN component of the hybrid model.

    Early stopping and checkpointing are driven by val F1 macro
    (higher is better).  Accuracy is tracked but is secondary.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        checkpoint_dir: str = "models",
        device: torch.device | None = None,
    ):
        self.model = model
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # ReduceLROnPlateau in max mode: plateau on F1 macro
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_f1_macro": [],
            "val_f1_macro": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"]
            labels_dev = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels_dev)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

        n = len(all_labels)
        return {
            "loss": total_loss / n,
            "f1_macro": f1_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            "accuracy": sum(p == t for p, t in zip(all_preds, all_labels)) / n,
        }

    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"]
                labels_dev = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels_dev)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.tolist())

        n = len(all_labels)
        return {
            "loss": total_loss / n,
            "f1_macro": f1_score(
                all_labels, all_preds, average="macro", zero_division=0
            ),
            "accuracy": sum(p == t for p, t in zip(all_preds, all_labels)) / n,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list]:
        """Full training loop. Early stopping on val F1 macro."""
        best_val_f1 = -1.0
        patience_counter = 0
        best_state: dict | None = None

        print(f"  Training for up to {self.epochs} epochs...")

        for epoch in range(self.epochs):
            train_m = self.train_epoch(train_loader)
            val_m = self.validate(val_loader)

            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_f1_macro"].append(train_m["f1_macro"])
            self.history["val_f1_macro"].append(val_m["f1_macro"])
            self.history["train_acc"].append(train_m["accuracy"])
            self.history["val_acc"].append(val_m["accuracy"])

            self.scheduler.step(val_m["f1_macro"])

            print(
                f"    epoch {epoch + 1:3d}/{self.epochs}"
                f"  train loss={train_m['loss']:.4f}"
                f" f1={train_m['f1_macro']:.4f}"
                f"  val loss={val_m['loss']:.4f}"
                f" f1={val_m['f1_macro']:.4f}"
            )

            if val_m["f1_macro"] > best_val_f1:
                best_val_f1 = val_m["f1_macro"]
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch + 1}.")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        print(f"  Best val F1 macro: {best_val_f1:.4f}")
        return self.history

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt["history"]
        print(f"Checkpoint loaded: {path}")


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
