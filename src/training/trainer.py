"""
Training Module — Phase 9
Trains CNN (ResNet18 backbone), CNN→HMM pipeline, and LR baseline.

Primary metric: F1 macro (early stopping and logging).
Secondary: accuracy.

Outputs:
    models/cnn_{ts}_best.pth
    models/hmm_{ts}.pkl
    models/lr_baseline_{ts}.pkl
    results/training_log.csv

Usage:
    python -m src.training.trainer            # full run
    python -m src.training.trainer --epochs 2 # smoke test
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataloader import (
    HandwritingDataset,
    get_train_transform,
    get_val_transform,
)
from src.models.cnn import EmotionCNN
from src.models.hmm import HMMClassifier
from src.utils.config import config

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
    dataset: HandwritingDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3-D handcrafted features from every sample in a dataset."""
    feats = np.array(
        [_image_features(path) for path, _ in dataset.samples],
        dtype=np.float32,
    )
    labels = np.array([lbl for _, lbl in dataset.samples])
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
        for images, labels in loader:
            images = images.to(device)
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

        for images, labels in loader:
            images = images.to(self.device)
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
            for images, labels in loader:
                images = images.to(self.device)
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
    """
    print("=" * 60)
    print("Phase 9 - TRAINING")
    print("=" * 60)

    cfg = config
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_epochs = epochs if epochs is not None else cfg.training.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device     : {device}")
    print(f"  epochs     : {n_epochs}")
    print(f"  pretrained : {cfg.cnn.use_pretrained}")

    # ── data ─────────────────────────────────────────────────────────────
    splits_dir = Path(cfg.data.splits_dir)
    train_dir = splits_dir / "train"
    val_dir = splits_dir / "val"

    for d in (train_dir, val_dir):
        if not d.exists():
            print(f"ERROR: split directory not found: {d}")
            print("  Run Phase 8 (split_dataset) first.")
            sys.exit(1)

    image_size = cfg.preprocessing.target_size
    train_ds = HandwritingDataset(
        str(train_dir), transform=get_train_transform(image_size)
    )
    val_ds = HandwritingDataset(str(val_dir), transform=get_val_transform(image_size))

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: no images found in split directories.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── [1/3] CNN ─────────────────────────────────────────────────────────
    print("\n[1/3] CNN")
    cnn = EmotionCNN(
        input_channels=cfg.cnn.input_channels,
        num_features=cfg.cnn.num_features,
        num_classes=2,
        dropout_rate=cfg.cnn.dropout_rate,
        use_pretrained=cfg.cnn.use_pretrained,
        freeze_backbone=cfg.cnn.freeze_backbone,
    ).to(device)

    trainer = Trainer(
        model=cnn,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        epochs=n_epochs,
        patience=cfg.training.patience,
        checkpoint_dir=cfg.training.checkpoint_dir,
        device=device,
    )
    history = trainer.train(train_loader, val_loader)

    cnn_path = Path(cfg.training.checkpoint_dir) / f"cnn_{run_ts}_best.pth"
    trainer.save_checkpoint(cnn_path)
    print(f"  CNN saved : {cnn_path}")

    # ── [2/3] HMM ─────────────────────────────────────────────────────────
    print("\n[2/3] HMM")
    print("  Extracting sequence features...")
    tr_seqs, tr_seq_labels, tr_lengths = extract_sequences(cnn, train_loader, device)
    va_seqs, va_seq_labels, va_lengths = extract_sequences(cnn, val_loader, device)
    print(f"  train seqs: {tr_seqs.shape}  " f"val seqs: {va_seqs.shape}")

    hmm_clf = HMMClassifier(
        n_states=cfg.hmm.n_states,
        n_iter=cfg.hmm.n_iter,
        covariance_type=cfg.hmm.covariance_type,
    )
    hmm_clf.fit(tr_seqs, tr_seq_labels, lengths=tr_lengths)
    hmm_preds, _ = hmm_clf.predict(va_seqs, lengths=va_lengths)
    hmm_f1 = float(f1_score(va_seq_labels, hmm_preds, average="macro", zero_division=0))
    print(f"  HMM val F1 macro: {hmm_f1:.4f}")

    hmm_path = Path(cfg.training.checkpoint_dir) / f"hmm_{run_ts}.pkl"
    hmm_clf.save(str(hmm_path))
    print(f"  HMM saved : {hmm_path}")

    # ── [3/3] LR baseline ─────────────────────────────────────────────────
    print("\n[3/3] LR baseline")
    print("  Computing handcrafted features (mean_intensity, pixel_density, slant)...")
    tr_lr, tr_lr_labels = gather_lr_features(train_ds)
    va_lr, va_lr_labels = gather_lr_features(val_ds)

    lr_clf = LogisticRegression(
        max_iter=1000,
        random_state=cfg.training.random_state,
        class_weight="balanced",
    )
    lr_clf.fit(tr_lr, tr_lr_labels)
    lr_preds = lr_clf.predict(va_lr)
    lr_f1 = float(f1_score(va_lr_labels, lr_preds, average="macro", zero_division=0))
    print(f"  LR baseline val F1 macro: {lr_f1:.4f}")

    lr_path = Path(cfg.training.checkpoint_dir) / f"lr_baseline_{run_ts}.pkl"
    joblib.dump(lr_clf, str(lr_path))
    print(f"  LR saved  : {lr_path}")

    # ── training log ──────────────────────────────────────────────────────
    log_rows: list[dict] = []
    for e, (tl, tf, vl, vf) in enumerate(
        zip(
            history["train_loss"],
            history["train_f1_macro"],
            history["val_loss"],
            history["val_f1_macro"],
        )
    ):
        log_rows.append(_log_row(run_ts, e + 1, "cnn", tl, tf, vl, vf))
    log_rows.append(_log_row(run_ts, "final", "cnn_hmm", val_f1=hmm_f1))
    log_rows.append(_log_row(run_ts, "final", "lr_baseline", val_f1=lr_f1))

    log_path = Path("results") / "training_log.csv"
    _write_log(log_path, log_rows)
    print(f"\n  Training log : {log_path}")

    # ── summary ───────────────────────────────────────────────────────────
    best_cnn_f1 = max(history["val_f1_macro"])
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  CNN best val F1 macro : {best_cnn_f1:.4f}")
    print(f"  CNN-HMM val F1 macro  : {hmm_f1:.4f}")
    print(f"  LR baseline val F1    : {lr_f1:.4f}")
    print("=" * 60)


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
    run_training(epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
