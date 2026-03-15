"""
Training Module
Handles model training, validation, and checkpointing.
"""

from pathlib import Path
from typing import Dict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer class for the CNN component of the hybrid model.

    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        checkpoint_dir: str = "models",
        device: torch.device = None,
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict:
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"Starting training for {self.epochs} epochs...")
        print("-" * 50)

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            self.scheduler.step(val_metrics["loss"])

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.4f}"
                )

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.save_checkpoint("best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        print("-" * 50)
        print("Training complete!")
        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        print(f"Checkpoint loaded: {path}")


if __name__ == "__main__":
    print("Trainer module ready")
