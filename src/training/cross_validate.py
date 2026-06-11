"""
Cross-Validation Module
Implements stratified k-fold cross-validation for the hybrid CNN-HMM model.
"""

from typing import Dict
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..models.cnn import EmotionCNN
from ..models.hmm import HMMClassifier
from ..data.dataloader import (
    HandwritingDataset,
    TransformSubset,
    get_train_transform,
    get_val_transform,
)
from .trainer import Trainer


class CrossValidator:
    """
    Stratified K-Fold Cross-Validation for the hybrid CNN-HMM pipeline.

    Each fold:
    1. Trains CNN with classification head (early stopping)
    2. Extracts sequence features from trained CNN
    3. Trains HMM on sequence features
    4. Evaluates HMM predictions
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 10,
        cnn_features: int = 256,
        hmm_states: int = 4,
        image_size=(224, 224),
        use_pretrained: bool = False,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.cnn_features = cnn_features
        self.hmm_states = hmm_states
        self.image_size = image_size
        self.use_pretrained = use_pretrained
        self.fold_results = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cross_validate(
        self,
        dataset: HandwritingDataset,
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform k-fold cross-validation on the full hybrid pipeline.

        Args:
            dataset: HandwritingDataset (should use val_transform or None;
                     train augmentation is applied per-fold via TransformSubset)

        Returns:
            Summary dict with mean/std of metrics across folds.
        """
        labels = np.array(dataset.get_labels())
        indices = np.arange(len(labels))

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        train_transform = get_train_transform(self.image_size)
        val_transform = get_val_transform(self.image_size)

        self.fold_results = []
        all_y_true = []
        all_y_pred = []

        print(f"\nStarting {self.n_splits}-Fold Cross-Validation")
        print(
            f"Total samples: {len(labels)} "
            f"(HAPPY: {(labels == 0).sum()}, SAD: {(labels == 1).sum()})"
        )
        print("=" * 60)

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
            print(f"\n--- Fold {fold + 1}/{self.n_splits} ---")
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

            # Create subsets with proper transforms
            train_subset = TransformSubset(
                dataset, train_idx, transform=train_transform
            )
            val_subset = TransformSubset(dataset, val_idx, transform=val_transform)

            train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=self.batch_size, shuffle=False
            )

            # Step 1: Train CNN
            print("\n  Step 1: Training CNN...")
            cnn_model = EmotionCNN(
                input_channels=1,
                num_features=self.cnn_features,
                use_pretrained=self.use_pretrained,
            ).to(self.device)

            trainer = Trainer(
                model=cnn_model,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                patience=self.patience,
                device=self.device,
            )
            trainer.train(train_loader, val_loader)

            # Step 2: Extract sequence features
            print("\n  Step 2: Extracting sequence features...")
            train_features, train_labels, train_lengths = self._extract_sequences(
                cnn_model, train_loader
            )
            val_features, val_labels, val_lengths = self._extract_sequences(
                cnn_model, val_loader
            )
            print(f"  Train: {train_features.shape}, " f"Val: {val_features.shape}")

            # Step 3: Train HMM
            print("\n  Step 3: Training HMM...")
            hmm_clf = HMMClassifier(n_states=self.hmm_states)
            hmm_clf.fit(train_features, train_labels, lengths=train_lengths)

            # Step 4: Evaluate
            y_pred, confidences = hmm_clf.predict(val_features, lengths=val_lengths)

            acc = accuracy_score(val_labels, y_pred)
            prec = precision_score(val_labels, y_pred, average="macro", zero_division=0)
            rec = recall_score(val_labels, y_pred, average="macro", zero_division=0)
            f1 = f1_score(val_labels, y_pred, average="macro", zero_division=0)

            fold_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
            self.fold_results.append(fold_metrics)
            all_y_true.extend(val_labels)
            all_y_pred.extend(y_pred)

            print(
                f"\n  Fold {fold + 1} Results: "
                f"Acc={acc:.4f}, Prec={prec:.4f}, "
                f"Rec={rec:.4f}, F1={f1:.4f}"
            )

        summary = self._aggregate_results()

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        for metric, stats in summary.items():
            print(
                f"{metric.capitalize():>10}: "
                f"{stats['mean']:.4f} (+/- {stats['std']:.4f})"
            )

        return summary

    def _extract_sequences(self, cnn_model, data_loader):
        """Extract spatial sequence features from CNN."""
        cnn_model.eval()
        all_features = []
        all_labels = []
        all_lengths = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                seq_feats = cnn_model.extractor.extract_spatial_features(images)

                for i in range(seq_feats.shape[0]):
                    seq = seq_feats[i].cpu().numpy()
                    all_features.append(seq)
                    all_lengths.append(seq.shape[0])

                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels)
        return features, labels, all_lengths

    def _aggregate_results(self) -> Dict[str, Dict[str, float]]:
        """Compute mean and std across folds."""
        summary = {}

        if not self.fold_results:
            return summary

        metrics = self.fold_results[0].keys()

        for metric in metrics:
            values = [fold[metric] for fold in self.fold_results]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return summary


if __name__ == "__main__":
    print("Cross-validation module ready")
    print("Usage: CrossValidator(n_splits=5).cross_validate(dataset)")
