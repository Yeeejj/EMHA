"""
Cross-Validation Module — LEGACY, QUARANTINED.

Crop-level StratifiedKFold for the hybrid CNN-HMM model. It does not group
crops by participant, so CrossValidator.cross_validate raises
LegacyPipelineError. Not in the CLAUDE.md module map: participant-level
folds belong to src/training/splits.py (Stage E). Kept until Stage F has
taken over the reusable parts; delete then (with the thesis author's OK).
"""

from typing import Dict
import numpy as np

import torch

from ..data.dataloader import CropDataset, legacy_pipeline_error


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
        dataset: CropDataset,
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform k-fold cross-validation on the full hybrid pipeline.

        Args:
            dataset: crops to cross-validate (legacy: split crop-wise)

        Returns:
            Summary dict with mean/std of metrics across folds.
        """
        raise legacy_pipeline_error("CrossValidator.cross_validate")

    def _extract_sequences(self, cnn_model, data_loader):
        """Extract spatial sequence features from CNN."""
        cnn_model.eval()
        all_features = []
        all_labels = []
        all_lengths = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"]
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
