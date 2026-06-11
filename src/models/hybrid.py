"""
Hybrid CNN-HMM Model
Combines CNN feature extraction with HMM classification.
"""

from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from .cnn import EmotionCNN
from .hmm import HMMClassifier


class HybridCNNHMM:
    """
    Hybrid model combining CNN and HMM for emotion recognition.

    Pipeline:
    1. CNN extracts spatial features from handwriting images
    2. Features are organized as left-to-right sequences
    3. HMM classifies the sequential patterns
    4. Output: Emotion prediction (HAPPY/SAD) with confidence
    """

    def __init__(
        self,
        cnn_features: int = 256,
        hmm_states: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        input_channels: int = 1,
        dropout_rate: float = 0.5,
        device: torch.device = None,
    ):
        self.cnn_features = cnn_features
        self.hmm_states = hmm_states
        self.image_size = image_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.cnn = EmotionCNN(
            input_channels=input_channels,
            num_features=cnn_features,
            num_classes=2,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.hmm = HMMClassifier(
            n_states=hmm_states,
        )

        self.is_trained = False

    def extract_all_features(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Extract sequence features from all samples in a DataLoader.

        Returns:
            features: Concatenated feature array (n_total_timesteps, 256)
            labels: Per-sample label array (n_samples,)
            lengths: Length of each sequence (n_samples,)
        """
        self.cnn.eval()
        all_features = []
        all_labels = []
        all_lengths = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                # (batch, seq_len, 256)
                seq_feats = self.cnn.extract_sequence_features(images)

                for i in range(seq_feats.shape[0]):
                    seq = seq_feats[i].cpu().numpy()  # (seq_len, 256)
                    all_features.append(seq)
                    all_lengths.append(seq.shape[0])

                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels)
        return features, labels, all_lengths

    def extract_global_features(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract single-vector global features (for simpler HMM mode)."""
        self.cnn.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                features = self.cnn.extract_features(images)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.concatenate(all_features), np.concatenate(all_labels)

    def predict(self, image: torch.Tensor) -> Tuple[str, float]:
        """
        Predict emotion for a single image tensor.

        Args:
            image: (1, channels, H, W) or (channels, H, W) tensor

        Returns:
            (emotion_string, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        self.cnn.eval()
        with torch.no_grad():
            seq_feats = self.cnn.extract_sequence_features(image)

        seq = seq_feats[0].cpu().numpy()  # (seq_len, 256)
        predictions, confidences = self.hmm.predict(seq, lengths=[seq.shape[0]])

        emotion = HMMClassifier.LABEL_TO_EMOTION[predictions[0]]
        return emotion, confidences[0]

    def save(self, cnn_path: str, hmm_path: str):
        """Save the complete hybrid model."""
        torch.save(self.cnn.state_dict(), cnn_path)
        self.hmm.save(hmm_path)

    def load(self, cnn_path: str, hmm_path: str):
        """Load a trained hybrid model."""
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.hmm.load(hmm_path)
        self.is_trained = True


if __name__ == "__main__":
    model = HybridCNNHMM(
        cnn_features=256,
        hmm_states=4,
        image_size=(224, 224),
    )
    print("Hybrid CNN-HMM Model initialized")
    print(f"CNN features: {model.cnn_features}")
    print(f"HMM states: {model.hmm_states}")
    print(f"Device: {model.device}")
