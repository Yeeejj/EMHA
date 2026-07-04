"""
HMM Classifier Module
Hidden Markov Model for classifying emotion sequences.
"""

from typing import Tuple, List
import numpy as np
import joblib
from hmmlearn import hmm


class HMMClassifier:
    """
    Hidden Markov Model classifier for emotion recognition.

    Uses separate HMMs for each emotion class (HAPPY, SAD).
    Classification is done by comparing log-likelihoods.

    Supports both:
    - Single-vector features: shape (n_samples, n_features)
    - Sequence features: list of arrays, each (seq_len, n_features)
    """

    LABEL_TO_EMOTION = {0: "HAPPY", 1: "SAD"}
    EMOTION_TO_LABEL = {"HAPPY": 0, "SAD": 1}

    def __init__(
        self,
        n_states: int = 4,
        n_iter: int = 100,
        covariance_type: str = "diag"
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.models = {}

    def _create_hmm(self, n_samples: int):
        """Create a new HMM instance, capping states at sample count."""
        n_components = min(self.n_states, n_samples)
        return hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lengths: List[int] = None,
    ):
        """
        Train separate HMMs for each emotion class.

        Args:
            features: Concatenated feature array (n_total, n_features).
                      If using sequences, concatenate all sequences and
                      provide lengths.
            labels: Per-sequence labels (one label per sequence).
            lengths: Length of each sequence. If None, each row of
                     features is treated as a single-timestep sequence.
        """
        if lengths is None:
            lengths = [1] * len(labels)

        unique_labels = np.unique(labels)

        # Build index mapping from sequence index to row ranges
        seq_starts = np.cumsum([0] + list(lengths))

        for label in unique_labels:
            mask = labels == label
            seq_indices = np.where(mask)[0]

            # Gather rows belonging to this class
            class_rows = []
            class_lengths = []
            for si in seq_indices:
                start = seq_starts[si]
                end = seq_starts[si + 1]
                class_rows.append(features[start:end])
                class_lengths.append(lengths[si])

            class_features = np.concatenate(class_rows, axis=0)

            model = self._create_hmm(len(seq_indices))
            model.fit(class_features, lengths=class_lengths)

            emotion_name = self.LABEL_TO_EMOTION[label]
            self.models[emotion_name] = model
            print(
                f"  HMM trained for {emotion_name}: "
                f"{len(seq_indices)} sequences, "
                f"{model.n_components} states"
            )

    def predict(
        self,
        features: np.ndarray,
        lengths: List[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict emotion for given features.

        Args:
            features: Feature array (n_total, n_features).
            lengths: Length of each sequence. If None, each row is one
                     single-timestep sequence.

        Returns:
            (predictions, confidences) arrays.
        """
        if lengths is None:
            lengths = [1] * len(features)

        seq_starts = np.cumsum([0] + list(lengths))
        n_sequences = len(lengths)

        predictions = []
        confidences = []

        for i in range(n_sequences):
            start = seq_starts[i]
            end = seq_starts[i + 1]
            sample = features[start:end]

            scores = {}
            for emotion, model in self.models.items():
                scores[emotion] = model.score(sample, lengths=[lengths[i]])

            predicted = max(scores, key=scores.get)
            pred_label = self.EMOTION_TO_LABEL[predicted]
            predictions.append(pred_label)
            confidences.append(self._compute_confidence(scores))

        return np.array(predictions), np.array(confidences)

    def _compute_confidence(self, scores: dict) -> float:
        """Compute confidence from log-likelihood scores using softmax."""
        values = np.array(list(scores.values()))
        probs = np.exp(values - np.max(values))
        probs = probs / probs.sum()
        return float(np.max(probs))

    def save(self, path: str):
        """Save trained models."""
        joblib.dump(self.models, path)

    def load(self, path: str):
        """Load trained models."""
        self.models = joblib.load(path)


if __name__ == "__main__":
    classifier = HMMClassifier(n_states=4)
    print(f"HMM Classifier initialized")
    print(f"States: {classifier.n_states}")
