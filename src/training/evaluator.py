"""
Evaluation Module
Computes metrics and generates evaluation reports.
"""

from typing import Dict, List, Tuple
import numpy as np

# Uncomment when dependencies are installed:
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     confusion_matrix, classification_report
# )


class Evaluator:
    """
    Evaluator class for computing classification metrics.

    Metrics computed:
    - Accuracy
    - Precision (per class and macro)
    - Recall (per class and macro)
    - F1-score (per class and macro)
    - Confusion matrix
    """

    LABELS = ["HAPPY", "SAD"]

    def __init__(self):
        self.results = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        # accuracy = accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='macro')
        # recall = recall_score(y_true, y_pred, average='macro')
        # f1 = f1_score(y_true, y_pred, average='macro')
        # cm = confusion_matrix(y_true, y_pred)

        # Placeholder values
        self.results = {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "confusion_matrix": np.zeros((2, 2))
        }

        return self.results

    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each class."""
        per_class = {}

        for i, label in enumerate(self.LABELS):
            # precision = precision_score(y_true, y_pred, labels=[i], average=None)[0]
            # recall = recall_score(y_true, y_pred, labels=[i], average=None)[0]
            # f1 = f1_score(y_true, y_pred, labels=[i], average=None)[0]

            per_class[label] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        return per_class

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print a full classification report."""
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)

        metrics = self.evaluate(y_true, y_pred)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              HAPPY  SAD")
        cm = metrics['confusion_matrix']
        print(f"Actual HAPPY   {cm[0,0]:4.0f}  {cm[0,1]:4.0f}")
        print(f"       SAD     {cm[1,0]:4.0f}  {cm[1,1]:4.0f}")

        per_class = self.compute_per_class_metrics(y_true, y_pred)
        print(f"\nPer-Class Metrics:")
        for label, scores in per_class.items():
            print(f"  {label}:")
            print(f"    Precision: {scores['precision']:.4f}")
            print(f"    Recall:    {scores['recall']:.4f}")
            print(f"    F1-Score:  {scores['f1']:.4f}")

        print("=" * 50 + "\n")

    def save_results(self, path: str):
        """Save evaluation results to file."""
        # import json
        # with open(path, 'w') as f:
        #     json.dump(self.results, f, indent=2)
        pass


if __name__ == "__main__":
    evaluator = Evaluator()

    # Example with dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])

    evaluator.print_report(y_true, y_pred)
