"""
Evaluation Module
Computes metrics and generates evaluation reports.
"""

import json
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


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
        """Compute all evaluation metrics."""
        self.results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
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
            precision = precision_score(
                y_true, y_pred, labels=[i], average=None, zero_division=0
            )
            rec = recall_score(
                y_true, y_pred, labels=[i], average=None, zero_division=0
            )
            f1 = f1_score(
                y_true, y_pred, labels=[i], average=None, zero_division=0
            )

            per_class[label] = {
                "precision": float(precision[0]) if len(precision) > 0 else 0.0,
                "recall": float(rec[0]) if len(rec) > 0 else 0.0,
                "f1": float(f1[0]) if len(f1) > 0 else 0.0,
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
        cm = metrics["confusion_matrix"]
        print(f"Actual HAPPY   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
        print(f"       SAD     {cm[1, 0]:4d}  {cm[1, 1]:4d}")

        per_class = self.compute_per_class_metrics(y_true, y_pred)
        print(f"\nPer-Class Metrics:")
        for label, scores in per_class.items():
            print(f"  {label}:")
            print(f"    Precision: {scores['precision']:.4f}")
            print(f"    Recall:    {scores['recall']:.4f}")
            print(f"    F1-Score:  {scores['f1']:.4f}")

        print("=" * 50 + "\n")

        # Also print sklearn's built-in report
        print(classification_report(
            y_true, y_pred, target_names=self.LABELS, zero_division=0
        ))

    def save_results(self, path: str):
        """Save evaluation results to file."""
        serializable = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    evaluator = Evaluator()

    # Example with dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])

    evaluator.print_report(y_true, y_pred)
