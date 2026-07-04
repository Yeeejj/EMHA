"""
Phase 12 — Artifact generation: thesis-ready figures at 300 DPI.

Generates all figures referenced in Chapters 4-5 of the thesis:
    - Confusion matrices (per-fold + aggregated)
    - ROC curves with AUC
    - Per-fold accuracy/F1 bar charts
    - Dataset composition charts (label balance, gender, age)
    - Grad-CAM overlays on sample crops (run on Colab GPU)

All figures saved to FIGURES/ at 300 DPI.

Run from the project root:

    python -m src.utils.artifact_generator [--gradcam]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

FIGURES_DIR = Path("FIGURES")
DPI = 300
LABELS = ["HAPPY", "SAD"]


def _savefig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    _savefig(fig, filename)


def plot_roc_curve(
    y_true: list[int],
    y_score: list[float],
    filename: str = "roc_curve.png",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, filename)


def plot_fold_metrics(
    fold_results: list[dict],
    filename: str = "fold_metrics.png",
) -> None:
    folds = [f"Fold {i+1}" for i in range(len(fold_results))]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(folds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in fold_results]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(folds)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Fold Metrics")
    ax.legend()
    ax.axhline(0.70, color="red", linestyle="--", lw=1, label="Target 70%")
    ax.axhline(0.85, color="green", linestyle="--", lw=1, label="Target 85%")
    fig.tight_layout()
    _savefig(fig, filename)


def plot_dataset_composition(labels_csv: Path, filename: str = "dataset_composition.png") -> None:
    import csv
    rows = []
    if not labels_csv.is_file():
        print(f"  WARNING: {labels_csv} not found, skipping composition plot.")
        return

    with labels_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    label_counts = {"HAPPY": 0, "SAD": 0}
    gender_counts: dict[str, int] = {}
    for r in rows:
        lbl = r.get("label", "").strip().upper()
        if lbl in label_counts:
            label_counts[lbl] += 1
        g = r.get("gender", "").strip().upper() or "UNKNOWN"
        gender_counts[g] = gender_counts.get(g, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(label_counts.keys(), label_counts.values(), color=["#4CAF50", "#F44336"])
    axes[0].set_title("Label Distribution")
    axes[0].set_ylabel("Participants")
    for k, v in label_counts.items():
        axes[0].text(list(label_counts.keys()).index(k), v + 0.5, str(v), ha="center")

    axes[1].bar(gender_counts.keys(), gender_counts.values(), color="#2196F3")
    axes[1].set_title("Gender Distribution")
    axes[1].set_ylabel("Participants")

    fig.suptitle("FINALE Dataset Composition", fontsize=14)
    fig.tight_layout()
    _savefig(fig, filename)


def generate_all(cv_results_path: Path, labels_csv: Path, include_gradcam: bool = False) -> None:
    print("Phase 12 - ARTIFACT GENERATION")
    print("=" * 60)
    print(f"Output dir: {FIGURES_DIR.resolve()} @ {DPI} DPI")

    if cv_results_path.is_file():
        with cv_results_path.open(encoding="utf-8") as fh:
            cv_data = json.load(fh)

        if "fold_results" in cv_data:
            plot_fold_metrics(cv_data["fold_results"], "fold_metrics.png")

        if "confusion_matrix" in cv_data:
            cm = np.array(cv_data["confusion_matrix"])
            plot_confusion_matrix(cm, "Aggregated Confusion Matrix", "confusion_matrix.png")

        if "y_true" in cv_data and "y_score" in cv_data:
            plot_roc_curve(cv_data["y_true"], cv_data["y_score"], "roc_curve.png")
    else:
        print(f"  WARNING: {cv_results_path} not found — skipping CV figures.")

    plot_dataset_composition(labels_csv, "dataset_composition.png")

    if include_gradcam:
        print("  Grad-CAM: run on Colab GPU (see EMHA_Colab_Pipeline.ipynb).")

    print("\nArtifact generation complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thesis figures.")
    parser.add_argument("--gradcam", action="store_true", help="Include Grad-CAM note")
    args = parser.parse_args()

    generate_all(
        cv_results_path=Path("results/cv_results.json"),
        labels_csv=Path("DATA/METADATA/labels.csv"),
        include_gradcam=args.gradcam,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
