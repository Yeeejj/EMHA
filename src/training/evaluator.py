"""
Evaluator — Phase 10
First and only touch of DATA/SPLITS/test/.

Outputs
-------
results/evaluation_report.csv  per-metric rows for CNN-HMM and LR baseline
results/confusion_matrix.png   confusion matrix figure (CNN-HMM, test split)
results/cv_results.json        5-fold CV mean ± std on combined train+val pool
results/model_comparison.csv   side-by-side LR vs CNN-HMM table
results/gradcam/               4 Grad-CAM overlays for thesis defence

Usage
-----
    python -m src.training.evaluator           # full evaluation
    python -m src.training.evaluator --smoke   # quick end-to-end check (2 folds)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.dataloader import HandwritingDataset, get_val_transform
from src.models.cnn import EmotionCNN
from src.models.hmm import HMMClassifier
from src.training.cross_validate import CrossValidator
from src.training.trainer import extract_sequences, gather_lr_features
from src.utils.config import config

plt.switch_backend("Agg")

_RESULTS = Path("results")
_MODELS = Path("models")
_LABELS = ["HAPPY", "SAD"]
_THRESHOLD = 0.70  # minimum F1 macro (thesis target: 70–85%)


# ─── checkpoint discovery ─────────────────────────────────────────────────────


def _latest(pattern: str) -> Path | None:
    candidates = sorted(_MODELS.glob(pattern))
    return candidates[-1] if candidates else None


# ─── model loading ────────────────────────────────────────────────────────────


def _load_cnn(path: Path, device: torch.device) -> EmotionCNN:
    cfg = config.cnn
    model = EmotionCNN(
        input_channels=cfg.input_channels,
        num_features=cfg.num_features,
        num_classes=2,
        dropout_rate=cfg.dropout_rate,
        use_pretrained=cfg.use_pretrained,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  CNN loaded : {path}")
    return model


def _load_hmm(path: Path) -> HMMClassifier:
    clf = HMMClassifier(
        n_states=config.hmm.n_states,
        n_iter=config.hmm.n_iter,
        covariance_type=config.hmm.covariance_type,
    )
    clf.load(str(path))
    print(f"  HMM loaded : {path}")
    return clf


def _load_lr(path: Path):
    clf = joblib.load(path)
    print(f"  LR  loaded : {path}")
    return clf


# ─── predictions ─────────────────────────────────────────────────────────────


def _cnn_predict(
    cnn: EmotionCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, softmax_prob_class1, true_labels)."""
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_labels: list[int] = []
    cnn.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = cnn(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def _hmm_predict(
    hmm_clf: HMMClassifier,
    cnn: EmotionCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, confidences, true_labels)."""
    seqs, labels, lengths = extract_sequences(cnn, loader, device)
    preds, confs = hmm_clf.predict(seqs, lengths=lengths)
    return preds, confs, labels


def _lr_predict(
    lr_clf,
    dataset: HandwritingDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (preds, proba_class1, true_labels) using handcrafted features."""
    feats, labels = gather_lr_features(dataset)
    preds = lr_clf.predict(feats)
    probs = lr_clf.predict_proba(feats)[:, 1]
    return preds, probs, labels


# ─── metrics ─────────────────────────────────────────────────────────────────


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    m: dict = {}
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    for i, name in enumerate(_LABELS):
        prec = precision_score(
            y_true, y_pred, labels=[i], average=None, zero_division=0
        )
        rec = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)
        m[f"precision_{name}"] = float(prec[0]) if len(prec) else 0.0
        m[f"recall_{name}"] = float(rec[0]) if len(rec) else 0.0
        m[f"f1_{name}"] = float(f1[0]) if len(f1) else 0.0
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            m["roc_auc"] = float("nan")
    else:
        m["roc_auc"] = float("nan")
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return m


# ─── confusion matrix figure ─────────────────────────────────────────────────


def _save_confusion_matrix(cm: list, out_path: Path) -> None:
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=_LABELS,
        yticklabels=_LABELS,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix — CNN-HMM (Test Split)",
    )
    thresh = arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix : {out_path}")


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────


class _GradCAM:
    """Grad-CAM for EmotionCNN with ResNet18 backbone."""

    def __init__(self, model: EmotionCNN, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._hook_fwd)
        self._bwd = target_layer.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, _m, _i, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _hook_bwd(self, _m, _gi, grad_out: tuple) -> None:
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return (cam_array in [0,1], predicted_class_idx)."""
        self.model.eval()
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        self.model.zero_grad()
        out[0, class_idx].backward()
        assert self.gradients is not None and self.activations is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def _gradcam_target_layer(cnn: EmotionCNN) -> torch.nn.Module:
    """Return layer4 (final conv block) of the ResNet18 backbone."""
    return cnn.extractor.feature_layers[-1]


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Undo [0.5, 0.5] normalisation → (H, W) uint8."""
    arr = t.squeeze().cpu().numpy()
    arr = arr * 0.5 + 0.5  # [0, 1]
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _overlay_cam(gray: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam_r = cv2.resize(cam, (gray.shape[1], gray.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam_r * 255), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(bgr, 0.6, heatmap, 0.4, 0)


def _save_gradcam_images(
    cnn: EmotionCNN,
    test_ds: HandwritingDataset,
    hmm_preds: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    out_dir: Path,
) -> list[str]:
    """Save 4 Grad-CAM panels (original | overlay) for the thesis defence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    val_tf = get_val_transform(config.preprocessing.target_size)

    # Bucket sample indices by prediction outcome
    buckets: dict[str, list[int]] = {
        "correct_HAPPY": [],
        "correct_SAD": [],
        "misclass_HAPPY": [],  # true HAPPY, HMM predicted SAD
        "misclass_SAD": [],  # true SAD,  HMM predicted HAPPY
    }
    for i, (true, pred) in enumerate(zip(test_labels, hmm_preds)):
        if true == 0 and pred == 0:
            buckets["correct_HAPPY"].append(i)
        elif true == 1 and pred == 1:
            buckets["correct_SAD"].append(i)
        elif true == 0 and pred == 1:
            buckets["misclass_HAPPY"].append(i)
        elif true == 1 and pred == 0:
            buckets["misclass_SAD"].append(i)

    cam_gen = _GradCAM(cnn, _gradcam_target_layer(cnn))
    saved: list[str] = []

    for cat, indices in buckets.items():
        if not indices:
            print(f"  Grad-CAM: no sample for '{cat}' — skipping")
            continue
        idx = indices[0]
        img_path, label = test_ds.samples[idx]

        pil_img = PILImage.open(img_path).convert("L")
        tensor = val_tf(pil_img).unsqueeze(0).to(device)

        # Use HMM prediction as the target class for backward pass
        target_cls = int(hmm_preds[idx])
        cam_arr, _ = cam_gen.generate(tensor, class_idx=target_cls)

        gray_u8 = _tensor_to_uint8(tensor.squeeze(0))
        overlay = _overlay_cam(gray_u8, cam_arr)

        true_name = _LABELS[label]
        pred_name = _LABELS[target_cls]
        out_path = out_dir / f"{cat}.png"

        # Side-by-side panel: original | Grad-CAM overlay
        orig_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        panel = np.hstack([orig_bgr, overlay])
        canvas = np.zeros((panel.shape[0] + 30, panel.shape[1], 3), dtype=np.uint8)
        canvas[30:] = panel
        caption = f"True: {true_name}  Predicted: {pred_name}"
        cv2.putText(
            canvas,
            caption,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), canvas)
        saved.append(str(out_path))
        print(f"  Grad-CAM [{cat}] : {out_path}")

    cam_gen.remove()
    if not saved:
        print("  WARNING: no Grad-CAM images saved (test set too small?)")
    return saved


# ─── CSV helpers ─────────────────────────────────────────────────────────────


def _save_evaluation_report(
    cnn_hmm: dict,
    lr: dict,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in cnn_hmm if k != "confusion_matrix"]

    def _fmt(v) -> str:
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "cnn_hmm", "lr_baseline"])
        for k in keys:
            w.writerow([k, _fmt(cnn_hmm[k]), _fmt(lr.get(k, ""))])
    print(f"  Evaluation report : {out_path}")


def _save_model_comparison(
    cnn_hmm: dict,
    lr: dict,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "f1_macro", "accuracy", "roc_auc"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "model": "CNN-HMM",
                "f1_macro": f"{cnn_hmm['f1_macro']:.4f}",
                "accuracy": f"{cnn_hmm['accuracy']:.4f}",
                "roc_auc": f"{cnn_hmm.get('roc_auc', float('nan')):.4f}",
            }
        )
        w.writerow(
            {
                "model": "LR baseline",
                "f1_macro": f"{lr['f1_macro']:.4f}",
                "accuracy": f"{lr['accuracy']:.4f}",
                "roc_auc": f"{lr.get('roc_auc', float('nan')):.4f}",
            }
        )
    print(f"  Model comparison  : {out_path}")


# ─── cross-validation ─────────────────────────────────────────────────────────


def _run_cross_validation(cfg, n_folds: int, cv_epochs: int) -> dict:
    """5-fold CV on train+val pool. Returns summary dict from CrossValidator."""
    splits_dir = Path(cfg.data.splits_dir)
    train_ds = HandwritingDataset(str(splits_dir / "train"), transform=None)
    val_ds = HandwritingDataset(str(splits_dir / "val"), transform=None)

    # Merge datasets by combining .samples lists
    combined: HandwritingDataset = object.__new__(HandwritingDataset)
    combined.data_dir = splits_dir / "train"
    combined.transform = None
    combined.samples = train_ds.samples + val_ds.samples
    print(f"  Combined pool : {len(combined.samples)} samples")

    cv = CrossValidator(
        n_splits=n_folds,
        random_state=cfg.training.random_state,
        batch_size=cfg.training.batch_size,
        epochs=cv_epochs,
        learning_rate=cfg.training.learning_rate,
        patience=cfg.training.patience,
        cnn_features=cfg.cnn.num_features,
        hmm_states=cfg.hmm.n_states,
        image_size=cfg.preprocessing.target_size,
        use_pretrained=cfg.cnn.use_pretrained,
    )
    return cv.cross_validate(combined)


# ─── main orchestration ───────────────────────────────────────────────────────


def run_evaluation(smoke: bool = False) -> None:
    print("=" * 60)
    print("Phase 10 - FINAL EVALUATION")
    print("TEST SPLIT: first and only access to DATA/SPLITS/test/")
    print("=" * 60)

    cfg = config
    _RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device : {device}")
    if smoke:
        print("  [smoke mode] — reduced CV folds and epochs")

    # ── locate checkpoints ────────────────────────────────────────────────
    cnn_path = _latest("cnn_*_best.pth")
    hmm_path = _latest("hmm_*.pkl")
    lr_path = _latest("lr_baseline_*.pkl")

    missing = [
        name
        for name, p in [("CNN", cnn_path), ("HMM", hmm_path), ("LR", lr_path)]
        if p is None
    ]
    if missing:
        print(f"ERROR: missing checkpoints: {', '.join(missing)}")
        print("  Run Phase 9 (trainer) first.")
        sys.exit(1)

    # ── load models ───────────────────────────────────────────────────────
    print("\nLoading models...")
    cnn = _load_cnn(cnn_path, device)
    hmm_clf = _load_hmm(hmm_path)
    lr_clf = _load_lr(lr_path)

    # ── load test data ─────────────────────────────────────────────────────
    test_dir = Path(cfg.data.splits_dir) / "test"
    if not test_dir.exists():
        print(f"ERROR: test split not found: {test_dir}")
        print("  Run Phase 8 (split_dataset) first.")
        sys.exit(1)

    val_tf = get_val_transform(cfg.preprocessing.target_size)
    test_ds = HandwritingDataset(str(test_dir), transform=val_tf)
    if len(test_ds) == 0:
        print("ERROR: no images found in test split.")
        sys.exit(1)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── CNN + HMM predictions on test split ───────────────────────────────
    print("\nRunning CNN predictions on test split...")
    cnn_preds, cnn_probs, test_labels = _cnn_predict(cnn, test_loader, device)

    print("Running HMM predictions on test split...")
    hmm_preds, _hmm_confs, _ = _hmm_predict(hmm_clf, cnn, test_loader, device)

    # CNN-HMM final: HMM label, CNN softmax prob for ROC-AUC
    cnn_hmm_metrics = _compute_metrics(test_labels, hmm_preds, cnn_probs)

    # ── LR predictions ────────────────────────────────────────────────────
    print("Running LR predictions on test split...")
    lr_preds, lr_probs, _ = _lr_predict(lr_clf, test_ds)
    lr_metrics = _compute_metrics(test_labels, lr_preds, lr_probs)

    # ── save reports ──────────────────────────────────────────────────────
    _save_evaluation_report(
        cnn_hmm_metrics, lr_metrics, _RESULTS / "evaluation_report.csv"
    )
    _save_confusion_matrix(
        cnn_hmm_metrics["confusion_matrix"], _RESULTS / "confusion_matrix.png"
    )
    _save_model_comparison(
        cnn_hmm_metrics, lr_metrics, _RESULTS / "model_comparison.csv"
    )

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    print("\nGenerating Grad-CAM visualisations...")
    gradcam_paths = _save_gradcam_images(
        cnn,
        test_ds,
        hmm_preds,
        test_labels,
        device,
        _RESULTS / "gradcam",
    )

    # ── cross-validation ──────────────────────────────────────────────────
    n_folds = 2 if smoke else cfg.training.n_folds
    cv_epochs = 2 if smoke else min(30, cfg.training.epochs)
    print(f"\nRunning {n_folds}-fold CV on train+val pool ({cv_epochs} epochs/fold)...")
    cv_results = _run_cross_validation(cfg, n_folds=n_folds, cv_epochs=cv_epochs)
    cv_path = _RESULTS / "cv_results.json"
    with cv_path.open("w", encoding="utf-8") as fh:
        json.dump(cv_results, fh, indent=2)
    print(f"  CV results : {cv_path}")

    # ── final summary ─────────────────────────────────────────────────────
    cnn_hmm_f1 = cnn_hmm_metrics["f1_macro"]
    lr_f1 = lr_metrics["f1_macro"]
    cnn_hmm_acc = cnn_hmm_metrics["accuracy"]
    lr_acc = lr_metrics["accuracy"]
    cnn_hmm_auc = cnn_hmm_metrics.get("roc_auc", float("nan"))
    lr_auc = lr_metrics.get("roc_auc", float("nan"))
    cv_f1_mean = cv_results.get("f1", {}).get("mean", float("nan"))
    cv_f1_std = cv_results.get("f1", {}).get("std", float("nan"))
    threshold_met = cnn_hmm_f1 >= _THRESHOLD

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(f"\n{'':16s}  {'F1 macro':>10}  {'Accuracy':>10}  {'ROC-AUC':>10}")
    print(
        f"{'CNN-HMM':16s}  {cnn_hmm_f1:10.4f}  {cnn_hmm_acc:10.4f}  {cnn_hmm_auc:10.4f}"
    )
    print(f"{'LR baseline':16s}  {lr_f1:10.4f}  {lr_acc:10.4f}  {lr_auc:10.4f}")

    print(f"\nCross-Validation ({n_folds}-fold, train+val pool)")
    print(f"  CNN-HMM F1 macro : {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")

    status = "MET" if threshold_met else "NOT MET"
    print(f"\nThreshold F1 macro >= {_THRESHOLD:.0%} : {status}")
    print("=" * 60)

    print("\nOutput paths:")
    print("  results/evaluation_report.csv")
    print("  results/confusion_matrix.png")
    print("  results/model_comparison.csv")
    print("  results/cv_results.json")
    for p in gradcam_paths:
        print(f"  {p}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10: final evaluation on the held-out test split."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 2 CV folds, 2 epochs — quick end-to-end verification.",
    )
    args = parser.parse_args()
    run_evaluation(smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
