"""
Evaluator — Phase 10 — LEGACY ENTRY POINTS QUARANTINED.

run_evaluation and _run_cross_validation (and the CLI) raise
LegacyPipelineError: they evaluated a fixed crop-level held-out test folder,
not participant-level CV folds, and their bodies have been removed.
Prediction (CropDataset batches), metric, confusion-matrix, CSV, and
Grad-CAM helpers are kept for reuse by the Stage H evaluator.
"""

from __future__ import annotations

import argparse
import csv
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

from src.data.dataloader import (
    CropDataset,
    LegacyPipelineError,
    legacy_pipeline_error,
)
from src.models.cnn import EmotionCNN
from src.models.hmm import HMMClassifier
from src.training.trainer import extract_sequences, gather_lr_features
from src.utils.config import config

plt.switch_backend("Agg")

_RESULTS = config.paths.results_dir
_MODELS = config.paths.models_dir
_LABELS = ["HAPPY", "SAD"]
_THRESHOLD = 0.70  # minimum F1 macro (thesis target: 70–85%)


# ─── checkpoint discovery ─────────────────────────────────────────────────────


def _latest(pattern: str) -> Path | None:
    candidates = sorted(_MODELS.glob(pattern))
    return candidates[-1] if candidates else None


# ─── model loading ────────────────────────────────────────────────────────────


def _load_cnn(path: Path, device: torch.device) -> EmotionCNN:
    model = EmotionCNN(config.cnn).to(device)
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
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
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
    dataset: CropDataset,
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
    """Final conv block: ResNet18 layer4, or the simple CNN's last conv."""
    return cnn.extractor.gradcam_layer


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
    test_ds: CropDataset,
    hmm_preds: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    out_dir: Path,
    val_tf,
) -> list[str]:
    """Save 4 Grad-CAM panels (original | overlay) for the thesis defence.

    val_tf is the eval transform for the crops' task family
    (src.data.transforms.get_eval_transform).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

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
        row = test_ds.table.iloc[idx]
        img_path, label = row["processed_path"], int(row["label_index"])

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
    """5-fold CV on train+val pool. LEGACY (disabled): crop-level folds."""
    raise legacy_pipeline_error("_run_cross_validation")


# ─── main orchestration ───────────────────────────────────────────────────────


def run_evaluation(smoke: bool = False) -> None:
    """LEGACY (disabled): raises LegacyPipelineError."""
    raise legacy_pipeline_error("run_evaluation")


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
    try:
        run_evaluation(smoke=args.smoke)
    except LegacyPipelineError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
