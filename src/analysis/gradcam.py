"""
Grad-CAM for the fine-tuned CNN — Stage H4 (secondary, descriptive).

gradcam() is plain Grad-CAM (Selvaraju et al.) with forward/backward hooks on
a named layer: the target class score is back-propagated, the layer's
gradients are averaged over space to weight its activation maps, and
ReLU(sum_c w_c A_c) is upsampled (bilinear) to the input size and scaled to
[0, 1]. Written here rather than taken from pytorch-grad-cam: no extra
dependency, and 1-channel input needs nothing special.

run() explains the outer-test predictions of the analysis run
(GradCAMConfig.analysis, ResNet18 backbone): from
results/cnn/<run>/crop_predictions.csv, per task family, the n_examples
most confident correct HAPPY, correct SAD, HAPPY-predicted-SAD and
SAD-predicted-HAPPY crops (middle-band rows excluded). Each crop is
explained by the fold model that predicted it (models/cnn/<run>/<family>/
fold_<k>.pth, whose stored test IDs must contain the participant), for the
class it predicted; its recomputed prob_sad must match the stored one
(GradCAMConfig.match_atol). Overlays are drawn on the processed crop shown
dark ink on white (un-inverted when PreprocessingConfig.invert), with the
participant code, cell, true label, prob_sad and fold in the caption.

Outputs, results/final/gradcam/:

    gradcam_<family>.png                 summary grid (300 dpi)
    <family>/<category>_<code>_<cell>.png   one overlay per crop

save_overlay(cnn, image, out_path) is the hook used by src.app.predict.

    python -m src.analysis.gradcam [--task-family word]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import to_rgb
from PIL import Image

from src.data.dataloader import load_tables
from src.data.transforms import NORMALIZE_MEAN, NORMALIZE_STD, get_eval_transform
from src.models.hybrid import cnn_checkpoint_path, load_cnn
from src.training.aggregate import FAMILIES
from src.training.evaluator import output_dir
from src.training.run_cnn import MODEL, run_name
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic

plt.switch_backend("Agg")

SAD, HAPPY = "SAD", "HAPPY"
# (category, true label, predicted label, most confident = highest prob_sad?)
CATEGORIES = (
    ("correct_HAPPY", HAPPY, HAPPY, False),
    ("correct_SAD", SAD, SAD, True),
    ("HAPPY_predicted_SAD", HAPPY, SAD, True),
    ("SAD_predicted_HAPPY", SAD, HAPPY, False),
)
_SURFACE, _INK, _INK_2, _HEAT = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834"
_PNG_META = {"Software": None}


# ─── Grad-CAM ────────────────────────────────────────────────────────────────


def resolve_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """The one submodule whose dotted name is `name` or ends with `.name`."""
    hits = [m for n, m in model.named_modules() if n == name or n.endswith("." + name)]
    if len(hits) != 1:
        raise ValueError(f"layer {name!r} matches {len(hits)} modules, need one")
    return hits[0]


def gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer: str,
    class_idx: int,
) -> np.ndarray:
    """(H, W) Grad-CAM map in [0, 1] for one (1, H, W) or (1, 1, H, W) image."""
    x = image if image.dim() == 4 else image.unsqueeze(0)
    if x.shape[0] != 1:
        raise ValueError("gradcam explains one image at a time")
    x = x.detach().clone().requires_grad_(True)  # graph exists even if frozen
    layer = resolve_layer(model, target_layer)
    store = {}
    fwd = layer.register_forward_hook(lambda _m, _i, out: store.update(act=out))
    bwd = layer.register_full_backward_hook(
        lambda _m, _gi, gout: store.update(grad=gout[0])
    )
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            model.zero_grad()
            model(x)[0, class_idx].backward()
    finally:
        fwd.remove()
        bwd.remove()
        model.train(was_training)
    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    peak = float(cam.max())
    return (cam / peak if peak > 0 else torch.zeros_like(cam)).numpy()


def display_image(image: torch.Tensor, cfg=config) -> np.ndarray:
    """(H, W) in [0, 1], dark ink on white, from an eval-normalized tensor."""
    arr = image.detach().reshape(image.shape[-2:]).cpu().numpy()
    arr = arr * NORMALIZE_STD[0] + NORMALIZE_MEAN[0]
    if cfg.preprocessing.invert:
        arr = 1.0 - arr
    return np.clip(arr, 0.0, 1.0)


def overlay(ax, image: torch.Tensor, cam: np.ndarray, cfg=config) -> None:
    """Grayscale crop with the heat map as one hue whose opacity = weight."""
    ax.imshow(display_image(image, cfg), cmap="gray", vmin=0, vmax=1)
    heat = np.zeros(cam.shape + (4,))
    heat[..., :3] = to_rgb(_HEAT)
    heat[..., 3] = cam * cfg.gradcam.overlay_alpha
    ax.imshow(heat)
    ax.set_xticks([])
    ax.set_yticks([])


def _predict(cnn, image: torch.Tensor) -> tuple:
    with torch.no_grad():
        probs = torch.softmax(cnn(image.unsqueeze(0)), dim=1)[0]
    sad = config.data.label_to_index[SAD]
    return int(probs.argmax()), float(probs[sad])


def save_overlay(
    cnn, image: torch.Tensor, out_path, class_idx=None, caption=None, cfg=config
) -> Path:
    """One overlay PNG for a (1, H, W) eval tensor (default: predicted class)."""
    if class_idx is None:
        class_idx = _predict(cnn, image)[0]
    cam = gradcam(cnn, image, cfg.gradcam.target_layer, class_idx)
    h, w = cam.shape
    fig, ax = plt.subplots(figsize=(max(2.0, 3.0 * w / h), 3.0))
    fig.patch.set_facecolor(_SURFACE)
    overlay(ax, image, cam, cfg)
    if caption:
        ax.set_title(caption, fontsize=7, color=_INK)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.evaluation.dpi, metadata=_PNG_META)
    plt.close(fig)
    return out_path


# ─── selection ────────────────────────────────────────────────────────────────


def crop_predictions_path(cfg=config) -> Path:
    name = run_name(cfg.gradcam.analysis, None, "resnet18")
    return cfg.paths.results_dir / "cnn" / name / "crop_predictions.csv"


def select_examples(crops: pd.DataFrame, family: str, cfg=config) -> pd.DataFrame:
    """The n_examples most confident crops of each category (outer test only)."""
    middle = crops["in_middle_band"].astype(str) == "True"
    part = crops[
        (crops["task_family"] == family) & (crops["model"] == MODEL) & ~middle
    ].copy()
    part["pred"] = np.where(part["prob_sad"] >= cfg.aggregate.threshold, SAD, HAPPY)
    chosen = []
    for name, true, pred, high in CATEGORIES:
        rows = part[(part["label"] == true) & (part["pred"] == pred)]
        rows = rows.sort_values(
            ["prob_sad", "participant_id", "cell"],
            ascending=[not high, True, True],
            kind="mergesort",
        )
        chosen.append(rows.head(cfg.gradcam.n_examples).assign(category=name))
    return pd.concat(chosen, ignore_index=True)


# ─── run ──────────────────────────────────────────────────────────────────────


def _caption(row) -> str:
    return (
        f"{row['participant_id']} {row['cell']}  true {row['label']}\n"
        f"prob_sad {row['prob_sad']:.3f}  fold {int(row['fold'])}"
    )


def explain_family(crops: pd.DataFrame, family: str, cfg=config) -> list:
    """Overlays + summary grid for one family; returns the written paths."""
    examples = select_examples(crops, family, cfg)
    manifest, _, _ = load_tables(cfg)
    path_of = manifest.assign(
        participant_id=manifest["participant_id"].astype(str)
    ).set_index(["participant_id", "cell"])["processed_path"]
    transform = get_eval_transform(cfg, family)
    name = run_name(cfg.gradcam.analysis, None, "resnet18")
    out = output_dir(None, cfg) / cfg.gradcam.output_subdir
    models, written, panels = {}, [], []

    for _, row in examples.iterrows():
        pid, fold = str(row["participant_id"]), int(row["fold"])
        if fold not in models:
            ckpt = cnn_checkpoint_path(cfg, name, family, fold)
            if not ckpt.is_file():
                raise FileNotFoundError(f"{ckpt} not found; run src.training.run_cnn")
            models[fold] = load_cnn(ckpt)
        cnn, state = models[fold]
        if pid not in set(map(str, state.get("test_ids", []))):
            raise RuntimeError(f"{pid} is not an outer-test participant of fold {fold}")
        with Image.open(path_of.loc[(pid, row["cell"])]) as img:
            image = transform(img.convert("L"))
        pred_idx, prob = _predict(cnn, image)
        if abs(prob - float(row["prob_sad"])) > cfg.gradcam.match_atol:
            raise RuntimeError(
                f"{pid} {row['cell']}: fold {fold} model gives prob_sad {prob:.4f}, "
                f"crop_predictions.csv {row['prob_sad']:.4f}; wrong checkpoint?"
            )
        cam = gradcam(cnn, image, cfg.gradcam.target_layer, pred_idx)
        panels.append((row, image, cam))
        path = out / family / f"{row['category']}_{pid}_{row['cell']}.png"
        written.append(save_overlay(cnn, image, path, pred_idx, _caption(row), cfg=cfg))

    written.insert(0, _grid(panels, family, out / f"gradcam_{family}.png", cfg))
    return written


def _grid(panels: list, family: str, path: Path, cfg=config) -> Path:
    n = cfg.gradcam.n_examples
    fig, axes = plt.subplots(
        len(CATEGORIES), n, figsize=(2.4 * n, 2.4 * len(CATEGORIES)), squeeze=False
    )
    fig.patch.set_facecolor(_SURFACE)
    by_cat = {
        name: [p for p in panels if p[0]["category"] == name] for name, *_ in CATEGORIES
    }
    for r, (name, *_rest) in enumerate(CATEGORIES):
        for c in range(n):
            ax = axes[r, c]
            if c < len(by_cat[name]):
                row, image, cam = by_cat[name][c]
                overlay(ax, image, cam, cfg)
                ax.set_title(_caption(row), fontsize=6, color=_INK)
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "none", ha="center", color=_INK_2, fontsize=7)
            if c == 0:
                ax.set_ylabel(name.replace("_", " "), fontsize=7, color=_INK)
    fig.suptitle(
        f"Secondary analysis: Grad-CAM ({cfg.gradcam.target_layer}), {family} "
        f"crops, outer-test predictions; orange = weight for the predicted class",
        fontsize=9,
        color=_INK,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.evaluation.dpi, metadata=_PNG_META)
    plt.close(fig)
    return path


def run(task_family: str | None = None) -> list:
    """Grad-CAM grids and overlays for one family (default: all); their paths."""
    require_frozen_or_synthetic(config, "gradcam")
    if task_family is not None and task_family not in FAMILIES:
        raise ValueError(f"task_family must be one of {FAMILIES}")
    path = crop_predictions_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run src.training.run_cnn first")
    crops = pd.read_csv(path, dtype={"participant_id": str})
    families = [task_family] if task_family else list(FAMILIES)
    written = []
    for family in families:
        if not (crops["task_family"] == family).any():
            print(f"gradcam: no {family} crops in {path}; skipped")
            continue
        written += explain_family(crops, family)
        print(f"gradcam: {family} -> {written[-1].parent.parent}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM of the CNN (Stage H4).")
    parser.add_argument("--task-family", choices=FAMILIES, default=None)
    args = parser.parse_args()
    try:
        run(args.task_family)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
