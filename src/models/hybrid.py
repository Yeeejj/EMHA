"""
CNN-HMM hybrid (the titled method) — Stage F.

For each task family, a fine-tuned EmotionCNN (run_cnn checkpoint, frozen
here) turns a processed crop into a left-to-right sequence of layer3 column
features, (W/16, 256); a per-family HMMClassifier (run_hybrid) scores the
sequence and returns a Platt-calibrated P(SAD).

    hybrid = HybridCNNHMM.from_fold(fold, run_name)        # word + cursive
    label, prob_sad = hybrid.predict(crop, "word")          # one crop
    label, prob_sad = hybrid.predict_participant(
        {"word": word_crops, "cursive": cursive_crops})     # one participant

predict_participant follows the protocol: mean of crop probabilities per
family (mean_prob), then an equal-weight mean over the fused families
(HybridConfig.families, word + cursive) that are present. A secondary
family (drawing) is never fused.

Images are processed crops as stored on disk (uint8, H x W canvases); the
project eval transform is applied here. Already-transformed tensors
(C, H, W) or (N, C, H, W) are used as is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.transforms import get_eval_transform
from src.features.embeddings import resolve_device
from src.utils.config import CNNConfig, config

from .cnn import EmotionCNN
from .hmm import HMMClassifier


def cnn_checkpoint_path(cfg, run_name: str, family: str, fold: int) -> Path:
    return cfg.paths.models_dir / "cnn" / run_name / family / f"fold_{fold}.pth"


def hmm_path(cfg, run_name: str, family: str, fold: int) -> Path:
    return (
        cfg.paths.models_dir
        / cfg.hybrid.model_subdir
        / run_name
        / family
        / f"fold_{fold}.joblib"
    )


def load_cnn(path: Path) -> tuple:
    """(EmotionCNN in eval mode, checkpoint dict) from a run_cnn checkpoint."""
    state = torch.load(path, map_location="cpu")
    cnn_cfg = {
        k: tuple(v) if isinstance(v, list) else v for k, v in state["cnn"].items()
    }
    model = EmotionCNN(CNNConfig(**cnn_cfg))
    model.load_state_dict(state["model_state_dict"])
    return model.eval(), state


def crop_sequences(cnn: EmotionCNN, loader: DataLoader, device) -> tuple:
    """(meta DataFrame, list of (T, 256) arrays) for every crop in loader.

    meta has participant_id, task_family, cell, label, label_index, one row
    per sequence in the same order. The CNN is put in eval mode.
    """
    index_to_label = {v: k for k, v in config.data.label_to_index.items()}
    rows, seqs = [], []
    cnn.eval()
    with torch.no_grad():
        for batch in loader:
            out = cnn.extract_sequence_features(batch["image"].to(device)).cpu()
            for i in range(out.shape[0]):
                seqs.append(out[i].numpy())
                label_index = int(batch["label"][i])
                rows.append(
                    {
                        "participant_id": batch["participant_id"][i],
                        "task_family": batch["task_family"][i],
                        "cell": batch["cell"][i],
                        "label": index_to_label[label_index],
                        "label_index": label_index,
                    }
                )
    return pd.DataFrame(rows), seqs


class HybridCNNHMM:
    """Per-family fine-tuned CNN + calibrated HMM."""

    def __init__(self, cnns: dict, hmms: dict, device: str = "auto"):
        missing = set(hmms) - set(cnns)
        if missing:
            raise ValueError(f"HMMs without a CNN for families: {sorted(missing)}")
        self.device = torch.device(resolve_device(device))
        self.cnns = {f: m.to(self.device).eval() for f, m in cnns.items()}
        self.hmms = dict(hmms)
        self.sad = config.data.label_to_index["SAD"]

    @classmethod
    def from_fold(
        cls,
        fold: int,
        run_name: str,
        families=None,
        cfg=config,
        device: str = "auto",
    ) -> "HybridCNNHMM":
        """Load run_cnn and run_hybrid artifacts of one outer fold."""
        families = list(families or cfg.hybrid.families)
        cnns = {
            f: load_cnn(cnn_checkpoint_path(cfg, run_name, f, fold))[0]
            for f in families
        }
        hmms = {
            f: HMMClassifier().load(hmm_path(cfg, run_name, f, fold)) for f in families
        }
        return cls(cnns, hmms, device)

    def _batch(self, images, family: str) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            x = images if images.dim() == 4 else images.unsqueeze(0)
            return x.float()
        transform = get_eval_transform(config, family)
        arr = np.asarray(images)
        if arr.ndim == 2:
            arr = arr[None]
        return torch.stack(
            [transform(Image.fromarray(a.astype(np.uint8))) for a in arr]
        )

    def crop_prob_sad(self, images, family: str) -> np.ndarray:
        """Calibrated P(SAD) for each crop of one family."""
        if family not in self.hmms:
            raise ValueError(f"no HMM for family {family!r}")
        x = self._batch(images, family).to(self.device)
        seqs = self.cnns[family].extract_sequence_features(x).cpu().numpy()
        hmm = self.hmms[family]
        proba = hmm.predict_proba(list(seqs))
        return proba[:, list(hmm.classes_).index(self.sad)]

    def predict(self, image, task_family: str) -> tuple:
        """(label, P(SAD)) for one crop."""
        p = float(self.crop_prob_sad(image, task_family)[0])
        return _label(p), p

    def predict_participant(self, images: dict) -> tuple:
        """(label, P(SAD)) for one participant from {family: crops}.

        Per family: mean crop P(SAD); then the equal-weight mean over the
        fused families present (secondary families are not fused).
        """
        fused = [f for f in config.hybrid.families if f in images and f in self.hmms]
        if not fused:
            raise ValueError(f"need crops of at least one of {config.hybrid.families}")
        means = [float(np.mean(self.crop_prob_sad(images[f], f))) for f in fused]
        p = float(np.mean(means))
        return _label(p), p


def _label(prob_sad: float) -> str:
    return "SAD" if prob_sad >= config.aggregate.threshold else "HAPPY"
