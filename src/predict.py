"""
Phase 11 — End-to-end inference (predict.py).

CLI: input = a scanned page set (folder) or a single crop image.
Pipeline: register -> extract -> preprocess -> model -> HAPPY/SAD + confidence.

This is the artifact that demonstrates the thesis goal: emotion from
handwriting alone, no questionnaire at inference time.

Usage (run from project root):

    python -m src.predict <path_to_respondent_folder>
    python -m src.predict <path_to_single_crop.png>

Example:
    python -m src.predict DATASET/raw/respondent_001
    python -m src.predict DATA/CROPS/P001/P001_draw_circles.png

Done when: prints label + confidence on a held-out respondent never seen
in training (defense demo artifact).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.hybrid import HybridCNNHMM
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import config, DRAWING_CROPS, WORD_CROPS, CURSIVE_CROPS

ALL_TASK_CODES = list(DRAWING_CROPS) + list(WORD_CROPS) + list(CURSIVE_CROPS)


def imread_unicode(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess_image(image: np.ndarray, skip_skew: bool) -> torch.Tensor:
    """Apply preprocessing pipeline and return a (1, 1, 224, 224) tensor."""
    cfg = config.preprocessing
    pipeline = PreprocessingPipeline(
        target_size=cfg.target_size,
        binarize_threshold=cfg.binarize_threshold,
        denoise_kernel_size=cfg.denoise_kernel_size,
        skip_skew=skip_skew,
    )
    processed = pipeline.process(image)  # float32 in [0, 1]
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor.float()


def predict_single_crop(
    crop_path: Path,
    model: HybridCNNHMM,
    task_code: str = "",
) -> tuple[str, float]:
    """Predict emotion from a single crop image."""
    image = imread_unicode(crop_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read crop: {crop_path}")

    skip_skew = any(
        task_code.startswith(p) for p in config.preprocessing.skip_skew_prefixes
    )
    tensor = preprocess_image(image, skip_skew=skip_skew)
    return model.predict(tensor)


def predict_respondent_folder(
    folder: Path,
    model: HybridCNNHMM,
) -> tuple[str, float]:
    """Predict emotion by majority vote over all available crops in a folder."""
    votes: dict[str, int] = {"HAPPY": 0, "SAD": 0}
    confidences: list[float] = []

    for task_code in ALL_TASK_CODES:
        crop_files = list(folder.glob(f"*_{task_code}.png"))
        if not crop_files:
            continue
        crop_path = crop_files[0]
        try:
            label, conf = predict_single_crop(crop_path, model, task_code)
            votes[label] += 1
            confidences.append(conf)
        except Exception:
            continue

    if not confidences:
        raise RuntimeError(f"No processable crops found in {folder}")

    prediction = max(votes, key=lambda k: votes[k])
    avg_confidence = float(np.mean(confidences))
    return prediction, avg_confidence


def load_model(cnn_path: str, hmm_path: str) -> HybridCNNHMM:
    model = HybridCNNHMM(
        cnn_features=config.cnn.num_features,
        hmm_states=config.hmm.n_states,
    )
    model.load(cnn_path, hmm_path)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict emotion (HAPPY/SAD) from handwriting."
    )
    parser.add_argument("input", help="Respondent folder or single crop path")
    parser.add_argument(
        "--cnn", default="models/cnn_best.pth", help="CNN checkpoint path"
    )
    parser.add_argument(
        "--hmm", default="models/hmm.pkl", help="HMM checkpoint path"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input path does not exist: {input_path}")
        return 1

    model = load_model(args.cnn, args.hmm)

    if input_path.is_dir():
        label, confidence = predict_respondent_folder(input_path, model)
        print(f"Input folder : {input_path}")
    else:
        task_code = input_path.stem.split("_", 1)[-1] if "_" in input_path.stem else ""
        label, confidence = predict_single_crop(input_path, model, task_code)
        print(f"Input crop   : {input_path}")

    print(f"Prediction   : {label}")
    print(f"Confidence   : {confidence:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
