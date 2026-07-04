"""
Configuration Module — INSIDE-OUT / EMHA
All hyperparameters live here in dataclasses. Modules run as python -m src.*
from the project root. See ProcessPipeline.txt Section A for labeling rules.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""

    raw_data_dir: str = "E:\\EMHA_Thesis\\DATASET\\raw"  # READ-ONLY
    metadata_dir: str = "DATA/METADATA"
    crops_dir: str = "DATA/CROPS"          # Phase 4 output
    processed_dir: str = "DATA/PROCESSED"  # Phase 7 output (mirrors CROPS)

    image_size: Tuple[int, int] = (224, 224)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""

    target_size: Tuple[int, int] = (224, 224)
    binarize_threshold: Optional[int] = None  # None = Otsu's method
    denoise_kernel_size: int = 3
    normalize: bool = True
    min_crop_size: Tuple[int, int] = (80, 80)
    blank_threshold: float = 245.0  # grayscale mean above this = blank
    # Task-code prefixes whose crops skip deskew (drawings have no text baseline).
    skip_skew_prefixes: Tuple[str, ...] = ("draw_",)


@dataclass
class CNNConfig:
    """CNN model configuration."""

    input_channels: int = 1          # Grayscale
    num_features: int = 256
    dropout_rate: float = 0.5
    use_pretrained: bool = True       # ResNet18 backbone (Section A Rule 6)
    pretrained_backbone: str = "resnet18"
    freeze_backbone: bool = True


@dataclass
class HMMConfig:
    """HMM model configuration."""

    n_states: int = 4
    n_iter: int = 100
    covariance_type: str = "diag"


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.001
    n_folds: int = 5
    random_state: int = 42
    checkpoint_dir: str = "models"
    save_best_only: bool = True


@dataclass
class LabelingConfig:
    """FINALE 24-item self-report scoring per ProcessPipeline.txt Section A.

    Happiness items kept raw; sadness items reverse-scored (6 - raw).
    adjusted_total = happiness_sum + (72 - sadness_sum)   range 24-120
    label = HAPPY if adjusted_total >= 72 else SAD        integer math only
    soft P_happy = (adjusted_total/24 - 1) / 4            stored, not used
    for the binary decision.  NO NEUTRAL class anywhere.
    """

    happiness_items: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 13, 16, 19, 20, 21, 23)
    sadness_items: Tuple[int, ...] = (1, 3, 5, 7, 9, 11, 14, 15, 17, 18, 22, 24)
    likert_min: int = 1
    likert_max: int = 5
    adjusted_total_threshold: int = 72   # HAPPY if >= 72, SAD otherwise


@dataclass
class Config:
    """Master configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

    project_name: str = "INSIDE-OUT"
    version: str = "1.0.0"

    def __post_init__(self):
        """Create writable pipeline directories.

        DATASET/raw is READ-ONLY and deliberately excluded here.
        """
        for dir_path in [
            self.data.metadata_dir,
            self.data.crops_dir,
            self.data.processed_dir,
            self.training.checkpoint_dir,
            "FIGURES",
            "results",
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = Config()


# ---------------------------------------------------------------------------
# Crop coordinate maps — fractional (0.0–1.0) relative to scan dimensions.
# Reference scan: ~1700 px wide × ~2200 px tall (anisotropic ~74.5 DPI H,
# ~81.1 DPI V).  Tune against real scans with src/data/preview_crops.py.
# Format: (x1_frac, y1_frac, x2_frac, y2_frac)
# ---------------------------------------------------------------------------

# page_drawing (page index 2): 2×2 grid below the header band.
# Row heights are unequal: top row (Circles, Dots) is shorter than bottom row.
DRAWING_CROPS = {
    "draw_circles": (0.0706, 0.2318, 0.4765, 0.5500),   # top-left
    "draw_dots":    (0.5235, 0.2318, 0.9294, 0.5500),   # top-right
    "draw_person":  (0.0706, 0.6273, 0.4765, 0.9455),   # bottom-left
    "draw_house":   (0.5235, 0.6273, 0.9294, 0.9455),   # bottom-right
}

# page_writing (page index 3), section 1: 5×3 word table (15 cells).
# Rows: content, melancholic, optimistic, disconnected, vibrant.
# Columns: left-hand, right-hand, uppercase.
WORD_CROPS = {
    "word_content_left":        (0.2529, 0.2000, 0.4706, 0.2655),
    "word_content_right":       (0.4765, 0.2000, 0.6941, 0.2655),
    "word_content_upper":       (0.7000, 0.2000, 0.9235, 0.2655),
    "word_melancholic_left":    (0.2529, 0.2745, 0.4706, 0.3400),
    "word_melancholic_right":   (0.4765, 0.2745, 0.6941, 0.3400),
    "word_melancholic_upper":   (0.7000, 0.2745, 0.9235, 0.3400),
    "word_optimistic_left":     (0.2529, 0.3491, 0.4706, 0.4145),
    "word_optimistic_right":    (0.4765, 0.3491, 0.6941, 0.4145),
    "word_optimistic_upper":    (0.7000, 0.3491, 0.9235, 0.4145),
    "word_disconnected_left":   (0.2529, 0.4236, 0.4706, 0.4891),
    "word_disconnected_right":  (0.4765, 0.4236, 0.6941, 0.4891),
    "word_disconnected_upper":  (0.7000, 0.4236, 0.9235, 0.4891),
    "word_vibrant_left":        (0.2529, 0.4982, 0.4706, 0.5636),
    "word_vibrant_right":       (0.4765, 0.4982, 0.6941, 0.5636),
    "word_vibrant_upper":       (0.7000, 0.4982, 0.9235, 0.5636),
}

# page_writing (page index 3), section 2: 5 cursive sentence rows.
# Each row includes its pre-printed prompt line.
CURSIVE_CROPS = {
    "cursive_01": (0.0765, 0.6318, 0.9235, 0.6909),
    "cursive_02": (0.0765, 0.6955, 0.9235, 0.7545),
    "cursive_03": (0.0765, 0.7591, 0.9235, 0.8182),
    "cursive_04": (0.0765, 0.8227, 0.9235, 0.8818),
    "cursive_05": (0.0765, 0.8864, 0.9235, 0.9455),
}


if __name__ == "__main__":
    print("INSIDE-OUT Configuration")
    print("=" * 40)
    print(f"\nProject: {config.project_name} v{config.version}")
    print(f"\nData Config:")
    print(f"  Raw (READ-ONLY): {config.data.raw_data_dir}")
    print(f"  Crops:           {config.data.crops_dir}")
    print(f"  Processed:       {config.data.processed_dir}")
    print(f"\nCNN Config:")
    print(f"  use_pretrained: {config.cnn.use_pretrained}  (ResNet18)")
    print(f"  num_features:   {config.cnn.num_features}")
    print(f"\nHMM Config:")
    print(f"  n_states: {config.hmm.n_states}")
    print(f"\nLabeling (Section A):")
    print(f"  threshold (adjusted_total >= {config.labeling.adjusted_total_threshold} -> HAPPY)")
    print(f"  NO NEUTRAL class")
