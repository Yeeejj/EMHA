"""
Configuration Module
Centralized configuration for model hyperparameters and paths.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""

    raw_data_dir: str = "E:\\EMHA_Thesis\\DATASET\\raw"
    staging_dir: str = "DATA/STAGING"
    metadata_dir: str = "DATA/METADATA"
    extracted_dir: str = "DATA/EXTRACTED"
    processed_dir: str = "DATA/PROCESSED"
    splits_dir: str = "DATA/SPLITS"

    image_size: Tuple[int, int] = (224, 224)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""

    target_size: Tuple[int, int] = (224, 224)
    binarize_threshold: Optional[int] = None  # None = Otsu's method
    denoise_kernel_size: int = 3
    normalize: bool = True
    # Quality gate thresholds (validate_quality.py)
    min_crop_size: Tuple[int, int] = (80, 80)
    blank_threshold: float = 245.0  # grayscale mean above this = blank page
    # Skew correction is meaningless on drawing crops (no text baseline).
    # Task codes whose prefix appears here will have skip_skew=True.
    skip_skew_prefixes: Tuple[str, ...] = ("draw_",)


@dataclass
class CNNConfig:
    """CNN model configuration."""

    input_channels: int = 1  # Grayscale
    num_features: int = 256
    dropout_rate: float = 0.5

    # Architecture options
    use_pretrained: bool = True
    pretrained_backbone: str = "resnet18"
    freeze_backbone: bool = True


@dataclass
class HMMConfig:
    """HMM model configuration."""

    n_states: int = 4
    n_iter: int = 100
    covariance_type: str = "diag"  # "full", "diag", "tied", "spherical"


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Cross-validation
    n_folds: int = 5
    random_state: int = 42

    # Checkpointing
    checkpoint_dir: str = "models"
    save_best_only: bool = True


@dataclass
class LabelingConfig:
    """FINALE 24-item self-report scoring and emotion-labeling thresholds.

    Items are 1-5 Likert. Each subscale is summed in its own keyed
    direction (no within-subscale reversal): higher happiness_score = happier,
    higher sadness_score = sadder. Both composites range 12-60. Thresholds are
    anchored to per-item means on the 1-5 scale: mean >= 3.5 -> 42 (elevated),
    mean <= 2.5 -> 30 (low). Confirm against the pilot distribution with the
    psychometrician panel before final lock.
    """

    # 1-based item numbers (item_01 .. item_24 in questionnaire_scores.csv).
    happiness_items: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 13, 16, 19, 20, 21, 23)
    sadness_items: Tuple[int, ...] = (1, 3, 5, 7, 9, 11, 14, 15, 17, 18, 22, 24)

    likert_min: int = 1
    likert_max: int = 5

    score_min: int = 12  # 12 items x 1
    score_max: int = 60  # 12 items x 5

    happiness_high: float = 42.0  # item mean >= 3.5
    happiness_low: float = 30.0  # item mean <= 2.5
    sadness_threshold: float = 42.0  # item mean >= 3.5


@dataclass
class Config:
    """Master configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

    # Project info
    project_name: str = "INSIDE-OUT"
    version: str = "1.0.0"

    def __post_init__(self):
        """Create writable pipeline directories if they don't exist.

        The raw data dir is deliberately excluded: it is READ-ONLY and must
        never be created or written to (see CLAUDE.md Critical Rule 1).
        """
        for dir_path in [
            self.data.staging_dir,
            self.data.metadata_dir,
            self.data.extracted_dir,
            self.data.processed_dir,
            self.data.splits_dir,
            self.training.checkpoint_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = Config()


# ---------------------------------------------------------------------------
# Crop coordinate maps for page_drawing and page_writing
# ---------------------------------------------------------------------------
# PLACEHOLDER pixel boxes (x1, y1, x2, y2), sized to the actual scan
# resolution of this dataset: 1700 x 2200 px (portrait). These are first-pass
# estimates of a proportional layout and are meant to be tuned against real
# scans using src/data/preview_crops.py.
SCAN_WIDTH_PX = 1700
SCAN_HEIGHT_PX = 2200

# page_drawing (index 2): 2x2 grid below the header band, with margins for the
# page border and per-quadrant labels.
DRAWING_CROPS = {
    "draw_circles": (120, 510, 810, 1210),  # top-left
    "draw_dots": (890, 510, 1580, 1210),  # top-right (pre-printed dots)
    "draw_person": (120, 1380, 810, 2080),  # bottom-left (person in rain)
    "draw_house": (890, 1380, 1580, 2080),  # bottom-right (house and tree)
}

# page_writing (index 3), section 1: 5-row x 3-column word table (15 cells).
# Rows: content, melancholic, optimistic, disconnected, vibrant.
# Columns: left (hand), right (hand), upper(case).
WORD_CROPS = {
    "word_content_left": (430, 440, 800, 584),
    "word_content_right": (810, 440, 1180, 584),
    "word_content_upper": (1190, 440, 1570, 584),
    "word_melancholic_left": (430, 604, 800, 748),
    "word_melancholic_right": (810, 604, 1180, 748),
    "word_melancholic_upper": (1190, 604, 1570, 748),
    "word_optimistic_left": (430, 768, 800, 912),
    "word_optimistic_right": (810, 768, 1180, 912),
    "word_optimistic_upper": (1190, 768, 1570, 912),
    "word_disconnected_left": (430, 932, 800, 1076),
    "word_disconnected_right": (810, 932, 1180, 1076),
    "word_disconnected_upper": (1190, 932, 1570, 1076),
    "word_vibrant_left": (430, 1096, 800, 1240),
    "word_vibrant_right": (810, 1096, 1180, 1240),
    "word_vibrant_upper": (1190, 1096, 1570, 1240),
}

# page_writing (index 3), section 2: 5 cursive sentence row-cells. Each cell
# includes its pre-printed prompt (expected, not an error).
CURSIVE_CROPS = {
    "cursive_01": (130, 1390, 1570, 1520),
    "cursive_02": (130, 1530, 1570, 1660),
    "cursive_03": (130, 1670, 1570, 1800),
    "cursive_04": (130, 1810, 1570, 1940),
    "cursive_05": (130, 1950, 1570, 2080),
}


if __name__ == "__main__":
    print("INSIDE-OUT Configuration")
    print("=" * 40)
    print(f"\nProject: {config.project_name} v{config.version}")
    print("\nData Config:")
    print(f"  Image size: {config.data.image_size}")
    print(
        f"  Train/Val/Test: {config.data.train_ratio}/"
        f"{config.data.val_ratio}/{config.data.test_ratio}"
    )
    print("\nCNN Config:")
    print(f"  Features: {config.cnn.num_features}")
    print(f"  Pretrained: {config.cnn.use_pretrained}")
    print("\nHMM Config:")
    print(f"  States: {config.hmm.n_states}")
    print("\nTraining Config:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  K-folds: {config.training.n_folds}")
