"""
Configuration Module — INSIDE-OUT / EMHA
All hyperparameters live here in dataclasses. Modules run as python -m src.*
from the project root. See CLAUDE.md for the data layout and labeling rules.

Every filesystem path derives from Config.paths (PathsConfig), which in turn
derives everything from two roots: data_root (inputs — scans, crops, labels;
may be read-only, e.g. Kaggle's /kaggle/input) and output_root (everything
the pipeline writes — results, model checkpoints; must be writable). Override
either root with the EMHA_DATA_ROOT / EMHA_OUTPUT_ROOT environment variables
so the same code runs unchanged locally, on Colab, and on Kaggle.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path

# src/utils/config.py -> src/utils -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    """Two roots; every other path is derived from one of them.

    data_root holds inputs: raw-3Page/raw-4Page scans and crops (read-only,
    never created or written to by this config) plus metadata/processed.
    output_root holds everything the pipeline writes: results and model
    checkpoints. Kept separate from data_root so a read-only data mount
    (e.g. Kaggle's /kaggle/input) never blocks writing outputs.
    """

    data_root: Path = field(
        default_factory=lambda: Path(os.environ.get("EMHA_DATA_ROOT", "DATASET"))
    )
    output_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("EMHA_OUTPUT_ROOT", str(_REPO_ROOT))
        )
    )

    @property
    def raw3_dir(self) -> Path:
        """Drawing-exercise scans and crops. READ-ONLY — never created here."""
        return self.data_root / "raw-3Page"

    @property
    def raw4_dir(self) -> Path:
        """Writing-exercise scans and crops. READ-ONLY — never created here."""
        return self.data_root / "raw-4Page"

    @property
    def metadata_dir(self) -> Path:
        return self.data_root / "metadata"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def results_dir(self) -> Path:
        return self.output_root / "results"

    @property
    def models_dir(self) -> Path:
        return self.output_root / "models"


@dataclass
class DataConfig:
    """Non-path dataset configuration. See Config.paths for filesystem roots."""

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

    input_channels: int = 1  # Grayscale
    num_features: int = 256
    dropout_rate: float = 0.5
    use_pretrained: bool = True  # ResNet18 backbone (Section A Rule 6)
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
    """Training configuration. Checkpoints are written under Config.paths.models_dir."""

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.001
    n_folds: int = 5
    random_state: int = 42
    # Global reproducibility seed for src.utils.seed.set_seed() (random, numpy,
    # torch). Distinct from random_state, which is passed directly into
    # individual sklearn calls (StratifiedKFold, train_test_split, etc.).
    seed: int = 42
    save_best_only: bool = True


@dataclass
class LabelingConfig:
    """FINALE 24-item self-report scoring (CLAUDE.md Non-Negotiable 3: labels
    are read from the export, never recomputed; this scheme is reference-only).

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
    adjusted_total_threshold: int = 72  # HAPPY if >= 72, SAD otherwise


def _try_mkdir(path: Path) -> bool:
    """Create path (with parents). Return False instead of raising on failure."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


@dataclass
class Config:
    """Master configuration class."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

    project_name: str = "INSIDE-OUT"
    version: str = "1.0.0"

    def ensure_output_dirs(self) -> None:
        """Create only the directories the pipeline is allowed to write to.

        Never creates anything under raw-3Page or raw-4Page (read-only,
        CLAUDE.md Non-Negotiable 2). metadata/processed live under data_root,
        which may itself be read-only (e.g. Kaggle's /kaggle/input); if so,
        this logs a note and skips them instead of raising. results/models
        live under output_root, which is assumed writable.
        """
        metadata_ok = _try_mkdir(self.paths.metadata_dir)
        processed_ok = _try_mkdir(self.paths.processed_dir)
        if not (metadata_ok and processed_ok):
            print(
                f"  NOTE: data_root ({self.paths.data_root}) appears read-only; "
                "metadata/processed were not created — assuming they already "
                "exist or are supplied externally."
            )

        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)


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
    "draw_circles": (0.0706, 0.2318, 0.4765, 0.5500),  # top-left
    "draw_dots": (0.5235, 0.2318, 0.9294, 0.5500),  # top-right
    "draw_person": (0.0706, 0.6273, 0.4765, 0.9455),  # bottom-left
    "draw_house": (0.5235, 0.6273, 0.9294, 0.9455),  # bottom-right
}

# page_writing (page index 3), section 1: 5×3 word table (15 cells).
# Rows: content, melancholic, optimistic, disconnected, vibrant.
# Columns: left-hand, right-hand, uppercase.
WORD_CROPS = {
    "word_content_left": (0.2529, 0.2000, 0.4706, 0.2655),
    "word_content_right": (0.4765, 0.2000, 0.6941, 0.2655),
    "word_content_upper": (0.7000, 0.2000, 0.9235, 0.2655),
    "word_melancholic_left": (0.2529, 0.2745, 0.4706, 0.3400),
    "word_melancholic_right": (0.4765, 0.2745, 0.6941, 0.3400),
    "word_melancholic_upper": (0.7000, 0.2745, 0.9235, 0.3400),
    "word_optimistic_left": (0.2529, 0.3491, 0.4706, 0.4145),
    "word_optimistic_right": (0.4765, 0.3491, 0.6941, 0.4145),
    "word_optimistic_upper": (0.7000, 0.3491, 0.9235, 0.4145),
    "word_disconnected_left": (0.2529, 0.4236, 0.4706, 0.4891),
    "word_disconnected_right": (0.4765, 0.4236, 0.6941, 0.4891),
    "word_disconnected_upper": (0.7000, 0.4236, 0.9235, 0.4891),
    "word_vibrant_left": (0.2529, 0.4982, 0.4706, 0.5636),
    "word_vibrant_right": (0.4765, 0.4982, 0.6941, 0.5636),
    "word_vibrant_upper": (0.7000, 0.4982, 0.9235, 0.5636),
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

    print("\nPaths:")
    print(f"  data_root:     {config.paths.data_root}")
    print(f"  output_root:   {config.paths.output_root}")
    print(f"  raw3_dir:      {config.paths.raw3_dir}  (READ-ONLY)")
    print(f"  raw4_dir:      {config.paths.raw4_dir}  (READ-ONLY)")
    print(f"  metadata_dir:  {config.paths.metadata_dir}")
    print(f"  processed_dir: {config.paths.processed_dir}")
    print(f"  results_dir:   {config.paths.results_dir}")
    print(f"  models_dir:    {config.paths.models_dir}")

    print("\nData Config:")
    print(f"  image_size: {config.data.image_size}")
    print(
        f"  train/val/test ratio: "
        f"{config.data.train_ratio}/{config.data.val_ratio}/{config.data.test_ratio}"
    )

    print("\nPreprocessing Config:")
    print(f"  target_size:         {config.preprocessing.target_size}")
    print(f"  binarize_threshold:  {config.preprocessing.binarize_threshold}")
    print(f"  denoise_kernel_size: {config.preprocessing.denoise_kernel_size}")
    print(f"  normalize:           {config.preprocessing.normalize}")
    print(f"  min_crop_size:       {config.preprocessing.min_crop_size}")
    print(f"  blank_threshold:     {config.preprocessing.blank_threshold}")
    print(f"  skip_skew_prefixes:  {config.preprocessing.skip_skew_prefixes}")

    print("\nCNN Config:")
    print(f"  input_channels:      {config.cnn.input_channels}")
    print(f"  num_features:        {config.cnn.num_features}")
    print(f"  dropout_rate:        {config.cnn.dropout_rate}")
    print(f"  use_pretrained:      {config.cnn.use_pretrained}  (ResNet18)")
    print(f"  pretrained_backbone: {config.cnn.pretrained_backbone}")
    print(f"  freeze_backbone:     {config.cnn.freeze_backbone}")

    print("\nHMM Config:")
    print(f"  n_states:         {config.hmm.n_states}")
    print(f"  n_iter:           {config.hmm.n_iter}")
    print(f"  covariance_type:  {config.hmm.covariance_type}")

    print("\nTraining Config:")
    print(f"  batch_size:     {config.training.batch_size}")
    print(f"  epochs:         {config.training.epochs}")
    print(f"  learning_rate:  {config.training.learning_rate}")
    print(f"  weight_decay:   {config.training.weight_decay}")
    print(f"  patience:       {config.training.patience}")
    print(f"  min_delta:      {config.training.min_delta}")
    print(f"  n_folds:        {config.training.n_folds}")
    print(f"  random_state:   {config.training.random_state}")
    print(f"  seed:           {config.training.seed}")
    print(f"  save_best_only: {config.training.save_best_only}")

    print("\nLabeling Config:")
    print(f"  happiness_items: {config.labeling.happiness_items}")
    print(f"  sadness_items:   {config.labeling.sadness_items}")
    print(
        f"  likert_min/max:  {config.labeling.likert_min}/{config.labeling.likert_max}"
    )
    threshold = config.labeling.adjusted_total_threshold
    print(f"  threshold (adjusted_total >= {threshold} -> HAPPY)")
    print("  NO NEUTRAL class")
