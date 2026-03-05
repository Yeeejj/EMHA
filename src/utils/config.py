"""
Configuration Module
Centralized configuration for model hyperparameters and paths.
"""

from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_dir: str = "DATA/RAW"
    labeled_dir: str = "DATA/LABELED"
    processed_dir: str = "DATA/PROCESSED"
    splits_dir: str = "DATA/SPLITS"
    metadata_dir: str = "DATA/METADATA"

    image_size: Tuple[int, int] = (224, 224)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    target_size: Tuple[int, int] = (224, 224)
    binarize_threshold: int = None  # None = Otsu's method
    denoise_kernel_size: int = 3
    normalize: bool = True


@dataclass
class CNNConfig:
    """CNN model configuration."""
    input_channels: int = 1  # Grayscale
    num_features: int = 256
    dropout_rate: float = 0.5

    # Architecture options
    use_pretrained: bool = False
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
    """Emotion labeling thresholds."""
    # DASS-21 Depression subscale
    dass_depression_threshold: float = 14.0  # Moderate+

    # Happiness scale thresholds
    happiness_high: float = 40.0
    happiness_low: float = 30.0


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
        """Create directories if they don't exist."""
        for dir_path in [
            self.data.raw_dir,
            self.data.labeled_dir,
            self.data.processed_dir,
            self.data.splits_dir,
            self.data.metadata_dir,
            self.training.checkpoint_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = Config()


if __name__ == "__main__":
    print("INSIDE-OUT Configuration")
    print("=" * 40)
    print(f"\nProject: {config.project_name} v{config.version}")
    print(f"\nData Config:")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Train/Val/Test: {config.data.train_ratio}/{config.data.val_ratio}/{config.data.test_ratio}")
    print(f"\nCNN Config:")
    print(f"  Features: {config.cnn.num_features}")
    print(f"  Pretrained: {config.cnn.use_pretrained}")
    print(f"\nHMM Config:")
    print(f"  States: {config.hmm.n_states}")
    print(f"\nTraining Config:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  K-folds: {config.training.n_folds}")
