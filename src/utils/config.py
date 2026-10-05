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

    # CropDataset (src/data/dataloader.py): label string -> class index, and
    # DataLoader worker processes.
    label_to_index: dict = field(default_factory=lambda: {"HAPPY": 0, "SAD": 1})
    num_workers: int = 2
    image_size: Tuple[int, int] = (224, 224)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class PreprocessingConfig:
    """Model-input preprocessing (Stage D) -- see src/preprocessing/pipeline.py.

    No Otsu binarization, no deskew, no square/stretch resize in the model
    path (CLAUDE.md Accuracy Strategy) -- ink_mask_for_stats() is the only
    place Otsu may appear, and only for descriptive statistics, never the
    model-input image.

    scale_factor/canvas_size are computed from real measured crop pixel
    sizes, not the project's nominal 300 dpi (which doesn't match the
    confirmed real ~200 dpi): drawing's largest crop (675x1016) and
    cursive's (1342x174) both land near 448px on their long side; word
    (368x118, already smaller than 448) is kept at native resolution
    rather than upscaled.

    min_crop_size/blank_threshold are QC thresholds (src/data/qc.py), not
    used by pipeline.py -- kept here since qc.py already depends on them.
    """

    background_blur_ksize: int = 51
    denoise: str = "median3"  # "median3" | "none"
    scale_factor: dict = field(
        default_factory=lambda: {"drawing": 0.44, "word": 1.0, "cursive": 0.33}
    )
    canvas_size: dict = field(
        default_factory=lambda: {
            "drawing": (297, 448),
            "word": (368, 118),
            "cursive": (443, 58),
        }
    )
    invert: bool = True
    min_crop_size: Tuple[int, int] = (80, 80)
    blank_threshold: float = 245.0  # grayscale mean above this = blank


@dataclass
class CNNConfig:
    """CNN model configuration (Stage F) -- see src/models/cnn.py.

    backbone "resnet18" (primary) or "simple" (old 4-block CNN, ablation).
    use_pretrained loads torchvision ResNet18_Weights[weights]; conv1 becomes
    1-channel by summing the RGB filters. Everything before the first of
    trainable_blocks (stem, layer1, layer2) is frozen, with its BatchNorm
    kept in eval mode. seq_layer is the block whose feature map, averaged
    over height, gives the column sequence for the HMM (layer3: 256 ch,
    stride 16). imagenet_input_norm rescales the eval-transform input
    (mean 0.5, std 0.5) to ImageNet grayscale statistics before the
    pretrained stem.
    """

    backbone: str = "resnet18"  # "resnet18" | "simple"
    use_pretrained: bool = True
    weights: str = "IMAGENET1K_V1"
    trainable_blocks: Tuple[str, ...] = ("layer3", "layer4")
    seq_layer: str = "layer3"  # "layer3" | "layer4"
    imagenet_input_norm: bool = True
    input_channels: int = 1  # Grayscale
    num_features: int = 256
    num_classes: int = 2
    dropout_rate: float = 0.5


@dataclass
class HMMConfig:
    """Per-class GaussianHMM classifier (Stage F) -- see src/models/hmm.py.

    Frames are standardized (standardize) and reduced to pca_components by
    PCA, both fit on training frames only. One hmmlearn GaussianHMM per class
    is fit n_restarts times (random_state = seed + restart); the restart with
    the best training log-likelihood is kept. topology "ergodic" (all
    transitions) or "left_right" (banded: stay or move one state right; the
    zero transitions stay zero during EM). Decision score = length-normalized
    LL_sad - LL_happy + log prior ratio; Platt-calibrated on held-out scores.
    """

    n_states: int = 4
    covariance_type: str = "diag"
    n_iter: int = 200
    tol: float = 1e-3
    min_covar: float = 1e-3
    n_restarts: int = 5
    topology: str = "ergodic"  # "ergodic" | "left_right"
    pca_components: int = 16
    standardize: bool = True


@dataclass
class HybridConfig:
    """CNN-HMM (titled method, Stage F) -- see src/training/run_hybrid.py.

    Per family and outer fold, the HMM setting (n_states x pca x topology)
    is chosen by inner-validation participant macro-F1 with
    selection_restarts restarts per class (decided with the thesis author
    2026-10-05), then the winner is refit on inner-train with
    HMMConfig.n_restarts and Platt-calibrated on inner-validation.
    families enter the fused prediction; secondary_family is only run with
    --include-drawing and is reported separately.
    """

    n_states_grid: Tuple[int, ...] = (2, 3, 4, 6)
    pca_grid: Tuple[int, ...] = (8, 16, 32)
    topology_grid: Tuple[str, ...] = ("ergodic", "left_right")
    selection_restarts: int = 2
    families: Tuple[str, ...] = ("word", "cursive")
    secondary_family: str = "drawing"
    output_subdir: str = "hybrid"
    model_subdir: str = "hmm"


@dataclass
class TrainingConfig:
    """Training configuration. Checkpoints are written under Config.paths.models_dir."""

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    # CNN fine-tuning (Trainer.fit, protocol section 5): AdamW with separate
    # learning rates for the trainable backbone blocks and the head, cosine
    # schedule over `epochs`, early stopping on inner-validation participant
    # macro-F1 (patience / min_delta), class-balanced cross-entropy.
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    class_weight: Optional[str] = "balanced"  # "balanced" | None
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
class CVConfig:
    """Participant-level cross-validation (Stage E) -- see src/training/splits.py.

    Revised with the thesis author on 2026-10-05, before any model result
    (supersedes the stored-inner-split design of commit 863bc9a):
    n_splits outer folds over ALL labeled participants, one row per
    participant in folds.csv, stratified on label x in_middle_band
    (equivalent to label x in_primary_analysis, its complement) and seeded by
    TrainingConfig.seed. The inner validation split (early stopping, Platt
    calibration) is computed at run time by splits.inner_split from each
    fold's filtered training IDs (inner_val_fraction, seed + fold), so it
    adapts when runners restrict to qc_passed / primary participants.

    PROVISIONAL until labeling is complete: more participants are still
    being collected, so folds.csv must be regenerated (--force) once all are
    labeled, before the protocol is frozen.
    """

    n_splits: int = 5
    inner_val_fraction: float = 0.15
    # --subset pilot: participants whose 3-digit code is in this inclusive
    # range (decided with the thesis author 2026-10-05), intersected with the
    # analysis set.
    pilot_id_range: Tuple[int, int] = (1, 100)
    stratify_columns: Tuple[str, ...] = ("label", "in_middle_band")
    folds_filename: str = "folds.csv"


@dataclass
class AggregateConfig:
    """Crop -> participant aggregation (Stage E) -- see src/training/aggregate.py.

    method "mean_prob" is primary (CLAUDE.md); "mean_logit" and
    "majority_vote" are secondary only. pred is SAD when prob_sad >=
    threshold. logit_eps clips probabilities away from 0/1 for mean_logit.
    """

    method: str = "mean_prob"
    threshold: float = 0.5
    logit_eps: float = 1e-6


@dataclass
class BaselineConfig:
    """Majority + logistic-regression baselines (Stage F) -- see
    src/training/run_baselines.py.

    Pipeline StandardScaler -> (PCA, embeddings only) -> LogisticRegression.
    C is chosen from C_grid by inner_cv-fold stratified CV (macro-F1) on the
    outer-train participants only, then refit on all of them. PCA keeps
    min(pca_components, smallest inner-train size - 1) components.
    """

    C_grid: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    inner_cv: int = 5
    class_weight: Optional[str] = "balanced"
    max_iter: int = 5000
    pca_components: int = 64
    # Exact SVD: faster than "auto" (randomized) on ~100 participants x
    # 512-1536 dims, and fully deterministic.
    pca_svd_solver: str = "full"
    scoring: str = "f1_macro"
    output_subdir: str = "baselines"


@dataclass
class LabelingConfig:
    """Labels are read verbatim from the tabulation export, never recomputed
    (CLAUDE.md Non-Negotiable 3) — see src/data/labeler.py.

    source_csv/output_csv default to None, meaning "derive from
    Config.paths.metadata_dir" (questionnaire_export.csv / labels.csv);
    set them explicitly to override.
    """

    source_csv: Optional[Path] = None
    output_csv: Optional[Path] = None
    id_column: str = "re"
    score_column: str = "adjusted_total"
    label_column: str = "label"
    label_map: dict = field(default_factory=lambda: {"HAPPY": "HAPPY", "SAD": "SAD"})
    cutoff: float = 72.0
    analysis_design: str = "extreme_groups"  # "extreme_groups" | "full_sample"
    # Set 2026-10-04 per DOCS/decisions/analysis_design.txt: 0.30 (the prior
    # default) left SAD's primary count below min_per_class; 0.20 is the
    # only candidate fraction in the preview table that clears it for both
    # classes. Decided from labels.csv score data only, before any model
    # result existed.
    middle_band_fraction: float = 0.20
    min_per_class: int = 100


@dataclass
class ReportConfig:
    """src/analysis/questionnaire_report.py — reliability/balance reporting.

    item_columns/reverse_items describe the raw 24-item export columns
    (lowercase q1..q24, as they actually appear in the sheet), independent
    of LabelingConfig.score_column (the already-computed composite).
    """

    n_bootstrap: int = 1000
    item_columns: Tuple[str, ...] = tuple(f"q{i}" for i in range(1, 25))
    reverse_items: Tuple[str, ...] = (
        "q1",
        "q3",
        "q5",
        "q7",
        "q9",
        "q11",
        "q14",
        "q15",
        "q17",
        "q18",
        "q22",
        "q24",
    )


@dataclass
class IngestConfig:
    """src/data/ingest.py — raw scan validation and checksums.

    expected_dpi=200 matches the real scans (confirmed via PIL on multiple
    participants' files: 1700x2200px, ~199.9996 DPI isotropic) — not the
    old, apparently-incorrect "~74.5x81.1 anisotropic" figure, and not 300.
    """

    expected_dpi: int = 200
    size_tolerance: float = 0.02


@dataclass
class CropConfig:
    """src/data/crop_manifest.py — crop indexing, verification, QC.

    expected_size_px measured directly from real files (confirmed matching
    the hardcoded boxes in src/cropping/p3/crop_pictures.py and
    p4/crop_p4.py exactly, consistent across participants). D1-D4 differ
    from each other because the drawing page's top row (D1, D2) is shorter
    than the bottom row (D3, D4). crops_dirs is not stored here -- it's
    derived at use time from Config.paths.raw3_dir/raw4_dir, consistent
    with "every path derives from one data root."
    """

    expected_size_px: dict = field(
        default_factory=lambda: {
            "D1": (674, 669),
            "D2": (662, 674),
            "D3": (675, 1016),
            "D4": (664, 1016),
            **{
                f"W{n}_{s}": (368, 118) for n in range(1, 6) for s in ("LH", "RH", "UC")
            },
            **{f"CS{k}": (1342, 174) for k in range(1, 6)},
        }
    )
    ink_threshold: int = 128  # for statistics only; never used to binarize a crop


@dataclass
class AugmentConfig:
    """Train-time augmentation (Stage D) -- see src/data/transforms.py.

    Translation, brightness/contrast, and light blur only (CLAUDE.md
    Accuracy Strategy): never rotation, affine scale/shear, flips,
    RandomResizedCrop, or elastic transforms.
    """

    enabled: bool = True
    max_translate_px: int = 8
    brightness: float = 0.15
    contrast: float = 0.15
    blur_prob: float = 0.2
    blur_sigma: Tuple[float, float] = (0.1, 0.6)


@dataclass
class FeaturesConfig:
    """Handcrafted features (Stage D) -- see src/features/handcrafted.py.

    px_to_mm derives from IngestConfig.expected_dpi (200, confirmed on the
    real scans), not the nominal 300 dpi in the original brief. Crops are
    unscaled cuts of the scans, so the scan dpi applies to them directly.

    ink_threshold is applied to the background-flattened crop (paper ~255).
    170 rather than CropConfig's 128: on a 60-crop sample of real
    word/cursive crops, 128 fragmented light-ink writers (p90 component
    count 44 vs 21 at 170). Chosen from crop images only, before any model
    result. border_* remove the pre-printed box rules, which sit within
    14 px of the crop edge on the sampled crops; only near-full rows/columns
    inside the edge band are cleared, so drawn lines in the interior survive.

    imputation: "task_family_median" = a crop's missing value is filled with
    the median of the SAME participant's other crops in the same task family
    (never pooled across participants, so no cross-fold leakage). If all of
    that participant's crops in the family are missing, fallback_value is
    used. Every fill is counted in handcrafted_imputation_log.csv.
    """

    px_to_mm: float = 25.4 / IngestConfig.expected_dpi
    ink_threshold: int = 170
    min_component_px: int = 8
    border_band_px: int = 16
    border_line_frac: float = 0.5
    slant_grad_sigma: float = 1.0
    slant_min_grad_frac: float = 0.2
    slant_max_deg: float = 45.0
    slant_bin_deg: float = 1.0
    slant_smooth_bins: int = 5
    slant_refine_deg: float = 5.0
    slant_min_pixels: int = 50
    xheight_profile_frac: float = 0.5
    xheight_merge_gap_px: int = 3
    xheight_min_band_px: int = 3
    # Stroke-level features (src/features/strokes.py). Skeleton paths shorter
    # than skeleton_min_path_px (~1.3 mm) are treated as spurs and ignored.
    # tremor_window_px is the coarse Gaussian scale (px of arc length) that
    # defines the smooth "intended" trajectory; tremor_fine_sigma_px only
    # removes pixel-staircase quantization. Tremor needs paths of at least
    # 4 * tremor_window_px so the edge-trimmed core is non-empty.
    skeleton_min_path_px: int = 10
    tremor_window_px: int = 6
    tremor_fine_sigma_px: float = 1.5
    # Drawing features. faint_gray_range is [lo, hi) on the flattened crop:
    # lighter than ink (ink_threshold) but darker than paper (~255). Faint
    # pixels within erasure_ink_margin_px of ink (antialiased stroke edges)
    # are excluded, and only regions surviving an erasure_open_px square
    # opening count -- smudges are areal; faint thin strokes (seen as light
    # hatching on real drawings) are removed by the opening. Erasure is read
    # from the crop flattened at erasure_background_ksize (~25 mm): the
    # model-path kernel (51 px, ~6.5 mm) flattens smudges of its own size
    # back to paper white.
    # erasure_open_px = 9 (~1.1 mm) and excluding the border band: with 5 px
    # and no band, 22/400 real drawings scored > 0, all from scan-blurred
    # light strokes ~5-6 px wide or the scan-edge shadow; with 9 px + band,
    # 1/400 (checked on crop images only, no labels).
    faint_gray_range: Tuple[int, int] = (170, 240)
    erasure_background_ksize: int = 201
    erasure_ink_margin_px: int = 2
    erasure_open_px: int = 9
    empty_space_grid: int = 8
    imputation: str = "task_family_median"
    fallback_value: float = 0.0
    output_subdir: str = "features"


@dataclass
class EmbeddingConfig:
    """Frozen ResNet18 embeddings (Stage D) -- see src/features/embeddings.py.

    device "auto" = "cuda" if torch.cuda.is_available() else "cpu" (resolved
    at run time so importing config never imports torch). weights names a
    torchvision ResNet18_Weights member; None = random init (tests only).
    pca_components is NOT used here: PCA/scalers are fit inside CV folds
    (Stage F) on training participants only.
    """

    batch_size: int = 32
    device: str = "auto"
    weights: Optional[str] = "IMAGENET1K_V1"
    pca_components: int = 64
    output_subdir: str = "embeddings"


@dataclass
class HandwritingEmbeddingConfig:
    """EXPLORATORY handwriting-encoder embeddings (Stage D, optional) -- see
    src/features/embeddings_handwriting.py. Never part of the pre-registered
    ensemble.

    model_name chosen with the thesis author on 2026-10-05:
    microsoft/trocr-base-handwritten (MIT; ViT-B/16 encoder fine-tuned on
    IAM). Its processor squashes every crop to 384x384, destroying absolute
    size and aspect ratio -- the reason this model is exploratory.
    model_revision "main" until pinned; the commit actually loaded is stored
    in every cache file. uninvert=True feeds dark-on-white crops (TrOCR's
    training polarity) when the processed crops are stored inverted.
    """

    model_name: str = "microsoft/trocr-base-handwritten"
    model_revision: str = "main"
    batch_size: int = 16
    device: str = "auto"
    task_families: Tuple[str, ...] = ("word", "cursive")
    uninvert: bool = True
    output_prefix: str = "handwriting"
    output_subdir: str = "embeddings"


@dataclass
class RunnerConfig:
    """One resumable command for pilot and full runs (src/pipeline_runner.py).

    default_steps run in this order for each analysis set (primary first);
    embeddings and handcrafted do not depend on the analysis set and run
    once per subset. A finished step writes a JSON marker under
    results/<state_subdir>/<subset>/; a rerun skips it while its outputs
    exist and its settings match, and refuses on a mismatch (a Deviation,
    never silently rerun). The settings fingerprint hashes the whole Config
    except fingerprint_exclude (machine paths and run-irrelevant sections).
    predict_middle: primary-set runs also score the middle band
    (EVALUATION_PROTOCOL.md section 6.2). decision_file is relative to the
    repo root and must exist, like the protocol-frozen tag, before any run.
    """

    default_steps: Tuple[str, ...] = (
        "embeddings",
        "handcrafted",
        "baselines",
        "cnn",
        "cnn_shuffled",
        "hybrid",
        "ensemble",
        "evaluate",
        "permutation",
        "secondary",
        "gradcam",
        "report",
    )
    default_analyses: Tuple[str, ...] = ("primary", "full")
    predict_middle: bool = True
    decision_file: str = "DOCS/decisions/analysis_design.txt"
    state_subdir: str = "pipeline"
    device: str = "auto"
    fingerprint_exclude: Tuple[str, ...] = (
        "paths",
        "runner",
        "demo",
        "smoke",
        "package",
    )


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
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    aggregate: AggregateConfig = field(default_factory=AggregateConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    handwriting_embedding: HandwritingEmbeddingConfig = field(
        default_factory=HandwritingEmbeddingConfig
    )

    project_name: str = "INSIDE-OUT"
    version: str = "1.0.0"
    runner: RunnerConfig = field(default_factory=RunnerConfig)

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
    print(f"  background_blur_ksize: {config.preprocessing.background_blur_ksize}")
    print(f"  denoise:               {config.preprocessing.denoise}")
    print(f"  scale_factor:          {config.preprocessing.scale_factor}")
    print(f"  canvas_size:           {config.preprocessing.canvas_size}")
    print(f"  invert:                {config.preprocessing.invert}")
    print(f"  min_crop_size:         {config.preprocessing.min_crop_size}")
    print(f"  blank_threshold:       {config.preprocessing.blank_threshold}")

    print("\nCNN Config:")
    print(f"  backbone:            {config.cnn.backbone}")
    print(f"  use_pretrained:      {config.cnn.use_pretrained}")
    print(f"  trainable_blocks:    {config.cnn.trainable_blocks}")
    print(f"  seq_layer:           {config.cnn.seq_layer}")
    print(f"  num_features:        {config.cnn.num_features}")
    print(f"  dropout_rate:        {config.cnn.dropout_rate}")

    print("\nHMM Config:")
    print(f"  n_states:         {config.hmm.n_states}")
    print(f"  n_iter:           {config.hmm.n_iter}")
    print(f"  covariance_type:  {config.hmm.covariance_type}")
    print(f"  n_restarts:       {config.hmm.n_restarts}")
    print(f"  topology:         {config.hmm.topology}")
    print(f"  pca_components:   {config.hmm.pca_components}")

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
    derived = "(derived from metadata_dir)"
    print(f"  source_csv:           {config.labeling.source_csv or derived}")
    print(f"  output_csv:           {config.labeling.output_csv or derived}")
    print(f"  id_column:            {config.labeling.id_column}")
    print(f"  score_column:         {config.labeling.score_column}")
    print(f"  label_column:         {config.labeling.label_column}")
    print(f"  label_map:            {config.labeling.label_map}")
    print(f"  cutoff:               {config.labeling.cutoff}")
    print(f"  analysis_design:      {config.labeling.analysis_design}")
    print(f"  middle_band_fraction: {config.labeling.middle_band_fraction}")
    print(f"  min_per_class:        {config.labeling.min_per_class}")

    print("\nReport Config:")
    print(f"  n_bootstrap:   {config.report.n_bootstrap}")
    print(f"  item_columns:  {config.report.item_columns}")
    print(f"  reverse_items: {config.report.reverse_items}")

    print("\nIngest Config:")
    print(f"  expected_dpi:   {config.ingest.expected_dpi}")
    print(f"  size_tolerance: {config.ingest.size_tolerance}")

    print("\nCrop Config:")
    print(f"  expected_size_px: {config.crop.expected_size_px}")
    print(f"  ink_threshold:    {config.crop.ink_threshold}")

    print("\nAugment Config:")
    print(f"  enabled:          {config.augment.enabled}")
    print(f"  max_translate_px: {config.augment.max_translate_px}")
    print(f"  brightness:       {config.augment.brightness}")
    print(f"  contrast:         {config.augment.contrast}")
    print(f"  blur_prob:        {config.augment.blur_prob}")
    print(f"  blur_sigma:       {config.augment.blur_sigma}")

    print("\nFeatures Config:")
    print(f"  px_to_mm:         {config.features.px_to_mm:.5f}")
    print(f"  ink_threshold:    {config.features.ink_threshold}")
    print(f"  min_component_px: {config.features.min_component_px}")
    print(f"  imputation:       {config.features.imputation}")
