"""
Handcrafted, interpretable handwriting features — Stage D.

Per crop: ink amount and darkness, stroke width, ink extent, component
count, and skeleton-based stroke features (src/features/strokes.py) for
every crop; slant, baseline angle, core-zone (x-)height, inter-component
gap, and right margin for word and cursive crops; placement, size, detail,
empty space, and erasure features for drawing crops. Per
participant: mean and std of every feature within each task family.

Input is the original crop (unscaled cut of a 200 dpi scan) listed in
crop_manifest.csv, background-flattened with
src.preprocessing.pipeline.flatten_background — never binarized into the
model path; the ink mask here is used only to measure. Raw crops are opened
read-only. Only qc_passed participants (participants.csv) are included.

Outputs (results/features/):
    handcrafted_crop.csv            one row per crop
    handcrafted_participant.csv     one row per participant
    handcrafted_imputation_log.csv  counts of every imputed value

Run from the project root:

    python -m src.features.handcrafted
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from src.features.strokes import (
    DRAWING_FEATURES,
    STROKE_FEATURES,
    drawing_features,
    stroke_features,
)
from src.preprocessing.pipeline import _load_qc_dropped, flatten_background
from src.utils.config import config

COMMON_FEATURES = (
    "ink_ratio",
    "ink_darkness_mean",
    "ink_darkness_std",
    "stroke_width_median_mm",
    "stroke_width_iqr_mm",
    "ink_bbox_height_mm",
    "ink_bbox_width_mm",
    "n_components",
)
TEXT_FEATURES = (
    "baseline_angle_deg",
    "slant_deg",
    "letter_height_mm",
    "inter_component_gap_mm",
    "right_margin_mm",
)
TEXT_FAMILIES = ("word", "cursive")
FAMILIES = ("drawing", "word", "cursive")
ID_COLUMNS = ("participant_id", "cell", "task_family")


ALL_FEATURES = COMMON_FEATURES + STROKE_FEATURES + TEXT_FEATURES + DRAWING_FEATURES


def features_for_family(task_family: str) -> tuple:
    """Feature names computed for a task family."""
    if task_family in TEXT_FAMILIES:
        return COMMON_FEATURES + STROKE_FEATURES + TEXT_FEATURES
    return COMMON_FEATURES + STROKE_FEATURES + DRAWING_FEATURES


def _remove_border_lines(mask: np.ndarray, band_px: int, frac: float) -> np.ndarray:
    """Clear pre-printed box rules: near-full ink rows/columns in the edge band.

    Only rows/columns within band_px of an edge whose ink fraction exceeds
    frac are cleared, so handwriting and interior drawn lines are kept.
    """
    out = mask.copy()
    height, width = mask.shape
    row_frac = mask.mean(axis=1)
    col_frac = mask.mean(axis=0)
    for i in range(height):
        if min(i, height - 1 - i) < band_px and row_frac[i] > frac:
            out[i, :] = False
    for j in range(width):
        if min(j, width - 1 - j) < band_px and col_frac[j] > frac:
            out[:, j] = False
    return out


def _components(mask: np.ndarray, min_px: int) -> tuple:
    """Return (cleaned mask, stats of kept components) with 8-connectivity.

    stats rows are cv2 CC_STAT (left, top, width, height, area).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_px]
    cleaned = np.isin(labels, keep)
    return cleaned, stats[keep]


def ink_mask(flat: np.ndarray, fcfg) -> tuple:
    """Measurement-only ink mask of a flattened crop: (mask, component stats)."""
    mask = flat < fcfg.ink_threshold
    mask = _remove_border_lines(mask, fcfg.border_band_px, fcfg.border_line_frac)
    return _components(mask, fcfg.min_component_px)


def stroke_widths_px(mask: np.ndarray) -> np.ndarray:
    """Local stroke widths (px) sampled on the skeleton.

    Width at a skeleton pixel = 2 * EDT - 1, where EDT is the Euclidean
    distance to the nearest background pixel (exact for odd widths, -1 px
    for even widths).
    """
    if not mask.any():
        return np.array([])
    dist = ndimage.distance_transform_edt(mask)
    skel = skeletonize(mask)
    return 2.0 * dist[skel] - 1.0


def estimate_slant_deg(flat: np.ndarray, mask: np.ndarray, fcfg) -> float:
    """Dominant slant of near-vertical strokes from gradient orientations.

    Orientation-histogram slant estimation in the style of Ding, Kimura,
    Miyake & Shridhar, "Evaluation and improvement of slant estimation for
    handwritten words", ICDAR 1999, pp. 753-756 (edge-direction histogram
    of near-vertical strokes), using image gradients instead of chain codes.
    The shear-projection alternative (Vinciarelli & Luettin, Pattern
    Recognition Letters 22(9):1043-1050, 2001) is not used.

    At each stroke-edge pixel, the stroke tangent is perpendicular to the
    gradient (gx, gy), and its angle from vertical is atan(gy / gx). Pixels
    with angle within +/-slant_max_deg contribute a magnitude-weighted
    histogram; the smoothed peak is refined by the weighted mean within
    +/-slant_refine_deg. Positive = forward (rightward) slant. The image is
    never sheared or rotated. Returns NaN with too few edge pixels.
    """
    smooth = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), fcfg.slant_grad_sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    near_ink = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not near_ink.any() or mag.max() == 0:
        return float("nan")
    strong = near_ink & (mag >= fcfg.slant_min_grad_frac * mag[near_ink].max())
    strong &= gx != 0

    angles = np.degrees(np.arctan(gy[strong] / gx[strong]))
    weights = mag[strong]
    keep = np.abs(angles) <= fcfg.slant_max_deg
    angles, weights = angles[keep], weights[keep]
    if angles.size < fcfg.slant_min_pixels:
        return float("nan")

    edges = np.arange(
        -fcfg.slant_max_deg, fcfg.slant_max_deg + fcfg.slant_bin_deg, fcfg.slant_bin_deg
    )
    hist, _ = np.histogram(angles, bins=edges, weights=weights)
    kernel = np.ones(fcfg.slant_smooth_bins) / fcfg.slant_smooth_bins
    hist = np.convolve(hist, kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0
    peak = centers[int(np.argmax(hist))]

    window = np.abs(angles - peak) <= fcfg.slant_refine_deg
    return float(np.average(angles[window], weights=weights[window]))


def core_zone_height_px(mask: np.ndarray, fcfg) -> float:
    """x-height estimate: median height of dense horizontal-profile bands.

    The profile is the number of background-to-ink transitions per row
    (stroke crossings), not the ink pixel count: a single horizontal stroke
    (a "t" bar, a "T" top) adds many pixels but only one crossing, so it
    cannot dominate the profile. Rows with >= xheight_profile_frac of the
    maximum crossings form the core zone (ascenders/descenders are sparse
    and fall below it). Bands separated by fewer than xheight_merge_gap_px
    rows are merged, so two-line cursive crops yield one band per line. For
    uppercase words this is the cap height. Returns NaN with no band of at
    least xheight_min_band_px rows.
    """
    profile = (mask[:, 1:] & ~mask[:, :-1]).sum(axis=1) + mask[:, 0]
    if profile.max() == 0:
        return float("nan")
    dense = np.where(profile >= fcfg.xheight_profile_frac * profile.max())[0]
    bands = []
    start = prev = dense[0]
    for r in dense[1:]:
        if r - prev > fcfg.xheight_merge_gap_px:
            bands.append(prev - start + 1)
            start = r
        prev = r
    bands.append(prev - start + 1)
    bands = [b for b in bands if b >= fcfg.xheight_min_band_px]
    return float(np.median(bands)) if bands else float("nan")


def inter_component_gap_px(stats: np.ndarray) -> float:
    """Median horizontal gap from each component to its nearest right neighbour.

    Only neighbours whose vertical extent overlaps are paired, so the two
    text lines of a cursive crop are not compared with each other.
    Overlapping (touching in x) pairs are skipped. NaN with no gap.
    """
    gaps = []
    left = stats[:, cv2.CC_STAT_LEFT]
    right = left + stats[:, cv2.CC_STAT_WIDTH]
    top = stats[:, cv2.CC_STAT_TOP]
    bottom = top + stats[:, cv2.CC_STAT_HEIGHT]
    for i in range(len(stats)):
        overlap = (top < bottom[i]) & (bottom > top[i])
        candidates = left[overlap & (left >= right[i])] - right[i]
        if candidates.size:
            gaps.append(float(candidates.min()))
    return float(np.median(gaps)) if gaps else float("nan")


def crop_features(img: np.ndarray, row: pd.Series, cfg) -> dict:
    """Interpretable features for one crop.

    img is the ORIGINAL grayscale crop (uint8); it is flattened here with
    flatten_background and never modified. row is its crop_manifest.csv row
    (needs task_family; baseline_angle_deg for word/cursive). cfg is the
    master Config. Missing values are NaN (imputed later, see
    impute_crop_features).
    """
    fcfg = cfg.features
    mm = fcfg.px_to_mm
    flat = flatten_background(img, cfg.preprocessing.background_blur_ksize)
    mask, stats = ink_mask(flat, fcfg)
    nan = float("nan")

    darkness = 255.0 - flat[mask].astype(np.float64)
    widths = stroke_widths_px(mask)
    rows_ink = np.where(mask.any(axis=1))[0]
    cols_ink = np.where(mask.any(axis=0))[0]

    feats = {
        "ink_ratio": float(mask.mean()),
        "ink_darkness_mean": float(darkness.mean()) if darkness.size else nan,
        "ink_darkness_std": float(darkness.std()) if darkness.size else nan,
        "stroke_width_median_mm": (
            float(np.median(widths)) * mm if widths.size else nan
        ),
        "stroke_width_iqr_mm": (
            float(np.subtract(*np.percentile(widths, [75, 25]))) * mm
            if widths.size
            else nan
        ),
        "ink_bbox_height_mm": (
            (rows_ink[-1] - rows_ink[0] + 1) * mm if rows_ink.size else nan
        ),
        "ink_bbox_width_mm": (
            (cols_ink[-1] - cols_ink[0] + 1) * mm if cols_ink.size else nan
        ),
        "n_components": float(len(stats)),
    }
    feats.update(stroke_features(flat, mask, cfg))

    if row["task_family"] in TEXT_FAMILIES:
        baseline = pd.to_numeric(row.get("baseline_angle_deg"), errors="coerce")
        feats["baseline_angle_deg"] = float(baseline)
        feats["slant_deg"] = estimate_slant_deg(flat, mask, fcfg)
        feats["letter_height_mm"] = core_zone_height_px(mask, fcfg) * mm
        feats["inter_component_gap_mm"] = inter_component_gap_px(stats) * mm
        feats["right_margin_mm"] = (
            (mask.shape[1] - 1 - cols_ink[-1]) * mm if cols_ink.size else nan
        )
    else:
        feats.update(drawing_features(img, mask, cfg))

    return feats


def impute_crop_features(crop_df: pd.DataFrame, fcfg) -> tuple:
    """Fill missing crop features by the documented rule; return (df, log).

    Rule "task_family_median": median of the same participant's other crops
    in the same task family; if all are missing, fcfg.fallback_value. Never
    pools across participants. log has one row per (task_family, feature,
    rule) with the number of values filled. Adds an "imputed" column listing
    the filled features of each crop.
    """
    if fcfg.imputation != "task_family_median":
        raise ValueError(f"Unknown imputation rule: {fcfg.imputation!r}")

    df = crop_df.copy()
    df["imputed"] = ""
    log_rows = []
    for family in FAMILIES:
        fam_idx = df.index[df["task_family"] == family]
        for feat in features_for_family(family):
            missing = df.loc[fam_idx, feat].isna()
            if not missing.any():
                continue
            fam = df.loc[fam_idx]
            medians = fam.groupby("participant_id")[feat].transform("median")
            miss_idx = missing[missing].index
            by_median = miss_idx[medians.loc[miss_idx].notna()]
            by_fallback = miss_idx[medians.loc[miss_idx].isna()]
            df.loc[by_median, feat] = medians.loc[by_median]
            df.loc[by_fallback, feat] = fcfg.fallback_value
            df.loc[miss_idx, "imputed"] += feat + ";"
            for rule, idx in (
                ("participant_family_median", by_median),
                ("fallback_value", by_fallback),
            ):
                if len(idx):
                    log_rows.append(
                        {
                            "task_family": family,
                            "feature": feat,
                            "rule": rule,
                            "n_values": len(idx),
                            "n_participants": df.loc[idx, "participant_id"].nunique(),
                        }
                    )
    df["imputed"] = df["imputed"].str.rstrip(";")
    log = pd.DataFrame(
        log_rows,
        columns=["task_family", "feature", "rule", "n_values", "n_participants"],
    )
    return df, log


def participant_table(crop_df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: mean and std of each feature per task family.

    Columns are <family>__<feature>_mean / _std (e.g. word__slant_deg_mean).
    std is the population std (ddof=0), so it is defined for any crop count.
    Expects an already-imputed per-crop table.
    """
    parts = []
    for family in FAMILIES:
        feats = list(features_for_family(family))
        fam = crop_df[crop_df["task_family"] == family]
        if fam.empty:
            continue
        grouped = fam.groupby("participant_id")[feats]
        mean = grouped.mean().add_suffix("_mean")
        std = grouped.std(ddof=0).add_suffix("_std")
        agg = pd.concat([mean, std], axis=1)
        ordered = [f"{f}_{s}" for f in feats for s in ("mean", "std")]
        parts.append(agg[ordered].add_prefix(f"{family}__"))
    table = pd.concat(parts, axis=1).sort_index()
    table.index.name = "participant_id"
    return table.reset_index()


def _qc_passed_ids(meta_dir: Path) -> set:
    participants_path = meta_dir / "participants.csv"
    if not participants_path.is_file():
        raise FileNotFoundError(f"{participants_path} not found.")
    participants = pd.read_csv(participants_path, dtype={"participant_id": str})
    return set(
        participants.loc[participants["status"] == "qc_passed", "participant_id"]
    )


def run() -> Path:
    """Compute crop and participant feature tables for qc_passed participants.

    Writes results/features/handcrafted_crop.csv, handcrafted_participant.csv
    and handcrafted_imputation_log.csv; returns the participant table path.
    Raises if no participant is qc_passed or any NaN survives imputation.
    """
    meta_dir = config.paths.metadata_dir
    manifest_path = meta_dir / "crop_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found; run python -m src.data.crop_manifest."
        )

    passed = _qc_passed_ids(meta_dir)
    if not passed:
        raise RuntimeError(
            "No qc_passed participants in participants.csv; run "
            "python -m src.data.crop_manifest --apply-qc-status first."
        )

    manifest = pd.read_csv(manifest_path, dtype={"participant_id": str})
    manifest = manifest[manifest["participant_id"].isin(passed)]
    dropped = _load_qc_dropped(meta_dir)
    is_dropped = [
        (pid, cell) in dropped
        for pid, cell in zip(manifest["participant_id"], manifest["cell"])
    ]
    manifest = manifest[~np.array(is_dropped, dtype=bool)].reset_index(drop=True)

    records = []
    for _, row in manifest.iterrows():
        with Image.open(row["path"]) as img:
            gray = np.array(img.convert("L"))
        record = {col: row[col] for col in ID_COLUMNS}
        record.update(crop_features(gray, row, config))
        records.append(record)
    columns = list(ID_COLUMNS) + list(ALL_FEATURES)
    crop_df = pd.DataFrame(records, columns=columns)

    crop_df, imputation_log = impute_crop_features(crop_df, config.features)
    table = participant_table(crop_df)

    if table.isna().any().any():
        bad = table.columns[table.isna().any()].tolist()
        raise RuntimeError(f"NaN survived imputation in columns: {bad}")
    missing_ids = passed - set(table["participant_id"])
    if missing_ids:
        print(f"  WARNING: qc_passed without crops: {sorted(missing_ids)}")

    out_dir = config.paths.results_dir / config.features.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / "handcrafted_crop.csv"
    participant_path = out_dir / "handcrafted_participant.csv"
    log_path = out_dir / "handcrafted_imputation_log.csv"
    crop_df.to_csv(crop_path, index=False)
    table.to_csv(participant_path, index=False)
    imputation_log.to_csv(log_path, index=False)

    print(f"Crops           : {len(crop_df)}")
    print(f"Participants    : {len(table)} (qc_passed: {len(passed)})")
    print(f"Feature columns : {table.shape[1] - 1}")
    print(f"Imputed values  : {int(imputation_log['n_values'].sum())}")
    for _, r in imputation_log.iterrows():
        print(
            f"  {r['task_family']:8s} {r['feature']:24s} {r['rule']:26s} "
            f"{r['n_values']} values / {r['n_participants']} participants"
        )
    print(f"Written         : {crop_path}")
    print(f"                  {participant_path}")
    print(f"                  {log_path}")
    return participant_path


def main() -> int:
    try:
        run()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
