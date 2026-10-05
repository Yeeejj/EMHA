"""
Stroke-level and drawing-specific handcrafted features — Stage D.

All measurements use the ink mask and the background-flattened crop from
src.features.handcrafted; nothing here modifies an image. Stroke features
are computed on the scikit-image skeleton of the ink mask, split into simple
paths at junctions:

    stroke_width_cv          width variation along paths
    darkness_cv              ink darkness variation along paths (pressure proxy)
    tremor_index             high-frequency curvature energy along paths
    mean_segment_length_mm   mean junction-to-junction/end path length
    endpoints_per_cm         stroke ends per cm of skeleton
    junctions_per_cm         stroke crossings/branchings per cm of skeleton

Drawing features (D1-D4) describe placement, size, detail, use of space,
and erasure-like smudges.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from src.preprocessing.pipeline import flatten_background

# 8-neighbourhood in clockwise order (P2..P9), used for the crossing number.
_RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
# Same neighbours, orthogonal first: preferred continuation when tracing.
_STEP_ORDER = ((-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1))

STROKE_FEATURES = (
    "stroke_width_cv",
    "darkness_cv",
    "tremor_index",
    "mean_segment_length_mm",
    "endpoints_per_cm",
    "junctions_per_cm",
)
DRAWING_FEATURES = (
    "ink_bbox_area_fraction",
    "centroid_x_rel",
    "centroid_y_rel",
    "lower_half_ink_fraction",
    "figure_height_mm",
    "detail_count",
    "empty_space_ratio",
    "erasure_score",
)


def crossing_number(skel: np.ndarray) -> np.ndarray:
    """Rutovitz crossing number of each skeleton pixel (0 off-skeleton).

    CN = half the number of 0/1 changes around the 8-neighbourhood ring:
    1 = endpoint, 2 = path interior, >= 3 = junction. Unlike a plain
    neighbour count, it is not fooled by staircase corners.
    """
    padded = np.pad(skel.astype(np.int8), 1)
    height, width = skel.shape
    ring = np.stack(
        [padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width] for dr, dc in _RING]
    )
    cn = np.abs(ring - np.roll(ring, -1, axis=0)).sum(axis=0) // 2
    return np.where(skel, cn, 0)


def _trace_component(pixels: set) -> list:
    """Order the pixels of a junction-free skeleton piece into walks."""
    remaining = set(pixels)
    walks = []

    def neighbours(p):
        r, c = p
        return [
            (r + dr, c + dc) for dr, dc in _STEP_ORDER if (r + dr, c + dc) in remaining
        ]

    while remaining:
        # Start at an end (fewest neighbours) so open paths are walked whole.
        start = min(remaining, key=lambda p: (len(neighbours(p)), p))
        remaining.remove(start)
        walk = [start]
        current = start
        while True:
            nxt = neighbours(current)
            if not nxt:
                break
            current = nxt[0]
            remaining.remove(current)
            walk.append(current)
        walks.append(walk)
    return walks


def _paths_from_skeleton(skel: np.ndarray, cn: np.ndarray, min_px: int) -> list:
    # Cut junction pixels AND their 8-neighbours: removing the junction pixel
    # alone leaves its arms touching diagonally, i.e. still one piece.
    junction_zone = ndimage.binary_dilation(cn >= 3, structure=np.ones((3, 3)))
    pieces = skel & ~junction_zone
    labels, n = ndimage.label(pieces, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return []
    rows, cols = np.nonzero(labels)
    order = np.argsort(labels[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]
    bounds = np.searchsorted(labels[rows, cols], np.arange(1, n + 2))
    paths = []
    for k in range(n):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < min_px:
            continue
        pixels = set(zip(rows[lo:hi].tolist(), cols[lo:hi].tolist()))
        for walk in _trace_component(pixels):
            if len(walk) >= min_px:
                paths.append(np.array(walk, dtype=np.int64))
    return paths


def skeleton_paths(mask: np.ndarray, min_px: int = 1) -> list:
    """Split the skeleton of mask into ordered simple paths.

    The mask is skeletonized (skimage.morphology.skeletonize), junction
    pixels (crossing number >= 3) and their 8-neighbours are removed, and
    each remaining
    8-connected piece is walked end to end. Returns a list of (N, 2) int
    arrays of (row, col) in walk order; paths shorter than min_px pixels
    are dropped.
    """
    skel = skeletonize(mask.astype(bool))
    return _paths_from_skeleton(skel, crossing_number(skel), min_px)


def path_length_px(path: np.ndarray) -> float:
    """Arc length of an ordered pixel path (steps of 1 or sqrt(2))."""
    if len(path) < 2:
        return 0.0
    return float(np.hypot(*np.diff(path, axis=0).T).sum())


def _resample_unit(path: np.ndarray) -> np.ndarray:
    """(x, y) points at unit arc-length spacing along an ordered path."""
    xy = path[:, ::-1].astype(np.float64)
    steps = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    t = np.arange(0.0, s[-1] + 1e-9, 1.0)
    return np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])


def _curvature(xy: np.ndarray, sigma: float) -> np.ndarray:
    x = ndimage.gaussian_filter1d(xy[:, 0], sigma, mode="nearest")
    y = ndimage.gaussian_filter1d(xy[:, 1], sigma, mode="nearest")
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom[denom == 0] = np.inf
    return (dx * ddy - dy * ddx) / denom


def tremor_energy(path: np.ndarray, fcfg) -> tuple:
    """(mean squared high-frequency curvature in px^-2, n samples) for a path.

    Multi-scale curvature in the sense of curvature scale space (Mokhtarian
    & Mackworth, "A theory of multiscale, curvature-based shape
    representation for planar curves", IEEE TPAMI 14(8):789-805, 1992): the
    path is resampled to unit arc length and curvature is computed after
    Gaussian smoothing at a fine scale (tremor_fine_sigma_px, removes pixel
    quantization) and a coarse scale (tremor_window_px, the smooth intended
    trajectory). Their difference is the high-frequency curvature that a
    tremulous, wavering line adds; its energy is averaged over the path
    core (tremor_window_px trimmed at each end to avoid edge effects).
    Returns (nan, 0) for paths shorter than 4 * tremor_window_px.
    """
    window = fcfg.tremor_window_px
    xy = _resample_unit(path)
    if len(xy) < 4 * window:
        return float("nan"), 0
    high = _curvature(xy, fcfg.tremor_fine_sigma_px) - _curvature(xy, window)
    core = high[window:-window]
    return float(np.mean(core * core)), len(core)


def _weighted_mean(values: list, weights: list) -> float:
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(vals) & (wts > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(vals[ok], weights=wts[ok]))


def stroke_features(img: np.ndarray, mask: np.ndarray, cfg) -> dict:
    """Stroke-level features on the skeleton of mask.

    img is the background-flattened crop (uint8, paper ~255); mask is its
    measurement ink mask; cfg is the master Config. Width (2 * EDT - 1) and
    darkness (255 - img) are sampled at skeleton pixels; each path's ends
    are trimmed by its maximum half-width so stroke tips and junction blobs
    do not count as width/pressure variation. CVs are per path, then
    length-weighted across paths. tremor_index is the square root of the
    length-weighted tremor_energy, in rad/mm. Per-cm densities use the
    skeleton pixel count as ink length. Missing values are NaN.
    """
    fcfg = cfg.features
    mm = fcfg.px_to_mm
    nan = float("nan")
    skel = skeletonize(mask.astype(bool))
    n_skel = int(skel.sum())
    if n_skel == 0:
        return {name: nan for name in STROKE_FEATURES}

    cn = crossing_number(skel)
    paths = _paths_from_skeleton(skel, cn, fcfg.skeleton_min_path_px)
    dist = ndimage.distance_transform_edt(mask)
    darkness = 255.0 - img.astype(np.float64)

    width_cv, dark_cv, cv_weights = [], [], []
    energies, energy_weights = [], []
    for path in paths:
        r, c = path[:, 0], path[:, 1]
        trim = int(np.ceil(dist[r, c].max()))
        core = slice(trim, len(path) - trim)
        if len(path) - 2 * trim >= fcfg.skeleton_min_path_px:
            widths = 2.0 * dist[r[core], c[core]] - 1.0
            dark = darkness[r[core], c[core]]
            width_cv.append(widths.std() / widths.mean())
            dark_cv.append(dark.std() / dark.mean() if dark.mean() > 0 else nan)
            cv_weights.append(len(widths))
        energy, n = tremor_energy(path, fcfg)
        energies.append(energy)
        energy_weights.append(n)

    energy = _weighted_mean(energies, energy_weights)
    lengths = [path_length_px(p) for p in paths]
    ink_cm = n_skel * mm / 10.0
    _, n_junctions = ndimage.label(cn >= 3, structure=np.ones((3, 3), dtype=bool))

    return {
        "stroke_width_cv": _weighted_mean(width_cv, cv_weights),
        "darkness_cv": _weighted_mean(dark_cv, cv_weights),
        "tremor_index": float(np.sqrt(energy)) / mm if np.isfinite(energy) else nan,
        "mean_segment_length_mm": float(np.mean(lengths)) * mm if lengths else nan,
        "endpoints_per_cm": float((cn == 1).sum()) / ink_cm,
        "junctions_per_cm": float(n_junctions) / ink_cm,
    }


def erasure_mask(img: np.ndarray, mask: np.ndarray, fcfg) -> tuple:
    """(smudge, away_from_ink) boolean masks.

    img is the ORIGINAL grayscale crop. It is flattened here with
    flatten_background at erasure_background_ksize (much larger than the
    model-path kernel): the standard kernel treats a smudge of its own size
    as background and flattens it back to paper white.
    smudge = areal faint-gray regions away from ink (erasure candidates);
    away_from_ink = pixels farther than erasure_ink_margin_px from ink and
    outside the border_band_px edge band (box rules and scan-edge shadow).
    """
    paper_flat = flatten_background(img, fcfg.erasure_background_ksize)
    lo, hi = fcfg.faint_gray_range
    margin = 2 * fcfg.erasure_ink_margin_px + 1
    near_ink = cv2.dilate(mask.astype(np.uint8), np.ones((margin, margin), np.uint8))
    away = near_ink == 0
    band = fcfg.border_band_px
    away[:band, :] = False
    away[-band:, :] = False
    away[:, :band] = False
    away[:, -band:] = False
    faint = ((paper_flat >= lo) & (paper_flat < hi) & away).astype(np.uint8)
    kernel = np.ones((fcfg.erasure_open_px, fcfg.erasure_open_px), np.uint8)
    return cv2.morphologyEx(faint, cv2.MORPH_OPEN, kernel) > 0, away


def drawing_features(img: np.ndarray, mask: np.ndarray, cfg) -> dict:
    """Drawing-specific features (D1-D4).

    img is the ORIGINAL grayscale crop (only erasure_score reads it, see
    erasure_mask), mask its measurement ink mask (border rules and specks
    already removed), cfg the master Config.

    ink_bbox_area_fraction  ink bounding-box area / crop area
    centroid_x_rel/_y_rel   ink centroid as a fraction of crop width/height
    lower_half_ink_fraction share of ink pixels in the lower half of the crop
    figure_height_mm        bbox height of the largest component (main figure)
    detail_count            components whose centroid lies in the figure box
                            (the figure itself included)
    empty_space_ratio       share of an empty_space_grid^2 tile grid with no ink
    erasure_score           share of non-ink area that is faint-gray smudge
                            (see erasure_mask and FeaturesConfig)
    """
    fcfg = cfg.features
    nan = float("nan")
    height, width = mask.shape
    smudge, away_from_ink = erasure_mask(img, mask, fcfg)
    erasure = float(smudge.sum()) / max(1, int(away_from_ink.sum()))

    grid = fcfg.empty_space_grid
    tiles = [
        t.any()
        for band in np.array_split(mask, grid, axis=0)
        for t in np.array_split(band, grid, axis=1)
    ]
    empty_space = 1.0 - float(np.mean(tiles))

    n_ink = int(mask.sum())
    if n_ink == 0:
        feats = {name: nan for name in DRAWING_FEATURES}
        feats.update(
            {"detail_count": 0.0, "empty_space_ratio": 1.0, "erasure_score": erasure}
        )
        return feats

    rows, cols = np.nonzero(mask)
    bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    stats, centroids = stats[1:], centroids[1:]
    fig = int(np.argmax(stats[:, cv2.CC_STAT_AREA]))
    left, top, w, h = (
        stats[fig, cv2.CC_STAT_LEFT],
        stats[fig, cv2.CC_STAT_TOP],
        stats[fig, cv2.CC_STAT_WIDTH],
        stats[fig, cv2.CC_STAT_HEIGHT],
    )
    cx, cy = centroids[:, 0], centroids[:, 1]
    inside = (cx >= left) & (cx < left + w) & (cy >= top) & (cy < top + h)

    return {
        "ink_bbox_area_fraction": float(bbox_area) / (height * width),
        "centroid_x_rel": float(cols.mean() + 0.5) / width,
        "centroid_y_rel": float(rows.mean() + 0.5) / height,
        "lower_half_ink_fraction": float(mask[height // 2 :].sum()) / n_ink,
        "figure_height_mm": float(h) * fcfg.px_to_mm,
        "detail_count": float(inside.sum()),
        "empty_space_ratio": empty_space,
        "erasure_score": erasure,
    }
