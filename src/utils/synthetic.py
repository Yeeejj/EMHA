"""
Synthetic data roots for tests and smoke runs — never real data.

make_synthetic_root(root, ...) writes a feature-level fixture laid out like
the real data and output roots (both point at `root`):

    root/.synthetic_root                         marker (see protocol_guard)
    root/metadata/labels.csv                     participant_id, total_score,
                                                 label, boundary_distance,
                                                 in_middle_band,
                                                 in_primary_analysis
    root/metadata/participants.csv               every participant qc_passed
    root/metadata/folds.csv                      src.training.splits folds
    root/results/features/handcrafted_participant.csv
    root/results/embeddings/resnet18_<family>.npz    per-crop, 512-d
    root/results/embeddings/handwriting_<family>.npz (optional, exploratory)
    root/processed/<pid>/<cell>.png              (optional, images=True)
    root/metadata/processed_manifest.csv         (optional, images=True)
    root/metadata/questionnaire_export.csv       (optional, export=True)
    root/raw-3Page, root/raw-4Page               (optional, raw=True)

signal=False: features and embeddings are pure noise, independent of the
label (any model should score near chance). signal=True: SAD participants'
first `n_signal` handcrafted columns and first `n_signal` embedding
dimensions are shifted by `shift` standard deviations, so a correct model
scores clearly above chance. images=True also writes processed-style crops
(inverted canvases: dark paper, bright strokes) at
PreprocessingConfig.canvas_size per family -- tests shrink canvas_size to
keep CNN runs fast; with signal, SAD crops get thicker, brighter strokes.

export=True writes a tabulation export with the same participants, scores
and labels as labels.csv (id, item, score and label columns named by
LabelingConfig/ReportConfig), so src.data.labeler reproduces labels.csv
exactly; items are noisy reflections of the score, so alpha is positive.
raw=True writes page scans (SCAN_SIZE_PX, IngestConfig.expected_dpi) and all
24 crops per participant under the real file names, at
CropConfig.expected_size_px, as dark strokes on white paper (with signal,
SAD strokes are thicker and darker), for ingest -> crop manifest ->
preprocessing -> handcrafted features end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.crop_manifest import CURSIVE_CELLS, DRAWING_CELLS, WORD_CELLS
from src.features.handcrafted import FAMILIES, features_for_family
from src.training.splits import make_outer_folds
from src.utils.protocol_guard import SYNTHETIC_MARKER

CELLS = {"drawing": DRAWING_CELLS, "word": WORD_CELLS, "cursive": CURSIVE_CELLS}
EMBED_DIM = 512
HANDWRITING_DIM = 32
# Page-shaped placeholder scans: ingest checks DPI and sizes relative to the
# median page, never an absolute page size.
SCAN_SIZE_PX = (170, 220)
PAPER = 255
N_STROKES = 6
# (ink grey level, stroke thickness px) for raw crops
RAW_INK = {"SAD": (40, 3), "HAPPY": (90, 1)}
ITEM_NOISE_SD = 0.7


def _labels(n: int, rng: np.random.Generator, cfg) -> pd.DataFrame:
    cutoff = cfg.labeling.cutoff
    scores = np.round(rng.normal(cutoff, 15.0, size=n)) + 0.5  # never == cutoff
    labels = np.where(scores > cutoff, "HAPPY", "SAD")
    distance = np.abs(scores - cutoff)
    n_mid = int(round(cfg.labeling.middle_band_fraction * n))
    middle = np.zeros(n, dtype=bool)
    middle[np.argsort(distance, kind="stable")[:n_mid]] = True
    return pd.DataFrame(
        {
            "participant_id": [f"{i + 1:03d}" for i in range(n)],
            "total_score": scores,
            "label": labels,
            "boundary_distance": distance,
            "in_middle_band": middle,
            "in_primary_analysis": ~middle,
        }
    )


def _handcrafted(labels, rng, signal, n_signal, shift) -> pd.DataFrame:
    columns = [
        f"{family}__{feat}_{stat}"
        for family in FAMILIES
        for feat in features_for_family(family)
        for stat in ("mean", "std")
    ]
    values = rng.normal(0.0, 1.0, size=(len(labels), len(columns)))
    if signal:
        sad = (labels["label"] == "SAD").to_numpy()
        # spread the signal over families: first n_signal columns of each
        for family in FAMILIES:
            idx = [i for i, c in enumerate(columns) if c.startswith(f"{family}__")]
            values[np.ix_(sad, idx[:n_signal])] += shift
    table = pd.DataFrame(values, columns=columns)
    table.insert(0, "participant_id", labels["participant_id"].to_numpy())
    return table


def _embeddings(labels, rng, family, dim, signal, n_signal, shift) -> dict:
    cells = CELLS[family]
    pids = np.repeat(labels["participant_id"].to_numpy(), len(cells))
    cell = np.tile(np.array(cells), len(labels))
    emb = rng.normal(0.0, 1.0, size=(len(pids), dim)).astype(np.float32)
    if signal:
        sad = np.repeat((labels["label"] == "SAD").to_numpy(), len(cells))
        emb[np.ix_(sad, np.arange(n_signal))] += shift
    return {
        "embeddings": emb,
        "participant_id": pids.astype(str),
        "cell": cell.astype(str),
        "cache_key": np.array("synthetic"),
    }


def _images(labels, rng, root: Path, signal: bool, cfg) -> pd.DataFrame:
    import cv2

    rows = []
    for pid, label in zip(labels["participant_id"], labels["label"]):
        sad = signal and label == "SAD"
        for family, cells in CELLS.items():
            w, h = cfg.preprocessing.canvas_size[family]
            for cell in cells:
                img = np.zeros((h, w), dtype=np.uint8)
                for _ in range(4):
                    p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                    p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                    cv2.line(img, p1, p2, 255 if sad else 180, 3 if sad else 1)
                path = root / "processed" / pid / f"{cell}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), img)
                rows.append(
                    {
                        "participant_id": pid,
                        "cell": cell,
                        "task_family": family,
                        "processed_path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def _export(labels, rng, cfg) -> pd.DataFrame:
    from src.analysis.questionnaire_report import LIKERT_MAX, LIKERT_MIN

    items = list(cfg.report.item_columns)
    lo, hi = len(items) * LIKERT_MIN, len(items) * LIKERT_MAX
    level = LIKERT_MIN + (LIKERT_MAX - LIKERT_MIN) * (
        (labels["total_score"].to_numpy() - lo) / (hi - lo)
    )
    noise = rng.normal(0.0, ITEM_NOISE_SD, size=(len(labels), len(items)))
    scored = np.clip(np.round(level[:, None] + noise), LIKERT_MIN, LIKERT_MAX)
    table = pd.DataFrame(scored.astype(int), columns=items)
    for col in cfg.report.reverse_items:
        table[col] = (LIKERT_MIN + LIKERT_MAX) - table[col]
    lbl = cfg.labeling
    table.insert(0, lbl.id_column, labels["participant_id"].to_numpy())
    table[lbl.score_column] = labels["total_score"].to_numpy()
    table[lbl.label_column] = labels["label"].to_numpy()
    return table


def _raw(labels, rng, root: Path, signal: bool, cfg) -> None:
    from PIL import Image, ImageDraw

    raw3, raw4 = root / "raw-3Page", root / "raw-4Page"
    dpi = (cfg.ingest.expected_dpi, cfg.ingest.expected_dpi)
    for pid, label in zip(labels["participant_id"], labels["label"]):
        ink, width = RAW_INK["SAD" if signal and label == "SAD" else "HAPPY"]
        for folder, name in (
            (raw3, f"EMHA-P3_DrawingExercise_{pid}.png"),
            (raw4, f"EMHA-P4_WritingExercise_{pid}.png"),
        ):
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("L", SCAN_SIZE_PX, PAPER).save(folder / name, dpi=dpi)
        for family, cells in CELLS.items():
            folder = raw3 if family == "drawing" else raw4
            prefix = (
                "P3_DrawingExercise" if family == "drawing" else "P4_WritingExercise"
            )
            for cell in cells:
                w, h = cfg.crop.expected_size_px[cell]
                img = Image.new("L", (w, h), PAPER)
                draw = ImageDraw.Draw(img)
                for _ in range(N_STROKES):
                    xy = [
                        (int(rng.integers(0, w)), int(rng.integers(0, h)))
                        for _ in range(2)
                    ]
                    draw.line(xy, fill=ink, width=width)
                path = folder / cell / f"EMHA-{prefix}_{pid}_{cell}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path)


def make_synthetic_root(
    root,
    n_participants: int = 200,
    signal: bool = False,
    seed: int = 0,
    n_signal: int = 3,
    shift: float = 1.0,
    handwriting: bool = False,
    images: bool = False,
    cfg=None,
    export: bool = False,
    raw: bool = False,
) -> Path:
    """Write a synthetic data root at `root` and return it."""
    from src.utils.config import config

    cfg = cfg or config
    root = Path(root)
    rng = np.random.default_rng(seed)
    meta = root / "metadata"
    features = root / "results" / "features"
    emb_dir = root / "results" / "embeddings"
    for d in (meta, features, emb_dir):
        d.mkdir(parents=True, exist_ok=True)
    (root / SYNTHETIC_MARKER).write_text("synthetic fixture, not real data\n")

    labels = _labels(n_participants, rng, cfg)
    labels.to_csv(meta / "labels.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": labels["participant_id"],
            "has_p3_scan": True,
            "has_p4_scan": True,
            "status": "qc_passed",
        }
    ).to_csv(meta / "participants.csv", index=False)
    folds = make_outer_folds(labels, cfg.cv.n_splits, cfg.training.seed)
    folds.to_csv(meta / cfg.cv.folds_filename, index=False)

    _handcrafted(labels, rng, signal, n_signal, shift).to_csv(
        features / "handcrafted_participant.csv", index=False
    )
    for family in FAMILIES:
        np.savez_compressed(
            emb_dir / f"resnet18_{family}.npz",
            **_embeddings(labels, rng, family, EMBED_DIM, signal, n_signal, shift),
        )
    if handwriting:
        for family in ("word", "cursive"):
            np.savez_compressed(
                emb_dir / f"handwriting_{family}.npz",
                **_embeddings(
                    labels, rng, family, HANDWRITING_DIM, signal, n_signal, shift
                ),
                exploratory=np.array(True),
            )
    if images:
        _images(labels, rng, root, signal, cfg).to_csv(
            meta / "processed_manifest.csv", index=False
        )
    if export:
        _export(labels, rng, cfg).to_csv(meta / "questionnaire_export.csv", index=False)
    if raw:
        _raw(labels, rng, root, signal, cfg)
    return root
