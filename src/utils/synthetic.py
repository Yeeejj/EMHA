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

signal=False: features and embeddings are pure noise, independent of the
label (any model should score near chance). signal=True: SAD participants'
first `n_signal` handcrafted columns and first `n_signal` embedding
dimensions are shifted by `shift` standard deviations, so a correct model
scores clearly above chance. No images are written.
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


def make_synthetic_root(
    root,
    n_participants: int = 200,
    signal: bool = False,
    seed: int = 0,
    n_signal: int = 3,
    shift: float = 1.0,
    handwriting: bool = False,
    cfg=None,
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
    return root
