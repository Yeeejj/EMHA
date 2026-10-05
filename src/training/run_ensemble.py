"""
The fixed ensemble named in EVALUATION_PROTOCOL.md (primary claim) — Stage F.

Components and weights are fixed by the protocol; there is no weights
argument and no way to add or drop a component:

    lr_handcrafted  / all      results/baselines/<analysis>/predictions.csv
    lr_embeddings   / concat   results/baselines/<analysis>/predictions.csv
    cnn_hmm_fused   / fused    results/hybrid/<analysis>/predictions.csv

combine() inner-joins the three components on (analysis, fold,
participant_id) and averages prob_sad with equal weights; pred is SAD when
prob_sad >= AggregateConfig.threshold (0.5). It never averages fewer than
three: any test-fold participant missing from a component raises.
Middle-band rows (in_middle_band) are kept only for participants every
component predicted; if a component has no middle-band rows at all, the
ensemble has none. label and in_middle_band must agree across components.
Exploratory models (e.g. lr_handwriting) are never included.

Output: results/ensemble/<analysis>/predictions.csv with model "ensemble",
feature_set "all", in the shared prediction schema.

    python -m src.training.run_ensemble [--analysis primary|full]
        [--stage smoke|pilot|full]

Each run appends point metrics to results/RUN_LOG.csv (src/utils/run_log.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.run_baselines import EXPLORATORY_MODELS, PRED_COLUMNS
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic
from src.utils.run_log import STAGES, command_line, log_predictions, resolve_stage

COMPONENTS = (
    ("lr_handcrafted", "all", "baselines"),
    ("lr_embeddings", "concat", "baselines"),
    ("cnn_hmm_fused", "fused", "hybrid"),
)
MODEL, FEATURE_SET = "ensemble", "all"
KEYS = ["analysis", "fold", "participant_id"]


def _component_key(frame: pd.DataFrame) -> tuple:
    pairs = set(zip(frame["model"], frame["feature_set"]))
    if len(pairs) != 1:
        raise ValueError(f"each frame must hold one model/feature_set, got {pairs}")
    return pairs.pop()


def combine(frames: list) -> pd.DataFrame:
    """Equal-weight ensemble of exactly the protocol's three components.

    frames: one prediction DataFrame per component (shared schema), in any
    order. Raises if the components are not exactly COMPONENTS, if any
    test-fold participant is missing from a component, or if labels /
    middle-band flags disagree.
    """
    expected = {(m, f) for m, f, _ in COMPONENTS}
    by_key = {}
    for frame in frames:
        key = _component_key(frame)
        if key[0] in EXPLORATORY_MODELS:
            raise ValueError(f"exploratory model {key[0]} is never in the ensemble")
        if key in by_key:
            raise ValueError(f"component {key} given twice")
        by_key[key] = frame.assign(participant_id=frame["participant_id"].astype(str))
    if set(by_key) != expected:
        raise ValueError(
            f"ensemble components are fixed: need {sorted(expected)}, "
            f"got {sorted(by_key)}"
        )

    parts = []
    for i, (model, fs, _) in enumerate(COMPONENTS):
        f = by_key[(model, fs)]
        if f.duplicated(KEYS).any():
            raise ValueError(f"{model}/{fs} has duplicate participant rows")
        parts.append(
            f[KEYS + ["label", "in_middle_band", "prob_sad"]].rename(
                columns={
                    "label": f"label_{i}",
                    "in_middle_band": f"middle_{i}",
                    "prob_sad": f"prob_{i}",
                }
            )
        )

    # Test-fold rows: every component must have every participant.
    tests = [p[~p[f"middle_{i}"].astype(bool)] for i, p in enumerate(parts)]
    union = set().union(*(set(map(tuple, t[KEYS].to_numpy())) for t in tests))
    for (model, fs, _), t in zip(COMPONENTS, tests):
        missing = union - set(map(tuple, t[KEYS].to_numpy()))
        if missing:
            raise ValueError(
                f"{model}/{fs} misses {len(missing)} test-fold participant(s), "
                f"e.g. {sorted(missing)[:3]}; never averaging fewer than three"
            )

    joined = parts[0]
    for p in parts[1:]:
        joined = joined.merge(p, on=KEYS, how="inner")
    n = len(COMPONENTS)
    for col in ("label", "middle"):
        values = joined[[f"{col}_{i}" for i in range(n)]].astype(str)
        if (values.nunique(axis=1) > 1).any():
            raise ValueError(f"components disagree on {col} for some participants")

    prob = joined[[f"prob_{i}" for i in range(n)]].mean(axis=1)
    out = pd.DataFrame(
        {
            "analysis": joined["analysis"],
            "model": MODEL,
            "feature_set": FEATURE_SET,
            "fold": joined["fold"],
            "participant_id": joined["participant_id"],
            "label": joined["label_0"],
            "prob_sad": prob,
            "pred": np.where(prob >= config.aggregate.threshold, "SAD", "HAPPY"),
            "in_middle_band": joined["middle_0"].astype(bool),
        }
    )
    return (
        out[list(PRED_COLUMNS)]
        .sort_values(["fold", "participant_id"])
        .reset_index(drop=True)
    )


def _component_frame(analysis: str, model: str, fs: str, subdir: str) -> pd.DataFrame:
    path = config.paths.results_dir / subdir / analysis / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the {subdir} runner first")
    preds = pd.read_csv(path, dtype={"participant_id": str})
    rows = preds[(preds["model"] == model) & (preds["feature_set"] == fs)]
    if rows.empty:
        raise ValueError(f"{path} has no {model}/{fs} rows")
    return rows


def run(analysis: str = "primary", stage: str | None = None) -> Path:
    """Build the ensemble from the component runs; return predictions.csv path."""
    require_frozen_or_synthetic(config, "run_ensemble")
    stage = resolve_stage(stage, None)
    frames = [_component_frame(analysis, m, f, d) for m, f, d in COMPONENTS]
    ensemble = combine(frames)
    out_dir = config.paths.results_dir / "ensemble" / analysis
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.csv"
    ensemble.to_csv(path, index=False)
    n_mid = int(ensemble["in_middle_band"].sum())
    print(
        f"Ensemble: {len(ensemble) - n_mid} test + {n_mid} middle-band rows -> {path}"
    )
    command = command_line("src.training.run_ensemble", analysis=analysis, stage=stage)
    log_predictions(stage, command, analysis, None, ensemble)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed equal-weight ensemble.")
    parser.add_argument("--analysis", choices=("primary", "full"), default="primary")
    parser.add_argument("--stage", choices=STAGES, default=None)
    args = parser.parse_args()
    try:
        run(args.analysis, args.stage)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
