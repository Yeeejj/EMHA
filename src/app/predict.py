"""
Defense demo: HAPPY or SAD for one participant's crops — Stage F.

Only the fold models for which the participant was in outer validation are
used, so the demo never shows a participant to a model trained on them:

  1. outer fold k = the participant's outer_fold in folds.csv; a --fold that
     differs, or a code outside the analysis set and its middle band, is
     refused (DemoRefusal).
  2. CNN-HMM (word + cursive): the fold-k run_cnn checkpoints must list the
     code in their recorded test_ids / middle_ids and never in
     inner_train_ids / inner_val_ids (run_hybrid fits the fold-k HMMs on
     that same inner split). Crops are rebuilt from the raw crops with
     src.preprocessing.pipeline.preprocess_crop (D1), in memory and
     read-only; QC-dropped crops are skipped as in training.
  3. LR components: run_baselines saves no models, so fit_lr is refit on
     fold k's outer-train participants with the same seed (seed + k) and
     the stored feature tables; assert_no_leakage(train, [code]).
  4. Every live component P(SAD) must equal the stored out-of-fold
     prediction (predictions.csv) within DemoConfig.match_atol, else refuse.
     The ensemble is the equal-weight mean of its three components
     (src.training.run_ensemble.COMPONENTS).

Models: ensemble (default), cnn_hmm (= cnn_hmm_fused), lr_handcrafted,
lr_embeddings. Grad-CAM overlays for DemoConfig.n_gradcam crops are saved
when src.analysis.gradcam (H4) exists; it must provide
save_overlay(cnn, image, out_path) -> Path for a (1, H, W) eval tensor.

Output: printed summary and results/demo/<code>/prediction.json.
Refuses to run on real data before the protocol-frozen tag.

    python -m src.app.predict --code 017 [--model ensemble] [--fold 2]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.dataloader import labels_csv_path, load_tables
from src.data.transforms import get_eval_transform
from src.models.hmm import HMMClassifier
from src.models.hybrid import (
    HybridCNNHMM,
    cnn_checkpoint_path,
    hmm_path,
    load_cnn,
)
from src.preprocessing.pipeline import preprocess_crop
from src.training.run_baselines import _prob_sad, feature_sets, fit_lr
from src.training.run_cnn import run_name
from src.training.run_ensemble import COMPONENTS
from src.training.splits import (
    analysis_ids,
    assert_no_leakage,
    fold_ids,
    load_folds,
    middle_band_ids,
    restrict_to_subset,
)
from src.utils.config import config
from src.utils.protocol_guard import require_frozen_or_synthetic

MODELS = ("ensemble", "cnn_hmm", "lr_handcrafted", "lr_embeddings")
LR_SETS = {"lr_handcrafted": ("all", False), "lr_embeddings": ("concat", True)}
HYBRID_COMPONENT = ("cnn_hmm_fused", "fused")
GRADCAM_MODULE = "src.analysis.gradcam"


class DemoRefusal(RuntimeError):
    """The participant cannot be shown without a model that trained on them."""


# ── fold and participant ─────────────────────────────────────────────────────


def _run_dir_name(cfg) -> str:
    d = cfg.demo
    return d.analysis if d.subset is None else f"{d.analysis}_{d.subset}"


def outer_fold(code: str, fold: int | None, cfg=config) -> dict:
    """{fold, train_ids, in_middle_band, self_report} for one participant.

    Refuses a code outside the analysis set (and, for the primary analysis,
    its middle band), or a fold other than the participant's outer fold.
    """
    meta = cfg.paths.metadata_dir
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    participants = pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
    folds = load_folds(cfg).set_index("participant_id")

    ids = restrict_to_subset(
        analysis_ids(labels, participants, cfg.demo.analysis), cfg.demo.subset
    )
    middle = []
    if cfg.demo.analysis == "primary":
        middle = middle_band_ids(labels, participants, cfg.demo.subset)
    if code not in set(ids) | set(middle):
        raise DemoRefusal(
            f"participant {code} is not in the {_run_dir_name(cfg)} analysis set "
            "or its middle band (unknown, unlabeled, or not qc_passed)"
        )
    k = int(folds.loc[code, "outer_fold"])
    if fold is not None and fold != k:
        raise DemoRefusal(
            f"participant {code} is in outer fold {k}; the fold-{fold} models "
            "trained on them, so they cannot be shown"
        )
    train, _ = fold_ids(folds.reset_index(), k, ids)
    assert_no_leakage(train, [code])
    label_col = cfg.labeling.label_column
    self_report = labels.set_index("participant_id").loc[code, label_col]
    return {
        "fold": k,
        "train_ids": train,
        "in_middle_band": code in set(middle),
        "self_report": str(self_report),
    }


def check_held_out(state: dict, code: str, fold: int, path: Path) -> None:
    """Refuse unless a checkpoint held `code` out of all fitting."""
    if any(state.get(k) is None for k in ("test_ids", "inner_train_ids")):
        raise DemoRefusal(f"{path} records no split IDs; re-run run_cnn")
    if int(state.get("fold", -1)) != fold:
        raise DemoRefusal(f"{path} is for fold {state.get('fold')}, not {fold}")
    fitted = {str(p) for p in state["inner_train_ids"] + state["inner_val_ids"]}
    held = {str(p) for p in state["test_ids"] + list(state.get("middle_ids") or [])}
    if code in fitted:
        raise DemoRefusal(f"{path} was trained on participant {code}")
    if code not in held:
        raise DemoRefusal(
            f"participant {code} is not in the outer validation of {path} "
            "(middle-band participants need run_cnn --predict-middle)"
        )


# ── crops ────────────────────────────────────────────────────────────────────


def load_crops(code: str, cfg=config) -> dict:
    """{family: (cells, uint8 array (N, H, W))} of processed crops.

    With a raw `path` column (real data) every crop is rebuilt from the raw
    crop by preprocess_crop, read-only and never written; manifests without
    it (synthetic fixture) supply the processed images directly.
    """
    manifest, _, dropped = load_tables(cfg)
    rows = manifest[manifest["participant_id"].astype(str) == code]
    rows = rows[[(code, c) not in dropped for c in rows["cell"]]]
    if rows.empty:
        raise DemoRefusal(f"no crops for participant {code}")
    from_raw = "path" in manifest.columns
    out = {}
    for family, part in rows.sort_values("cell").groupby("task_family"):
        images = []
        for _, row in part.iterrows():
            with Image.open(row["path"] if from_raw else row["processed_path"]) as im:
                gray = np.array(im.convert("L"))
            if from_raw:
                gray = preprocess_crop(gray, family, cfg.preprocessing)
            images.append(gray)
        out[family] = (list(part["cell"]), np.stack(images))
    return out


# ── components ───────────────────────────────────────────────────────────────


def lr_component(model: str, code: str, fold: int, train: list, cfg=config):
    """Refit fold `fold`'s LR (same data, seed) and return P(SAD) for `code`."""
    fs, use_pca = LR_SETS[model]
    X = next(X for m, f, X, _ in feature_sets(cfg) if (m, f) == (model, fs))
    if code not in X.index:
        raise DemoRefusal(f"{model}/{fs}: no features for participant {code}")
    assert_no_leakage(train, [code])
    labels = pd.read_csv(labels_csv_path(cfg), dtype={"participant_id": str})
    label_of = labels.set_index("participant_id")[cfg.labeling.label_column]
    y_train = label_of.loc[train].map(cfg.data.label_to_index).to_numpy()
    seed = cfg.training.seed + fold
    pipe = fit_lr(X.loc[train], y_train, cfg, pca=use_pca, seed=seed)
    return float(_prob_sad(pipe, X.loc[[code]])[0])


def load_hybrid(code: str, fold: int, cfg=config) -> HybridCNNHMM:
    """Fold-k CNN-HMM for the fused families, after check_held_out."""
    name = run_name(cfg.demo.analysis, cfg.demo.subset, "resnet18")
    cnns, hmms = {}, {}
    for family in cfg.hybrid.families:
        ckpt = cnn_checkpoint_path(cfg, name, family, fold)
        hmm_file = hmm_path(cfg, name, family, fold)
        for path, runner in ((ckpt, "run_cnn"), (hmm_file, "run_hybrid")):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found; run src.training.{runner}")
        cnn, state = load_cnn(ckpt)
        check_held_out(state, code, fold, ckpt)
        cnns[family] = cnn
        hmms[family] = HMMClassifier().load(hmm_file)
    return HybridCNNHMM(cnns, hmms, cfg.demo.device)


def stored_prob(model: str, fs: str, subdir: str, code: str, fold: int, cfg=config):
    """P(SAD) of `code` in a runner's predictions.csv for this model/fold."""
    path = cfg.paths.results_dir / subdir / _run_dir_name(cfg) / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the {subdir} runner first")
    preds = pd.read_csv(path, dtype={"participant_id": str})
    row = preds[
        (preds["model"] == model)
        & (preds["feature_set"] == fs)
        & (preds["fold"] == fold)
        & (preds["participant_id"] == code)
    ]
    if len(row) != 1:
        raise DemoRefusal(f"{path}: no single {model}/{fs} fold-{fold} row for {code}")
    return float(row["prob_sad"].iloc[0])


def _check_match(model, fs, subdir, code, fold, live, cfg) -> None:
    stored = stored_prob(model, fs, subdir, code, fold, cfg)
    if abs(live - stored) > cfg.demo.match_atol:
        raise DemoRefusal(
            f"{model}/{fs}: live P(SAD) {live:.6f} differs from the reported "
            f"out-of-fold {stored:.6f} (tolerance {cfg.demo.match_atol}); the "
            "models or inputs changed since the run"
        )


# ── grad-cam ─────────────────────────────────────────────────────────────────


def _gradcam_module():
    """src.analysis.gradcam (H4) if it exists, else None."""
    if importlib.util.find_spec(GRADCAM_MODULE) is None:
        return None
    return importlib.import_module(GRADCAM_MODULE)


def save_gradcams(hybrid, crops, crop_probs, label, out_dir, cfg=config) -> list:
    """Overlays for the n_gradcam crops that most support `label`."""
    module = _gradcam_module()
    if module is None or hybrid is None:
        return []
    ranked = [
        (p, family, cell, img)
        for family, probs in crop_probs.items()
        for p, cell, img in zip(probs, *crops[family])
    ]
    ranked.sort(key=lambda r: r[0], reverse=label == "SAD")
    paths = []
    for _, family, cell, img in ranked[: cfg.demo.n_gradcam]:
        image = get_eval_transform(cfg, family)(Image.fromarray(img))
        out = out_dir / f"gradcam_{cell}.png"
        paths.append(str(module.save_overlay(hybrid.cnns[family], image, out)))
    return paths


# ── predict ──────────────────────────────────────────────────────────────────


def _label(prob_sad: float, cfg) -> str:
    return "SAD" if prob_sad >= cfg.aggregate.threshold else "HAPPY"


def predict_participant(
    code: str, model: str = config.demo.default_model, fold: int | None = None
) -> dict:
    """Predict one participant with the fold models that never trained on them.

    Returns label, prob_sad, per_family_probs (CNN-HMM mean crop P(SAD) per
    fused family; empty for LR-only models), component_probs, gradcam_paths,
    plus fold, model, in_middle_band and self_report_label.
    """
    cfg = config
    require_frozen_or_synthetic(cfg, "predict")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    code = str(code).strip()
    info = outer_fold(code, fold, cfg)
    k = info["fold"]

    wanted = {
        "ensemble": [m for m, _, _ in COMPONENTS],
        "cnn_hmm": [HYBRID_COMPONENT[0]],
    }.get(model, [model])
    components, per_family, crop_probs = {}, {}, {}
    crops, hybrid = None, None
    if HYBRID_COMPONENT[0] in wanted:
        hybrid = load_hybrid(code, k, cfg)
        crops = load_crops(code, cfg)
        missing = set(cfg.hybrid.families) - set(crops)
        if missing:
            raise DemoRefusal(f"participant {code} has no {sorted(missing)} crops")
        with torch.no_grad():
            for family in cfg.hybrid.families:
                crop_probs[family] = hybrid.crop_prob_sad(crops[family][1], family)
                per_family[family] = float(np.mean(crop_probs[family]))
        live = float(np.mean(list(per_family.values())))
        _check_match(*HYBRID_COMPONENT, cfg.hybrid.output_subdir, code, k, live, cfg)
        components[HYBRID_COMPONENT[0]] = live
    for name in (m for m in wanted if m in LR_SETS):
        live = lr_component(name, code, k, info["train_ids"], cfg)
        _check_match(
            name, LR_SETS[name][0], cfg.baselines.output_subdir, code, k, live, cfg
        )
        components[name] = live

    prob = float(np.mean([components[m] for m in wanted]))
    label = _label(prob, cfg)
    out_dir = cfg.paths.results_dir / cfg.demo.output_dir / code
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "participant_id": code,
        "model": model,
        "fold": k,
        "label": label,
        "prob_sad": prob,
        "per_family_probs": per_family,
        "component_probs": components,
        "gradcam_paths": save_gradcams(hybrid, crops, crop_probs, label, out_dir, cfg),
        "in_middle_band": info["in_middle_band"],
        "self_report_label": info["self_report"],
        "n_crops": {f: len(c[0]) for f, c in (crops or {}).items()},
    }
    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2))
    return result


def _probs(d: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in d.items()) or "n/a"


def print_result(r: dict) -> None:
    band = "  (middle band: outside the primary claim)" if r["in_middle_band"] else ""
    print(f"Participant   : {r['participant_id']}{band}")
    print(f"Model         : {r['model']}  (outer fold {r['fold']} models)")
    print(f"Held out      : yes; the fold-{r['fold']} models never trained on them")
    if r["n_crops"]:
        counts = "  ".join(f"{f} {n}" for f, n in r["n_crops"].items())
        print(f"Crops         : {sum(r['n_crops'].values())}  ({counts})")
    print(f"Prediction    : {r['label']}   P(SAD) = {r['prob_sad']:.3f}")
    print(f"Components    : {_probs(r['component_probs'])}")
    print(f"Per family    : {_probs(r['per_family_probs'])}  (CNN-HMM)")
    print(f"Self-report   : {r['self_report_label']}")
    shown = ", ".join(r["gradcam_paths"]) or "not available (H4 Grad-CAM not built)"
    print(f"Grad-CAM      : {shown}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Defense demo: one participant.")
    parser.add_argument("--code", required=True, help="3-digit participant code")
    parser.add_argument("--model", choices=MODELS, default=config.demo.default_model)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        print_result(predict_participant(args.code, args.model, args.fold))
    except (DemoRefusal, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"REFUSED: {exc}" if isinstance(exc, DemoRefusal) else f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
