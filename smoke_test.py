"""
Smoke test: the whole pipeline on a temporary synthetic root — Stage G.

One command runs every stage end to end and prints PASS / FAIL with
timing, stopping at the first failure:

    A  config, paths, seed
    B  synthetic data (signal, export, raw crops), labeler, questionnaire report
    C  ingest, participant registry, crop manifest + QC status
       (--real-sample: also on copies of 3 real participants, read-only)
    D  preprocessing, transforms, handcrafted + stroke features, embeddings
    E  folds, a deliberately leaky split that must be caught, aggregation
    F  baselines, CNN (1 epoch), CNN-HMM, ensemble   -- primary and full
    G  run log: one point-estimate row per F runner and (model, feature_set)
       for each analysis, CI columns empty
    H  evaluator, permutation tests (LR models, N = SmokeConfig.permutation_n),
       secondary analyses, Grad-CAM, RESULTS.txt

Everything is written under a temporary directory that is removed at the
end; the real DATASET is never written (with --real-sample it is only read,
and its files are checksummed before and after). Sizes and the tiny model
settings come from SmokeConfig; every config override is restored on exit.
Asking for a later stage also runs the earlier ones it needs, shown as
prerequisites.

Run from the project root:

    python smoke_test.py [--stages A-H] [--real-sample DATASET]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.utils.config import config

STAGES = "ABCDEFGH"
ANALYSES = ("primary", "full")
TITLES = {
    "A": "config, paths, seed",
    "B": "synthetic data, labeler, questionnaire report",
    "C": "ingest, registry, crop manifest",
    "D": "preprocessing, transforms, features, embeddings",
    "E": "folds, leakage guard, aggregation",
    "F": "baselines, CNN, CNN-HMM, ensemble",
    "G": "run log",
    "H": "evaluator, permutation, secondary, Grad-CAM, report",
}
FAIL_TAIL_LINES = 25
RUNNERS = {
    "src.training.run_baselines": "baselines",
    "src.training.run_cnn": "cnn",
    "src.training.run_hybrid": "hybrid",
    "src.training.run_ensemble": "ensemble",
}


class StageFailure(Exception):
    """A smoke check did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise StageFailure(message)


def parse_stages(text: str) -> list:
    """'A-H', 'A-C', 'F', or 'A,C,E' -> sorted stage letters."""
    letters = set()
    for part in text.upper().replace(" ", "").split(","):
        lo, _, hi = part.partition("-")
        hi = hi or lo
        if lo not in STAGES or hi not in STAGES or lo > hi:
            raise ValueError(f"bad stage range {part!r}; use letters in {STAGES}")
        letters.update(STAGES[STAGES.index(lo) : STAGES.index(hi) + 1])
    return sorted(letters)


def _run_cli(module, *argv: str) -> None:
    """Run a module's main() as `python -m module argv`; require exit 0."""
    with patch.object(sys, "argv", [module.__name__, *argv]):
        try:
            rc = module.main()
        except SystemExit as exc:
            rc = exc.code
    check(rc in (0, None), f"{module.__name__} {' '.join(argv)} exited {rc}")


def _tree_hashes(*roots: Path) -> dict:
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in roots
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def _scaled(sizes: dict, divisor: int) -> dict:
    return {
        k: (max(1, w // divisor), max(1, h // divisor)) for k, (w, h) in sizes.items()
    }


def _overrides(stack: contextlib.ExitStack, root: Path) -> None:
    """Point config at the temp root with SmokeConfig's tiny settings."""
    s = config.smoke
    settings = [
        (config.paths, "data_root", root),
        (config.paths, "output_root", root),
        (config.labeling, "source_csv", None),
        (config.labeling, "output_csv", None),
        (config.labeling, "min_per_class", s.min_per_class),
        (
            config.crop,
            "expected_size_px",
            _scaled(config.crop.expected_size_px, s.size_divisor),
        ),
        (
            config.preprocessing,
            "canvas_size",
            _scaled(config.preprocessing.canvas_size, s.size_divisor),
        ),
        (config.training, "epochs", s.epochs),
        (config.data, "num_workers", 0),
        (config.cnn, "use_pretrained", s.pretrained),
        (
            config.embedding,
            "weights",
            config.embedding.weights if s.pretrained else None,
        ),
        (config.hmm, "n_states", s.hmm_states),
        (config.hmm, "n_iter", s.hmm_iter),
        (config.hmm, "n_restarts", s.hmm_restarts),
        (config.hybrid, "n_states_grid", (s.hmm_states,)),
        (config.hybrid, "pca_grid", (s.hmm_pca,)),
        (config.hybrid, "topology_grid", (config.hmm.topology,)),
        (config.hybrid, "selection_restarts", s.hmm_restarts),
    ]
    for obj, name, value in settings:
        stack.enter_context(patch.object(obj, name, value))


def stage_a(ctx: dict) -> str:
    import torch

    from src.utils.config import CURSIVE_CROPS, DRAWING_CROPS, WORD_CROPS
    from src.utils.seed import set_seed

    root = ctx["root"]
    check(
        (len(DRAWING_CROPS), len(WORD_CROPS), len(CURSIVE_CROPS)) == (4, 15, 5),
        "crop maps must be 4 + 15 + 5 = 24 cells",
    )
    check(len(config.crop.expected_size_px) == 24, "expected_size_px needs 24 cells")
    for path in (
        config.paths.metadata_dir,
        config.paths.results_dir,
        config.paths.models_dir,
    ):
        check(root in Path(path).parents, f"{path} is outside the temp root")
    config.ensure_output_dirs()
    for raw in (config.paths.raw3_dir, config.paths.raw4_dir):
        check(not raw.exists(), f"ensure_output_dirs created read-only {raw}")
    set_seed(config.training.seed)
    a = torch.rand(4)
    set_seed(config.training.seed)
    check(torch.equal(a, torch.rand(4)), "set_seed is not reproducible")
    return f"root {root}"


def stage_b(ctx: dict) -> str:
    from src.analysis import questionnaire_report
    from src.data import labeler
    from src.utils.synthetic import make_synthetic_root

    s, root = config.smoke, ctx["root"]
    make_synthetic_root(
        root,
        n_participants=s.n_participants,
        signal=True,
        shift=s.signal_shift,
        seed=config.training.seed,
        export=True,
        raw=True,
    )
    meta = config.paths.metadata_dir
    synthetic = pd.read_csv(meta / "labels.csv", dtype={"participant_id": str})
    (meta / "labels.csv").unlink()
    _run_cli(labeler)
    labels = pd.read_csv(meta / "labels.csv", dtype={"participant_id": str})
    check(labels.equals(synthetic), "labeler output differs from the export's labels")
    report = questionnaire_report.build_report()
    check(Path(report).is_file(), "questionnaire report not written")
    ctx["raw_hashes"] = _tree_hashes(config.paths.raw3_dir, config.paths.raw4_dir)
    counts = labels["label"].value_counts().to_dict()
    n_primary = int(labels["in_primary_analysis"].sum())
    return f"{len(labels)} labeled {counts}, {n_primary} primary"


def _index_raw(n_expected: int) -> pd.DataFrame:
    """ingest -> registry -> crop manifest + QC status on config's data root."""
    from src.data import collector, crop_manifest, ingest

    _run_cli(ingest)
    _run_cli(collector, "--build")
    _run_cli(crop_manifest, "--apply-qc-status")
    meta = config.paths.metadata_dir
    manifest = pd.read_csv(meta / "crop_manifest.csv", dtype={"participant_id": str})
    parts = pd.read_csv(meta / "participants.csv", dtype={"participant_id": str})
    check(
        len(manifest) == 24 * n_expected,
        f"{len(manifest)} crops, expected 24 x {n_expected}",
    )
    check(len(parts) == n_expected, f"{len(parts)} participants, expected {n_expected}")
    check((parts["status"] == "qc_passed").all(), "not every participant passed QC")
    return manifest


def _real_sample(ctx: dict, real_sizes: dict) -> str:
    s = config.smoke
    src = Path(ctx["real_sample"])
    src3, src4 = src / "raw-3Page", src / "raw-4Page"
    check(src3.is_dir() and src4.is_dir(), f"{src} has no raw-3Page / raw-4Page")
    codes = sorted(
        p.stem.rsplit("_", 1)[-1] for p in src3.glob("EMHA-P3_DrawingExercise_*.png")
    )[: s.n_real_participants]
    check(
        len(codes) == s.n_real_participants,
        f"fewer than {s.n_real_participants} P3 scans",
    )
    files = [
        p
        for folder in (src3, src4)
        for p in folder.rglob("*.png")
        if any(p.stem.endswith(f"_{c}") or f"_{c}_" in p.stem for c in codes)
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    real_root = ctx["root"].parent / "real_sample"
    for p in files:
        dest = real_root / p.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(config.paths, "data_root", real_root))
        stack.enter_context(patch.object(config.paths, "output_root", real_root))
        stack.enter_context(patch.object(config.crop, "expected_size_px", real_sizes))
        _index_raw(len(codes))
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    check(before == after, "a real raw file changed during the smoke test")
    return f"; real sample {codes} ({len(files)} files, unchanged)"


def stage_c(ctx: dict) -> str:
    _index_raw(config.smoke.n_participants)
    detail = f"{24 * config.smoke.n_participants} crops indexed, all qc_passed"
    if ctx.get("real_sample"):
        detail += _real_sample(ctx, ctx["real_sizes"])
    return detail


def stage_d(ctx: dict) -> str:
    from PIL import Image

    from src.data.transforms import get_eval_transform, get_train_transform
    from src.features import embeddings, handcrafted
    from src.features.strokes import STROKE_FEATURES
    from src.preprocessing import pipeline

    n = config.smoke.n_participants
    manifest = pd.read_csv(pipeline.run(overwrite=True), dtype={"participant_id": str})
    check(
        len(manifest) == 24 * n, f"{len(manifest)} processed crops, expected {24 * n}"
    )
    for family, part in manifest.groupby("task_family"):
        w, h = config.preprocessing.canvas_size[family]
        with Image.open(part["processed_path"].iloc[0]) as img:
            check(img.size == (w, h), f"{family} canvas {img.size} != {(w, h)}")
            gray = img.convert("L")
        for transform in (get_train_transform, get_eval_transform):
            shape = tuple(transform(config, family)(gray).shape)
            check(shape == (1, h, w), f"{transform.__name__}({family}) gave {shape}")

    table = pd.read_csv(handcrafted.run(), dtype={"participant_id": str})
    check(len(table) == n, f"handcrafted table has {len(table)} rows, expected {n}")
    check(not table.isna().any().any(), "NaN in handcrafted features")
    for feat in STROKE_FEATURES:
        check(f"word__{feat}_mean" in table.columns, f"stroke feature {feat} missing")

    for path in embeddings.run():
        with np.load(path, allow_pickle=False) as z:
            check(str(z["cache_key"]) != "synthetic", f"{path} not recomputed")
            check(z["embeddings"].shape[1] == embeddings.EMBED_DIM, f"{path} bad width")
    raw_now = _tree_hashes(config.paths.raw3_dir, config.paths.raw4_dir)
    check(raw_now == ctx["raw_hashes"], "raw crops changed during B-D")
    return (
        f"{len(manifest)} crops processed, {table.shape[1] - 1} features, raw unchanged"
    )


def stage_e(ctx: dict) -> str:
    from src.training import splits
    from src.training.aggregate import aggregate_crops

    splits.run()
    folds = splits.load_folds(config)
    labels = pd.read_csv(
        config.paths.metadata_dir / "labels.csv", dtype={"participant_id": str}
    )
    check(
        set(folds["participant_id"]) == set(labels["participant_id"]),
        "folds.csv must hold every labeled participant",
    )
    ids = sorted(folds["participant_id"])
    train, test = splits.fold_ids(folds, 0, ids)
    try:
        splits.assert_no_leakage(train, test + train[:1])
    except splits.LeakageError:
        pass
    else:
        raise StageFailure("a leaky split was not caught")

    crops = pd.DataFrame(
        {
            "participant_id": ["001"] * 3 + ["002"] * 2,
            "task_family": ["word", "word", "cursive", "drawing", "word"],
            "prob_sad": [0.9, 0.6, 0.3, 0.2, 0.4],
        }
    )
    agg = aggregate_crops(crops).set_index("participant_id")
    check(np.isclose(agg.loc["001", "prob_sad"], 0.6), "mean_prob aggregation wrong")
    check(
        agg.loc["001", "pred"] == "SAD" and agg.loc["002", "pred"] == "HAPPY",
        "aggregated predictions wrong",
    )
    n_folds = folds["outer_fold"].nunique()
    return f"{len(folds)} participants in {n_folds} folds, leak caught"


def stage_f(ctx: dict, analysis: str) -> str:
    from src.training import run_baselines, run_cnn, run_ensemble, run_hybrid
    from src.training.run_baselines import PRED_COLUMNS

    run_baselines.run(analysis)
    for family in config.smoke.cnn_families:
        run_cnn.run(task_family=family, analysis=analysis)
    run_hybrid.run(analysis=analysis)
    preds = pd.read_csv(run_ensemble.run(analysis), dtype={"participant_id": str})
    check(tuple(preds.columns) == PRED_COLUMNS, "ensemble predictions schema differs")
    check(preds["prob_sad"].between(0, 1).all(), "ensemble prob_sad outside [0, 1]")
    check(not preds["participant_id"].duplicated().any(), "a participant tested twice")
    acc = float((preds["pred"] == preds["label"]).mean())
    return f"ensemble on {len(preds)} participants (accuracy {acc:.2f}, synthetic)"


def _complete_groups(path: Path) -> set:
    """(model, feature_set) pairs of a predictions.csv covering every fold."""
    preds = pd.read_csv(path, dtype={"participant_id": str})
    test = preds[preds["in_middle_band"].astype(str) != "True"]
    folds = set(range(config.cv.n_splits))
    return {
        key
        for key, part in test.groupby(["model", "feature_set"])
        if set(part["fold"].astype(int)) == folds
    }


def _run_log() -> pd.DataFrame:
    from src.utils.run_log import log_path

    path = log_path()
    check(path.is_file(), f"{path} not written")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def stage_g(ctx: dict) -> str:
    log = _run_log()
    module = log["command"].str.split().str[2]
    check(set(module) == set(RUNNERS), f"run log modules {sorted(set(module))}")
    for col in ("macro_f1_ci_low", "macro_f1_ci_high"):
        check((log[col] == "").all(), f"a runner row has {col} filled")
    dup = log.duplicated(["command", "analysis", "model", "feature_set"])
    check(not dup.any(), "a runner logged one (model, feature_set) twice")
    for analysis in ANALYSES:
        for mod, folder in RUNNERS.items():
            rows = log[(module == mod) & (log["analysis"] == analysis)]
            logged = set(zip(rows["model"], rows["feature_set"]))
            path = config.paths.results_dir / folder / analysis / "predictions.csv"
            expected = _complete_groups(path)
            check(
                logged == expected,
                f"{mod} {analysis}: logged {sorted(logged)}, "
                f"predictions {sorted(expected)}",
            )
    ctx["runner_rows"] = len(log)
    return f"{len(log)} runner rows, one per (runner, analysis, model), CIs empty"


def stage_h(ctx: dict) -> str:
    from src.analysis import gradcam, permutation, report, secondary
    from src.training import evaluator

    s, pcfg = config.smoke, config.permutation
    final = evaluator.output_dir()
    comparison = pd.read_csv(evaluator.run())
    check(
        set(comparison["analysis"]) == set(ANALYSES),
        f"model_comparison.csv analyses {sorted(set(comparison['analysis']))}",
    )
    with patch.object(pcfg, "n_lr", s.permutation_n), patch.object(
        pcfg, "cnn_analyses", ()
    ):
        jsons = [p for a in ANALYSES for p in permutation.run(a, n_jobs=1)]
    check(
        len(jsons) == len(ANALYSES) * len(pcfg.lr_models),
        f"{len(jsons)} permutation results, expected one per LR model and set",
    )
    secondary.run()
    overlays = gradcam.run()
    results = report.run()

    expected = [
        final / "model_comparison.csv",
        final / "per_fold_metrics.csv",
        final / "confusion_matrices.csv",
        final / "secondary.txt",
        final / "secondary_metrics.csv",
        results,
        *jsons,
        *overlays,
    ]
    absent = [str(p) for p in expected if not Path(p).is_file()]
    check(not absent, f"outputs missing: {absent}")
    check(len(overlays) > 0, "Grad-CAM wrote nothing")
    text = (final / "model_comparison.txt").read_text(encoding="utf-8")
    n_perm = sum("Permutation test," in ln for ln in text.splitlines())
    check(n_perm == len(jsons), f"{n_perm} permutation lines in model_comparison")
    check(
        "SYNTHETIC DATA ROOT" in Path(results).read_text(encoding="utf-8"),
        "RESULTS.txt does not say SYNTHETIC DATA ROOT",
    )
    log = _run_log().iloc[ctx["runner_rows"] :]
    ev_rows = log[log["command"].str.startswith("python -m src.training.evaluator")]
    check(len(ev_rows) == len(comparison), "evaluator run-log rows != comparison")
    for col in ("macro_f1_ci_low", "macro_f1_ci_high"):
        check((ev_rows[col] != "").all(), f"an evaluator row has no {col}")
    return (
        f"{len(comparison)} model rows, {len(jsons)} permutation tests, "
        f"{len(overlays)} Grad-CAM files, RESULTS.txt (synthetic)"
    )


def _steps(wanted: list) -> list:
    """[(label, title, fn, requested)] from A to the last wanted stage."""
    steps = []
    for letter in STAGES[: STAGES.index(wanted[-1]) + 1]:
        requested = letter in wanted
        title = TITLES[letter]
        if letter == "F":
            for analysis in ANALYSES:
                steps.append(
                    (
                        f"F/{analysis}",
                        title,
                        lambda c, a=analysis: stage_f(c, a),
                        requested,
                    )
                )
        else:
            fn = globals()[f"stage_{letter.lower()}"]
            steps.append((letter, title, fn, requested))
    return steps


def main(stages: str = "A-H", real_sample: Path | None = None) -> int:
    """Run the smoke stages; 0 if none failed, else 1."""
    wanted = parse_stages(stages)
    real_sizes = dict(config.crop.expected_size_px)
    start = time.perf_counter()
    failed = False
    print("=" * 72)
    real = f", real sample {real_sample}" if real_sample else ""
    print(
        f"EMHA smoke test: stages {''.join(wanted)}, "
        f"{config.smoke.n_participants} synthetic participants{real}"
    )
    print("=" * 72)
    with tempfile.TemporaryDirectory(
        prefix="emha_smoke_", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp) / "root"
        root.mkdir()
        ctx = {"root": root, "real_sample": real_sample, "real_sizes": real_sizes}
        with contextlib.ExitStack() as stack:
            _overrides(stack, root)
            for label, title, fn, requested in _steps(wanted):
                note = "" if requested else "  (prerequisite)"
                buffer = io.StringIO()
                t0 = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(
                        buffer
                    ):
                        detail = fn(ctx)
                    status = "PASS"
                except (Exception, SystemExit) as exc:
                    status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
                    failed = True
                dt = time.perf_counter() - t0
                print(f"  [{status}] {label:10s} {title}{note}  ({dt:.1f} s)")
                print(f"           {detail}")
                if failed:
                    tail = buffer.getvalue().strip().splitlines()[-FAIL_TAIL_LINES:]
                    if tail:
                        print("           --- last output ---")
                        for line in tail:
                            print(f"           {line}")
                    break
    total = time.perf_counter() - start
    print("=" * 72)
    verdict = "FAILED" if failed else "ALL STAGES PASSED"
    print(f"{verdict} in {total:.1f} s")
    if not failed and total > config.smoke.time_budget_s:
        print(f"WARNING: over the {config.smoke.time_budget_s:.0f} s budget")
    return 1 if failed else 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description="EMHA pipeline smoke test.")
    parser.add_argument("--stages", default="A-H", help="e.g. A-H, A-C, F")
    parser.add_argument(
        "--real-sample",
        type=Path,
        default=None,
        help="data root with raw-3Page/raw-4Page; stage C also indexes copies "
        "of 3 real participants (read-only)",
    )
    args = parser.parse_args()
    return main(args.stages, args.real_sample)


if __name__ == "__main__":
    sys.exit(_cli())
