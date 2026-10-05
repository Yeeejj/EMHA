"""
One resumable command for the pilot and full runs — Stage G.

Runs RunnerConfig.default_steps (or --steps) in order for each analysis set,
primary first, by calling each module's run function:

    embeddings     src.features.embeddings.run(device)          once per subset
    handcrafted    src.features.handcrafted.run()               once per subset
    baselines      src.training.run_baselines.run(analysis, subset, predict_middle)
    cnn            src.training.run_cnn.run(analysis, subset, predict_middle, device)
    cnn_shuffled   src.training.run_cnn.run(..., shuffle_labels=True)  null run
    hybrid         src.training.run_hybrid.run(analysis, subset, predict_middle,
                                               device)
    ensemble       src.training.run_ensemble.run(<run name>)
    evaluate       src.training.evaluator.run(analysis, subset)
    permutation    src.analysis.permutation.run(analysis, subset)
    secondary      src.analysis.secondary.run(analysis, subset)
    gradcam        src.analysis.gradcam.run(analysis, subset, device)
    report         src.analysis.report.run(analysis, subset)

predict_middle is RunnerConfig.predict_middle on the primary set only. The
first failing step stops the run with its error; a step whose module (or
run function) is not built yet stops it with a "not built" message.

Resume: each finished (subset, analysis, step) writes
results/<RunnerConfig.state_subdir>/<subset>/<analysis>__<step>.json with its
outputs and settings (subset, predict_middle, config fingerprint). A rerun
skips a step whose marker matches and whose outputs exist; a marker whose
settings differ refuses the run (delete it deliberately and log a
Deviation). Inside a step, run_cnn and run_hybrid also resume per
(family, fold), so an interrupted Kaggle session loses at most one fold.

Refuses to start unless the git tag `protocol-frozen` and the analysis-design
decision file (RunnerConfig.decision_file) both exist — on synthetic roots
too. Every runner additionally keeps its own protocol guard.

    python -m src.pipeline_runner --subset pilot|full [--analysis primary,full]
        [--steps baselines,cnn ...] [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.utils import protocol_guard
from src.utils.config import config

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSET_CHOICES = ("pilot", "full")
ANALYSES = ("primary", "full")
ALL_ANALYSES = "all"  # marker name for steps that do not depend on the set

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


class StepNotBuiltError(RuntimeError):
    """A step's module or its run function does not exist yet."""


class ResumeMismatchError(RuntimeError):
    """A finished step's marker was written with different settings."""


@dataclass(frozen=True)
class Ctx:
    analysis: str
    subset: str | None  # None = all participants (the full run)
    run_name: str
    predict_middle: bool
    device: str


@dataclass(frozen=True)
class Step:
    module: str
    call: Callable  # (run_fn, Ctx) -> output path(s) or None
    per_analysis: bool = True


STEPS = {
    "embeddings": Step(
        "src.features.embeddings", lambda f, c: f(device=c.device), False
    ),
    "handcrafted": Step("src.features.handcrafted", lambda f, c: f(), False),
    "baselines": Step(
        "src.training.run_baselines",
        lambda f, c: f(
            analysis=c.analysis, subset=c.subset, predict_middle=c.predict_middle
        ),
    ),
    "cnn": Step(
        "src.training.run_cnn",
        lambda f, c: f(
            analysis=c.analysis,
            subset=c.subset,
            predict_middle=c.predict_middle,
            device=c.device,
        ),
    ),
    "cnn_shuffled": Step(
        "src.training.run_cnn",
        lambda f, c: f(
            analysis=c.analysis, subset=c.subset, shuffle_labels=True, device=c.device
        ),
    ),
    "hybrid": Step(
        "src.training.run_hybrid",
        lambda f, c: f(
            analysis=c.analysis,
            subset=c.subset,
            predict_middle=c.predict_middle,
            device=c.device,
        ),
    ),
    # run_ensemble.run(analysis) only uses its argument as the results folder
    # name (rows keep the components' analysis), so the run name selects the
    # pilot folders written by the component runners.
    "ensemble": Step("src.training.run_ensemble", lambda f, c: f(c.run_name)),
    "evaluate": Step(
        "src.training.evaluator", lambda f, c: f(analysis=c.analysis, subset=c.subset)
    ),
    "permutation": Step(
        "src.analysis.permutation",
        lambda f, c: f(analysis=c.analysis, subset=c.subset),
    ),
    "secondary": Step(
        "src.analysis.secondary", lambda f, c: f(analysis=c.analysis, subset=c.subset)
    ),
    "gradcam": Step(
        "src.analysis.gradcam",
        lambda f, c: f(analysis=c.analysis, subset=c.subset, device=c.device),
    ),
    "report": Step(
        "src.analysis.report", lambda f, c: f(analysis=c.analysis, subset=c.subset)
    ),
}


# ── preconditions ────────────────────────────────────────────────────────────


def precondition_problems(repo_root: Path | None = None) -> list:
    """Reasons the run may not start (empty list = may start)."""
    repo_root = REPO_ROOT if repo_root is None else repo_root
    problems = []
    if not protocol_guard.protocol_frozen(repo_root):
        problems.append(
            f"git tag '{protocol_guard.PROTOCOL_TAG}' not found in {repo_root}: "
            "freeze EVALUATION_PROTOCOL.md first (CLAUDE.md Non-Negotiable 8)"
        )
    decision = repo_root / config.runner.decision_file
    if not decision.is_file():
        problems.append(f"analysis-design decision file {decision} not found")
    return problems


# ── resume markers ───────────────────────────────────────────────────────────


def config_fingerprint() -> str:
    """SHA-256 of every Config section that can change a result."""
    excluded = set(config.runner.fingerprint_exclude)
    settings = {k: v for k, v in asdict(config).items() if k not in excluded}
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_dir(subset: str) -> Path:
    return config.paths.results_dir / config.runner.state_subdir / subset


def marker_path(subset: str, analysis: str, step: str) -> Path:
    return state_dir(subset) / f"{analysis}__{step}.json"


def _settings(ctx: Ctx, fingerprint: str) -> dict:
    return {
        "subset": ctx.subset,
        "predict_middle": ctx.predict_middle,
        "config_sha256": fingerprint,
    }


def _outputs(result) -> list:
    if result is None:
        return []
    items = result if isinstance(result, (list, tuple)) else [result]
    return [str(Path(p)) for p in items]


def is_done(marker: Path, settings: dict) -> bool:
    """True if marker records a finished step with these settings and outputs."""
    if not marker.is_file():
        return False
    state = json.loads(marker.read_text(encoding="utf-8"))
    if state.get("settings") != settings:
        raise ResumeMismatchError(
            f"{marker} was written with settings {state.get('settings')}, now "
            f"{settings}. Changing settings after a step finished is a Deviation: "
            "log it, then delete the marker (and that step's outputs) to rerun."
        )
    return all(Path(p).exists() for p in state.get("outputs", []))


def write_marker(marker: Path, step: str, settings: dict, result, seconds: float):
    marker.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "settings": settings,
        "outputs": _outputs(result),
        "seconds": round(seconds, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(marker)


# ── steps ────────────────────────────────────────────────────────────────────


def resolve(step: str) -> Callable:
    """Import a step's module and return its run function."""
    spec = STEPS[step]
    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        if exc.name != spec.module:  # a missing dependency is a real error
            raise
        raise StepNotBuiltError(f"{spec.module} is not built yet") from exc
    fn = getattr(module, "run", None)
    if not callable(fn):
        raise StepNotBuiltError(f"{spec.module} has no run() function yet")
    return fn


def run_name(analysis: str, subset: str | None) -> str:
    return analysis if subset is None else f"{analysis}_{subset}"


def _plan(analyses: list, steps: list) -> list:
    """[(analysis or ALL_ANALYSES, step)] in run order, primary first."""
    ordered = [a for a in ANALYSES if a in analyses]
    plan = []
    for analysis in ordered:
        for step in steps:
            if not STEPS[step].per_analysis:
                if (ALL_ANALYSES, step) not in plan:
                    plan.append((ALL_ANALYSES, step))
            else:
                plan.append((analysis, step))
    return plan


def _validate(subset: str, analyses: list, steps: list) -> list:
    errors = []
    if subset not in SUBSET_CHOICES:
        errors.append(f"subset must be one of {SUBSET_CHOICES}, got {subset!r}")
    bad = [a for a in analyses if a not in ANALYSES]
    if bad or not analyses:
        errors.append(f"analysis must be from {ANALYSES}, got {analyses}")
    unknown = [s for s in steps if s not in STEPS]
    if unknown or not steps:
        errors.append(f"unknown steps {unknown}; choose from {list(STEPS)}")
    if len(set(steps)) != len(steps):
        errors.append(f"steps repeated: {steps}")
    return errors


def run(
    subset: str,
    analyses: list[str],
    device: str,
    steps: list[str] | None = None,
) -> int:
    """Run the steps for each analysis set; 0 ok, 1 a step failed, 2 refused."""
    steps = list(config.runner.default_steps) if steps is None else list(steps)
    analyses = list(analyses)
    errors = _validate(subset, analyses, steps)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return EXIT_REFUSED
    problems = precondition_problems()
    if problems:
        print("REFUSED: the pipeline may not start:")
        for p in problems:
            print(f"  - {p}")
        return EXIT_REFUSED

    module_subset = None if subset == "full" else subset
    fingerprint = config_fingerprint()
    plan = _plan(analyses, steps)
    print("=" * 72)
    print(f"EMHA pipeline: subset {subset}, analyses {analyses}, device {device}")
    print(f"steps: {', '.join(steps)}")
    print(f"state: {state_dir(subset)}")
    print("=" * 72)
    start = time.perf_counter()
    for analysis, step in plan:
        label = f"{analysis}/{step}"
        ctx = Ctx(
            analysis=ANALYSES[0] if analysis == ALL_ANALYSES else analysis,
            subset=module_subset,
            run_name=run_name(analysis, module_subset),
            predict_middle=config.runner.predict_middle and analysis == "primary",
            device=device,
        )
        settings = _settings(ctx, fingerprint)
        marker = marker_path(subset, analysis, step)
        t0 = time.perf_counter()
        try:
            if is_done(marker, settings):
                print(f"[SKIP] {label}: already finished ({marker.name})")
                continue
            print(f"[RUN ] {label}", flush=True)
            fn = resolve(step)
            result = STEPS[step].call(fn, ctx)
        except StepNotBuiltError as exc:
            print(f"[FAIL] {label}: {exc}. Finished steps are kept; rerun the same")
            print("       command once it exists, or pass --steps without it.")
            return EXIT_FAILED
        except ResumeMismatchError as exc:
            print(f"[FAIL] {label}: {exc}")
            return EXIT_REFUSED
        except (Exception, SystemExit) as exc:  # stop on the first error
            traceback.print_exc()
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            print("       Finished steps are kept; rerun the same command to resume.")
            return EXIT_FAILED
        dt = time.perf_counter() - t0
        write_marker(marker, step, settings, result, dt)
        print(f"[DONE] {label} ({dt:.1f} s)")
    print("=" * 72)
    print(f"ALL STEPS FINISHED in {time.perf_counter() - start:.1f} s")
    return EXIT_OK


def _list_arg(values: list | None) -> list | None:
    """['a,b', 'c'] -> ['a', 'b', 'c'] (comma- or space-separated)."""
    if values is None:
        return None
    return [v for item in values for v in item.split(",") if v]


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run every training and evaluation step, resumably."
    )
    parser.add_argument("--subset", choices=SUBSET_CHOICES, default="full")
    parser.add_argument(
        "--analysis",
        nargs="+",
        default=None,
        help="primary,full (default: RunnerConfig.default_analyses)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        help=f"subset of {','.join(STEPS)} (default: RunnerConfig.default_steps)",
    )
    parser.add_argument("--device", default=config.runner.device)
    args = parser.parse_args(argv)
    analyses = _list_arg(args.analysis) or list(config.runner.default_analyses)
    return run(args.subset, analyses, args.device, _list_arg(args.steps))


if __name__ == "__main__":
    sys.exit(main())
