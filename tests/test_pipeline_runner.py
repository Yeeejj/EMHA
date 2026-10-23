"""Tests for src.pipeline_runner.

Fast tests replace every step with a recording fake; the end-to-end test runs
the real built steps on a smoke-sized synthetic root (smoke_test stages A-E
build it with SmokeConfig's tiny settings). git is always mocked.
"""

import contextlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import smoke_test
from src import pipeline_runner as pr
from src.training import run_cnn
from src.training.run_baselines import PRED_COLUMNS
from src.utils import protocol_guard
from src.utils.config import config

BUILT = [
    "embeddings",
    "handcrafted",
    "baselines",
    "cnn",
    "cnn_shuffled",
    "hybrid",
    "ensemble",
]
SHARED = ["embeddings", "handcrafted"]
MODEL_STEPS = [s for s in BUILT if s not in SHARED]


_REAL_RUN = subprocess.run


def _git(tag_present: bool):
    """Fake `git tag --list protocol-frozen`; any other command runs for real."""

    def fake_run(cmd, *args, **kwargs):
        if list(cmd[:2]) != ["git", "tag"]:
            return _REAL_RUN(cmd, *args, **kwargs)
        out = protocol_guard.PROTOCOL_TAG + "\n" if tag_present else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return fake_run


@pytest.fixture
def tagged(monkeypatch):
    monkeypatch.setattr(protocol_guard.subprocess, "run", _git(True))


@pytest.fixture
def fakes(tmp_path, monkeypatch, tagged):
    """Every step is a fake that records its call and writes one output file."""
    monkeypatch.setattr(config.paths, "output_root", tmp_path)
    calls = []

    def make(step):
        def fake(**kwargs):
            calls.append((step, kwargs))
            out = tmp_path / "out" / f"{step}_{len(calls)}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(step)
            return out

        return fake

    monkeypatch.setattr(pr, "resolve", make)
    return calls


def test_order_primary_first_and_shared_steps_once(fakes):
    assert pr.run("pilot", ["full", "primary"], "cpu", None) == pr.EXIT_OK
    steps = [s for s, _ in fakes]
    once_after = ["secondary", "gradcam", "report"]
    expected = (
        SHARED
        + MODEL_STEPS
        + MODEL_STEPS
        + ["evaluate", "permutation", "permutation"]
        + once_after
    )
    assert steps == expected
    first = fakes[2 : 2 + len(MODEL_STEPS)]
    assert all(kw["analysis"] == "primary" for _, kw in first)
    assert fakes[0][1] == {"device": "cpu"} and fakes[1][1] == {}
    by_step = {s: kw for s, kw in fakes}
    assert by_step["evaluate"] == {"analysis": None, "subset": "pilot"}
    perms = [kw["analysis"] for s, kw in fakes if s == "permutation"]
    assert perms == ["primary", "full"]
    for step in once_after:
        assert by_step[step] == {"subset": "pilot"}


def test_gated_steps_wait_for_every_analysis(fakes, capsys):
    steps = MODEL_STEPS + ["evaluate", "secondary", "gradcam", "report"]
    assert pr.run("pilot", ["primary"], "cpu", steps) == pr.EXIT_OK
    out = capsys.readouterr().out
    assert [s for s, _ in fakes] == MODEL_STEPS
    for step in ("evaluate", "secondary", "gradcam", "report"):
        assert f"[WAIT] {step}: needs full/baselines" in out
        assert not pr.marker_path("pilot", pr.ALL_ANALYSES, step).exists()
    assert "waiting for the other analysis set" in out

    fakes.clear()
    assert pr.run("pilot", ["primary", "full"], "cpu", steps) == pr.EXIT_OK
    ran = [s for s, _ in fakes]
    assert ran == MODEL_STEPS + ["evaluate", "secondary", "gradcam", "report"]
    assert "[WAIT]" not in capsys.readouterr().out


def test_gate_uses_the_commands_model_steps(fakes, capsys):
    without_null = [s for s in MODEL_STEPS if s != "cnn_shuffled"]
    steps = without_null + ["evaluate"]
    assert pr.run("pilot", ["primary", "full"], "cpu", steps) == pr.EXIT_OK
    assert [s for s, _ in fakes][-1] == "evaluate"
    assert "[WAIT]" not in capsys.readouterr().out

    marker = pr.marker_path("pilot", pr.ALL_ANALYSES, "evaluate")
    marker.unlink()
    fakes.clear()
    assert pr.run("pilot", ["primary", "full"], "cpu", ["evaluate"]) == pr.EXIT_OK
    assert fakes == []
    assert "[WAIT] evaluate: needs primary/cnn_shuffled" in capsys.readouterr().out


def test_arguments_per_analysis_and_subset(fakes):
    assert pr.run("pilot", ["primary", "full"], "cuda", BUILT) == pr.EXIT_OK
    by = {(s, kw.get("analysis")): kw for s, kw in fakes}
    assert by[("baselines", "primary")] == {
        "analysis": "primary",
        "subset": "pilot",
        "predict_middle": config.runner.predict_middle,
    }
    assert by[("baselines", "full")]["predict_middle"] is False
    assert by[("cnn_shuffled", "primary")]["shuffle_labels"] is True
    assert by[("hybrid", "full")]["device"] == "cuda"
    ensembles = [kw for s, kw in fakes if s == "ensemble"]
    assert ensembles == [
        {"analysis": "primary", "subset": "pilot"},
        {"analysis": "full", "subset": "pilot"},
    ]

    fakes.clear()
    assert pr.run("full", ["primary"], "cpu", ["baselines", "ensemble"]) == 0
    assert fakes[0][1]["subset"] is None
    assert fakes[1][1] == {"analysis": "primary", "subset": None}


def test_rerun_skips_finished_steps(fakes, capsys):
    assert pr.run("pilot", ["primary"], "cpu", BUILT) == pr.EXIT_OK
    n = len(fakes)
    assert pr.run("pilot", ["primary"], "cpu", BUILT) == pr.EXIT_OK
    assert len(fakes) == n
    assert capsys.readouterr().out.count("[SKIP]") == len(BUILT)

    marker = pr.marker_path("pilot", "primary", "hybrid")
    Path(pr.json.loads(marker.read_text())["outputs"][0]).unlink()
    assert pr.run("pilot", ["primary"], "cpu", BUILT) == pr.EXIT_OK
    assert [s for s, _ in fakes[n:]] == ["hybrid"]


def test_changed_settings_refuse_to_resume(fakes, monkeypatch, capsys):
    assert pr.run("pilot", ["primary"], "cpu", ["baselines"]) == pr.EXIT_OK
    monkeypatch.setattr(config.training, "epochs", config.training.epochs + 1)
    assert pr.run("pilot", ["primary"], "cpu", ["baselines"]) == pr.EXIT_REFUSED
    assert "Deviation" in capsys.readouterr().out
    assert len(fakes) == 1


def test_device_does_not_change_the_fingerprint(fakes):
    assert pr.run("pilot", ["primary"], "cpu", ["cnn"]) == pr.EXIT_OK
    assert pr.run("pilot", ["primary"], "cuda", ["cnn"]) == pr.EXIT_OK
    assert len(fakes) == 1


def test_stops_at_first_error(fakes, monkeypatch, capsys):
    def boom(f, c):
        raise ValueError("bad fold")

    monkeypatch.setitem(pr.STEPS, "cnn", pr.Step("src.training.run_cnn", boom))
    assert pr.run("pilot", ["primary", "full"], "cpu", BUILT) == pr.EXIT_FAILED
    out = capsys.readouterr().out
    assert "[FAIL] primary/cnn: ValueError: bad fold" in out
    assert [s for s, _ in fakes] == ["embeddings", "handcrafted", "baselines"]
    assert pr.marker_path("pilot", "primary", "baselines").is_file()
    assert not pr.marker_path("pilot", "primary", "cnn").exists()


def test_unbuilt_module_stops_with_clear_message(tmp_path, monkeypatch, tagged):
    monkeypatch.setattr(config.paths, "output_root", tmp_path)
    monkeypatch.setitem(
        pr.STEPS,
        "baselines",
        pr.Step("src.analysis._not_built_for_test", lambda f, c: f()),
    )
    with pytest.raises(pr.StepNotBuiltError, match="not built yet"):
        pr.resolve("baselines")
    assert pr.run("pilot", ["primary"], "cpu", ["baselines"]) == pr.EXIT_FAILED


def test_module_without_run_is_not_built(monkeypatch):
    monkeypatch.setitem(pr.STEPS, "baselines", pr.Step("src.utils.seed", None))
    with pytest.raises(pr.StepNotBuiltError, match="no run"):
        pr.resolve("baselines")


def _ctx_variants():
    for analysis in (pr.ALL_ANALYSES, *pr.ANALYSES):
        for subset in ("pilot", None):
            yield pr.Ctx(
                analysis="primary" if analysis == pr.ALL_ANALYSES else analysis,
                subset=subset,
                run_name=pr.run_name(analysis, subset),
                predict_middle=analysis == "primary",
                device="cpu",
            )


@pytest.mark.parametrize("step", list(pr.STEPS))
def test_every_step_matches_the_real_run_signature(step):
    """Bind each STEPS call against the real module's run(): signature drift
    must fail here, not in a Kaggle session."""
    spec = pr.STEPS[step]
    real = importlib.import_module(spec.module).run
    signature = inspect.signature(real)

    def bind_only(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return None

    for ctx in _ctx_variants():
        spec.call(bind_only, ctx)


def test_refuses_without_protocol_tag(fakes, monkeypatch, capsys):
    monkeypatch.setattr(protocol_guard.subprocess, "run", _git(False))
    assert pr.run("pilot", ["primary"], "cpu", None) == pr.EXIT_REFUSED
    out = capsys.readouterr().out
    assert "REFUSED" in out and protocol_guard.PROTOCOL_TAG in out
    assert fakes == []


def test_refuses_without_decision_file(fakes, monkeypatch, capsys):
    monkeypatch.setattr(config.runner, "decision_file", "DOCS/decisions/missing.txt")
    assert pr.run("full", ["primary"], "cpu", None) == pr.EXIT_REFUSED
    assert "decision file" in capsys.readouterr().out
    assert fakes == []


def test_bad_arguments_refused(fakes):
    assert pr.run("half", ["primary"], "cpu", None) == pr.EXIT_REFUSED
    assert pr.run("pilot", ["middle"], "cpu", None) == pr.EXIT_REFUSED
    assert pr.run("pilot", ["primary"], "cpu", ["train"]) == pr.EXIT_REFUSED
    assert fakes == []


def test_cli_parses_lists(monkeypatch):
    seen = {}

    def fake_run(subset, analyses, device, steps):
        seen.update(subset=subset, analyses=analyses, device=device, steps=steps)
        return 0

    monkeypatch.setattr(pr, "run", fake_run)
    argv = ["--subset", "pilot", "--analysis", "primary,full"]
    assert pr.main(argv + ["--steps", "baselines,cnn", "hybrid"]) == 0
    assert seen == {
        "subset": "pilot",
        "analyses": ["primary", "full"],
        "device": config.runner.device,
        "steps": ["baselines", "cnn", "hybrid"],
    }
    assert pr.main(["--device", "cuda"]) == 0
    assert seen["subset"] == "full" and seen["steps"] is None
    assert seen["analyses"] == list(config.runner.default_analyses)


@pytest.fixture(scope="module")
def e2e_root(tmp_path_factory):
    """Smoke-sized synthetic root after stages A-E (labels, crops, folds).

    Tiny settings: SmokeConfig's, plus the CNN steps train only the families
    the CNN-HMM and ensemble need (SmokeConfig.cnn_families).
    """
    root = tmp_path_factory.mktemp("runner") / "root"
    root.mkdir()
    ctx = {"root": root, "real_sample": None}
    with contextlib.ExitStack() as stack, pytest.MonkeyPatch.context() as mp:
        smoke_test._overrides(stack, root)
        mp.setattr(protocol_guard.subprocess, "run", _git(True))
        mp.setattr(run_cnn, "FAMILIES", tuple(config.smoke.cnn_families))
        for stage in (
            smoke_test.stage_a,
            smoke_test.stage_b,
            smoke_test.stage_c,
            smoke_test.stage_d,
            smoke_test.stage_e,
        ):
            stage(ctx)
        yield root


def test_end_to_end_on_synthetic_fixture(e2e_root, monkeypatch, capsys):
    monkeypatch.setattr(config.permutation, "n_lr", config.smoke.permutation_n)
    analyses = list(config.runner.default_analyses)
    plan = pr._plan(analyses, list(config.runner.default_steps))
    assert pr.run("pilot", analyses, "cpu", None) == pr.EXIT_OK
    out = capsys.readouterr().out
    assert out.count("[DONE]") == len(plan) and "ALL STEPS FINISHED" in out
    assert "[SKIP] cnn_head_fused: CNN-path tests run on the full run only" in out

    results = config.paths.results_dir
    for analysis in analyses:
        for sub in ("baselines", "cnn", "cnn_shuffled", "hybrid", "ensemble"):
            path = results / sub / f"{analysis}_pilot" / "predictions.csv"
            assert path.is_file(), path
    ens = pd.read_csv(
        results / "ensemble" / "primary_pilot" / "predictions.csv",
        dtype={"participant_id": str},
    )
    assert tuple(ens.columns) == PRED_COLUMNS
    assert set(ens["analysis"]) == {"primary"}
    assert ens["in_middle_band"].any()
    assert ens["prob_sad"].between(0, 1).all()
    for analysis, step in plan:
        assert pr.marker_path("pilot", analysis, step).is_file(), step

    final = results / "final_pilot"
    comp = pd.read_csv(final / "model_comparison.csv")
    assert set(comp["analysis"]) == set(analyses)
    assert ("ensemble", "all") in set(zip(comp["model"], comp["feature_set"]))
    text = (final / "model_comparison.txt").read_text(encoding="utf-8")
    perm_lines = [ln for ln in text.splitlines() if "Permutation test," in ln]
    assert len(perm_lines) == 2 * len(config.permutation.pilot_models)
    assert (final / "secondary.txt").is_file()
    assert (final / config.gradcam.output_subdir).is_dir()
    report = (final / "RESULTS.txt").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA ROOT" in report and "SUBSET PILOT" in report
    assert not (results / "final").exists()

    from src.training import evaluator

    evaluator.run(None, "pilot")
    again = (final / "model_comparison.txt").read_text(encoding="utf-8")
    assert [ln for ln in again.splitlines() if "Permutation test," in ln] == (
        perm_lines
    )

    def no_call(step):
        raise AssertionError(f"{step} resolved on a finished run")

    monkeypatch.setattr(pr, "resolve", no_call)
    assert pr.run("pilot", analyses, "cpu", None) == pr.EXIT_OK
    assert capsys.readouterr().out.count("[SKIP]") == len(plan)


def test_end_to_end_refuses_without_tag(e2e_root, monkeypatch, capsys):
    monkeypatch.setattr(protocol_guard.subprocess, "run", _git(False))
    before = sorted(p for p in config.paths.results_dir.rglob("*") if p.is_file())
    assert pr.run("full", ["primary"], "cpu", BUILT) == pr.EXIT_REFUSED
    after = sorted(p for p in config.paths.results_dir.rglob("*") if p.is_file())
    assert before == after
    assert "REFUSED" in capsys.readouterr().out


def test_thread_report_covers_main_process_and_a_loky_worker(monkeypatch):
    monkeypatch.setattr(config.runner, "hmm_n_jobs", 2)
    report = pr.thread_report()
    main, worker = report["main"], report["worker"]
    assert worker["pid"] != main["pid"]
    for state in (main, worker):
        assert set(state["env"]) == set(pr.THREAD_ENV_VARS)
        assert any(p.startswith("blas:") for p in state["pools"])
        assert any(p.startswith("openmp:") for p in state["pools"])


def test_pinned_env_is_seen_in_main_process_and_worker(tmp_path):
    """A fresh process started with the pin: both sides report 1 thread."""
    code = (
        "import json; from src import pipeline_runner as pr; "
        "from src.utils.config import config; config.runner.hmm_n_jobs = 2; "
        "print(json.dumps(pr.thread_report()))"
    )
    env = {**os.environ, **{k: "1" for k in pr.THREAD_ENV_VARS}}
    done = _REAL_RUN(
        [sys.executable, "-c", code],
        cwd=Path(pr.__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["worker"]["pid"] != report["main"]["pid"]
    for state in report.values():
        assert state["env"] == {k: "1" for k in pr.THREAD_ENV_VARS}
        assert all(p.endswith("=1") for p in state["pools"]), state["pools"]


def test_run_header_prints_thread_report(fakes, capsys):
    pr.run("full", ["primary"], "cpu", ["baselines"])
    out = capsys.readouterr().out
    assert "threads (main pid" in out and "threads (worker pid" in out
    assert "OMP_NUM_THREADS" in out and "blas:" in out
