"""Tests for src.pipeline_runner.

Fast tests replace every step with a recording fake; the end-to-end test runs
the real built steps on a smoke-sized synthetic root (smoke_test stages A-E
build it with SmokeConfig's tiny settings). git is always mocked.
"""

import contextlib
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import smoke_test
from src import pipeline_runner as pr
from src.training import run_cnn
from src.training.run_baselines import PRED_COLUMNS
from src.utils import protocol_guard
from src.utils.config import config

# Steps whose modules exist today (Stage H modules are not built yet).
BUILT = [
    "embeddings",
    "handcrafted",
    "baselines",
    "cnn",
    "cnn_shuffled",
    "hybrid",
    "ensemble",
]


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
    # ensemble passes its run name positionally; wrap so the fake records it
    real_ensemble = pr.STEPS["ensemble"]
    monkeypatch.setitem(
        pr.STEPS,
        "ensemble",
        pr.Step(real_ensemble.module, lambda f, c: f(name=c.run_name)),
    )
    return calls


def test_order_primary_first_and_shared_steps_once(fakes):
    assert pr.run("pilot", ["full", "primary"], "cpu", None) == pr.EXIT_OK
    steps = [s for s, _ in fakes]
    shared = ["embeddings", "handcrafted"]
    per = [s for s in config.runner.default_steps if s not in shared]
    assert steps == shared + per + per
    first = fakes[2 : 2 + len(per)]
    assert all(kw.get("analysis", "primary") == "primary" for _, kw in first)
    assert fakes[0][1] == {"device": "cpu"} and fakes[1][1] == {}


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
    names = [kw["name"] for s, kw in fakes if s == "ensemble"]
    assert names == ["primary_pilot", "full_pilot"]

    fakes.clear()
    assert pr.run("full", ["primary"], "cpu", ["baselines", "ensemble"]) == 0
    assert fakes[0][1]["subset"] is None
    assert fakes[1][1] == {"name": "primary"}


def test_rerun_skips_finished_steps(fakes, capsys):
    assert pr.run("pilot", ["primary"], "cpu", BUILT) == pr.EXIT_OK
    n = len(fakes)
    assert pr.run("pilot", ["primary"], "cpu", BUILT) == pr.EXIT_OK
    assert len(fakes) == n  # nothing called again
    assert capsys.readouterr().out.count("[SKIP]") == len(BUILT)

    # a step whose recorded output vanished is redone; the others stay skipped
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
        "report",
        pr.Step("src.analysis._not_built_for_test", lambda f, c: f()),
    )
    with pytest.raises(pr.StepNotBuiltError, match="not built yet"):
        pr.resolve("report")
    assert pr.run("pilot", ["primary"], "cpu", ["report"]) == pr.EXIT_FAILED


def test_module_without_run_is_not_built(monkeypatch):
    monkeypatch.setitem(pr.STEPS, "report", pr.Step("src.utils.seed", None))
    with pytest.raises(pr.StepNotBuiltError, match="no run"):
        pr.resolve("report")


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


# ── end to end on the synthetic fixture ─────────────────────────────────────


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
    assert pr.run("pilot", ["primary"], "cpu", BUILT) == pr.EXIT_OK
    out = capsys.readouterr().out
    assert out.count("[DONE]") == len(BUILT) and "ALL STEPS FINISHED" in out

    results = config.paths.results_dir
    for sub in ("baselines", "cnn", "cnn_shuffled", "hybrid", "ensemble"):
        assert (results / sub / "primary_pilot" / "predictions.csv").is_file(), sub
    ens = pd.read_csv(
        results / "ensemble" / "primary_pilot" / "predictions.csv",
        dtype={"participant_id": str},
    )
    assert tuple(ens.columns) == PRED_COLUMNS
    assert set(ens["analysis"]) == {"primary"}
    assert ens["in_middle_band"].any()  # predict_middle on the primary set
    assert ens["prob_sad"].between(0, 1).all()
    for step in BUILT:
        analysis = "all" if step in ("embeddings", "handcrafted") else "primary"
        assert pr.marker_path("pilot", analysis, step).is_file(), step

    # rerun: every step skipped, no module run function called
    def no_call(step):
        raise AssertionError(f"{step} resolved on a finished run")

    monkeypatch.setattr(pr, "resolve", no_call)
    assert pr.run("pilot", ["primary"], "cpu", BUILT) == pr.EXIT_OK
    assert capsys.readouterr().out.count("[SKIP]") == len(BUILT)


def test_end_to_end_refuses_without_tag(e2e_root, monkeypatch, capsys):
    monkeypatch.setattr(protocol_guard.subprocess, "run", _git(False))
    before = sorted(p for p in config.paths.results_dir.rglob("*") if p.is_file())
    assert pr.run("full", ["primary"], "cpu", BUILT) == pr.EXIT_REFUSED
    after = sorted(p for p in config.paths.results_dir.rglob("*") if p.is_file())
    assert before == after
    assert "REFUSED" in capsys.readouterr().out
