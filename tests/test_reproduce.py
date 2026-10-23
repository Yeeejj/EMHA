"""scripts/reproduce_full.{sh,ps1} end to end on the synthetic fixture.

A module fixture builds one synthetic "dataset" the way smoke_test stages B
and C do (export, labels, raw scans and crops, QC'd manifest, folds), then
copies it to fresh roots. Each script reproduces the results on its own copy
through tests/reproduce_launcher.py (smoke-sized config); the two runs must
agree. The scripts must also stop at the first failing step.
"""

import contextlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import smoke_test
from src.utils.config import config

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "tests" / "reproduce_launcher.py"
WSL_LAUNCHER_DIRS = ("system32", "windowsapps")


def _is_wsl_launcher(exe: Path) -> bool:
    return exe.parent.name.lower() in WSL_LAUNCHER_DIRS


def _find_bash(path=None):
    """Git Bash on Windows (never the WSL launcher); plain bash elsewhere.

    Returns (bash, skip_reason); bash is None when no usable bash exists.
    """
    if os.name != "nt":
        return shutil.which("bash", path=path), "bash not found on PATH"
    git = shutil.which("git", path=path)
    if git:
        for parent in list(Path(git).resolve().parents)[:3]:
            candidate = parent / "bin" / "bash.exe"
            if candidate.is_file() and not _is_wsl_launcher(candidate):
                return str(candidate), ""
    wsl = []
    search = os.environ.get("PATH", "") if path is None else path
    for folder in filter(None, search.split(os.pathsep)):
        candidate = Path(folder) / "bash.exe"
        if not candidate.is_file():
            continue
        if not _is_wsl_launcher(candidate):
            return str(candidate), ""
        wsl.append(str(candidate))
    if wsl:
        return None, (
            "only the WSL launcher was found (" + ", ".join(wsl) + "); "
            "the [sh] tests need Git Bash: install Git for Windows or put "
            "its bin folder on PATH"
        )
    return None, "Git Bash not found (no git install and no bash.exe on PATH)"


BASH, BASH_SKIP = _find_bash()
PWSH = shutil.which("pwsh") or shutil.which("powershell")
SHELLS = {
    "sh": ([BASH, (REPO / "scripts" / "reproduce_full.sh").as_posix()], BASH),
    "ps1": (
        [PWSH, "-NoProfile", "-File", str(REPO / "scripts" / "reproduce_full.ps1")],
        PWSH,
    ),
}
FLAGS = {"sh": ["--device", "cpu", "--skip-exploratory"]}
FLAGS["ps1"] = ["-Device", "cpu", "-SkipExploratory"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """The frozen inputs a fresh clone would get with the dataset."""
    root = tmp_path_factory.mktemp("reproduce") / "dataset"
    with contextlib.ExitStack() as stack:
        smoke_test._overrides(stack, root)
        ctx = {"root": root}
        smoke_test.stage_b(ctx)
        smoke_test.stage_c(ctx)
    for derived in ("results", "models"):
        shutil.rmtree(root / derived, ignore_errors=True)
    return root


def _run(kind, root, *extra):
    cmd, exe = SHELLS[kind]
    if exe is None:
        pytest.skip(BASH_SKIP if kind == "sh" else "no PowerShell found")
    env = {
        **os.environ,
        "EMHA_DATA_ROOT": root.as_posix(),
        "EMHA_OUTPUT_ROOT": root.as_posix(),
        "EMHA_PYTHON": sys.executable,
        "EMHA_PYTHON_PREFIX": LAUNCHER.as_posix(),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        [*cmd, *FLAGS[kind], *extra],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _steps(output: str) -> list:
    return [ln.strip() for ln in output.splitlines() if ln.startswith("==> ")]


def test_both_scripts_list_the_same_steps(dataset):
    sh, ps1 = _run("sh", dataset, "--dry-run"), _run("ps1", dataset, "-DryRun")
    assert sh.returncode == 0 and ps1.returncode == 0, sh.stderr + ps1.stderr
    steps = _steps(sh.stdout)
    assert steps == _steps(ps1.stdout)
    modules = [s.split()[3] for s in steps if s.startswith("==> python -m")]
    assert modules[0] == "src.data.labeler" and modules[-1] == "src.analysis.report"
    assert "src.data.crop_manifest" not in modules


@pytest.fixture(scope="module")
def reproduced(dataset, tmp_path_factory):
    out = {}
    for kind in SHELLS:
        root = tmp_path_factory.mktemp(f"clone_{kind}") / "root"
        shutil.copytree(dataset, root)
        out[kind] = (root, _run(kind, root))
    return out


@pytest.mark.parametrize("kind", list(SHELLS))
def test_script_runs_end_to_end(reproduced, kind):
    root, proc = reproduced[kind]
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    final = root / "results" / "final"
    for name in (
        "RESULTS.txt",
        "model_comparison.csv",
        "secondary.txt",
        "permutation_ensemble_primary.json",
        "permutation_lr_embeddings_full.json",
    ):
        assert (final / name).is_file(), name
    assert list((final / "gradcam").glob("gradcam_*.png"))
    assert "synthetic root: not checked" in proc.stdout


def _results_body(path: Path) -> str:
    """RESULTS.txt sections 1-4 (numbers); provenance paths differ by root."""
    text = path.read_text(encoding="utf-8")
    return text[text.index("\n1. ") : text.index("\n5. ")]


def test_two_fresh_runs_reproduce_the_same_numbers(reproduced):
    (root_a, proc_a), (root_b, proc_b) = reproduced["sh"], reproduced["ps1"]
    assert proc_a.returncode == 0 and proc_b.returncode == 0
    for name in (
        "model_comparison.csv",
        "per_fold_metrics.csv",
        "secondary_metrics.csv",
    ):
        a = (root_a / "results" / "final" / name).read_bytes()
        assert a == (root_b / "results" / "final" / name).read_bytes(), name
    body_a = _results_body(root_a / "results" / "final" / "RESULTS.txt")
    body_b = _results_body(root_b / "results" / "final" / "RESULTS.txt")
    assert body_a == body_b
    assert re.search(r"\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]", body_a)


@pytest.mark.parametrize("kind", list(SHELLS))
def test_script_stops_at_the_first_failing_step(dataset, tmp_path, kind):
    root = tmp_path / "broken"
    shutil.copytree(dataset, root)
    (root / "metadata" / "questionnaire_export.csv").unlink()
    proc = _run(kind, root)
    assert proc.returncode != 0
    steps = _steps(proc.stdout)
    assert steps[-1] == "==> python -m src.data.labeler"
    assert not (root / "results" / "questionnaire").exists()


def test_launcher_refuses_a_real_data_root(tmp_path):
    env = {**os.environ, "EMHA_DATA_ROOT": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, str(LAUNCHER), "-m", "src.analysis.report"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "not a synthetic data root" in proc.stderr
    assert config.paths.data_root


@pytest.mark.skipif(os.name != "nt", reason="Windows bash lookup")
def test_find_bash_never_returns_the_wsl_launcher(tmp_path):
    system32 = tmp_path / "System32"
    system32.mkdir()
    (system32 / "bash.exe").touch()
    bash, reason = _find_bash(path=str(system32))
    assert bash is None
    assert "only the WSL launcher" in reason and str(system32) in reason


@pytest.mark.skipif(os.name != "nt", reason="Windows bash lookup")
def test_find_bash_prefers_git_install_over_path(tmp_path):
    git_root, system32 = tmp_path / "Git", tmp_path / "System32"
    for folder in (git_root / "cmd", git_root / "bin", system32):
        folder.mkdir(parents=True)
    for exe in (git_root / "cmd" / "git.exe", git_root / "bin" / "bash.exe"):
        exe.touch()
    (system32 / "bash.exe").touch()
    search = os.pathsep.join([str(system32), str(git_root / "cmd")])
    bash, _ = _find_bash(path=search)
    assert Path(bash) == (git_root / "bin" / "bash.exe").resolve()
