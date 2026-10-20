"""Notebooks are bootstrap only: no def, class, stored outputs, or tokens."""

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
BOOTSTRAPS = ("colab_bootstrap.ipynb", "kaggle_bootstrap.ipynb")
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
DEF_OR_CLASS = re.compile(r"(^|\W)(def|class) ")
TOKEN_LIKE = re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def _cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return nb["cells"]


def test_bootstrap_notebooks_exist():
    assert {p.name for p in NOTEBOOKS} >= set(BOOTSTRAPS)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_def_or_class(path):
    for i, cell in enumerate(_cells(path)):
        source = "".join(cell["source"])
        assert not DEF_OR_CLASS.search(source), f"{path.name} cell {i}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_outputs_or_tokens(path):
    for i, cell in enumerate(_cells(path)):
        assert not cell.get("outputs"), f"{path.name} cell {i} has stored output"
        assert not TOKEN_LIKE.search("".join(cell["source"])), f"{path.name} cell {i}"


@pytest.mark.parametrize("name", BOOTSTRAPS)
def test_full_clone_and_secret(name):
    text = "".join("".join(c["source"]) for c in _cells(NOTEBOOK_DIR / name))
    assert "--depth" not in text
    assert '"GITHUB_TOKEN"' in text
    assert "git rev-parse HEAD" in text and "torch.cuda.is_available()" in text


def _colab_text():
    return "\n".join(
        "".join(c["source"])
        for c in _cells(NOTEBOOK_DIR / "colab_bootstrap.ipynb")
        if c["cell_type"] == "code"
    )


def _param(text, name):
    match = re.search(rf"^{name} = (.+?)(\s+#.*)?$", text, re.MULTILINE)
    assert match, f"parameter {name} missing"
    return match.group(1).strip()


def test_colab_run_mode_calls_pipeline_runner():
    text = _colab_text()
    assert "src.pipeline_runner" in text
    for name in ("MODE", "REQUIRE_GPU", "AUTO_DISCONNECT", "SUBSET", "ANALYSIS"):
        _param(text, name)
    assert _param(text, "MODE").strip('"') in ("run", "smoke")
    assert '"--subset"' in text and '"--steps"' in text and '"--analysis"' in text


def test_colab_run_mode_data_local_outputs_on_drive():
    text = _colab_text()
    assert 'os.environ["EMHA_DATA_ROOT"] = LOCAL_DATA_ROOT' in text
    assert 'os.environ["EMHA_OUTPUT_ROOT"] = DRIVE_OUTPUT_ROOT' in text
    assert not _param(text, "LOCAL_DATA_ROOT").strip('"').startswith("/content/drive")
    assert _param(text, "DRIVE_OUTPUT_ROOT").strip('"').startswith("/content/drive/")


def test_colab_run_mode_gpu_gate_before_data_copy_and_releases_runtime():
    text = _colab_text()
    assert "runtime.unassign" in text and "flush_and_unmount" in text
    gate = text.index("REQUIRE_GPU not in")
    assert gate < text.index("unpack_archive(DRIVE_ZIP")
    runner = text.index('"-m", "src.pipeline_runner"')
    assert text.index("unpack_archive(DRIVE_ZIP") < runner


def test_colab_smoke_mode_kept():
    text = _colab_text()
    assert 'if MODE == "smoke":' in text and "smoke_test.py" in text
    assert "REAL_SAMPLE" in text and "DRIVE_DATA_ROOT" in text
