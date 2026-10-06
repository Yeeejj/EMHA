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
