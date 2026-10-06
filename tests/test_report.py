"""Tests for src.analysis.report: every number in RESULTS.txt matches its source.

A module fixture builds real upstream outputs on a synthetic root
(questionnaire report, baselines, evaluator, a small permutation test,
secondary analyses) and then RESULTS.txt from them.
"""

import hashlib
import json
import re

import pandas as pd
import pytest

from src.analysis import permutation, questionnaire_report, report, secondary
from src.training import evaluator, run_baselines
from src.utils.config import config
from src.utils.synthetic import make_synthetic_root

NUM = r"[-+]?\d+\.\d+"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        root = make_synthetic_root(
            tmp_path_factory.mktemp("report") / "root",
            n_participants=120,
            signal=True,
            export=True,
        )
        mp.setattr(config.paths, "data_root", root)
        mp.setattr(config.paths, "output_root", root)
        mp.setattr(config.labeling, "output_csv", None)
        mp.setattr(config.evaluation, "n_bootstrap", 200)
        questionnaire_report.build_report()
        run_baselines.run("primary")
        run_baselines.run("full")
        evaluator.run(stage="smoke")
        permutation.permutation_test("lr_handcrafted", "primary", n=3, n_jobs=1)
        secondary.run()
        path = report.build()
        yield root, path, path.read_text(encoding="utf-8")


def _section(text, number):
    start = text.index(f"\n{number}. ")
    nxt = re.search(rf"\n{number + 1}\. ", text[start + 1 :])
    return text[start : start + 1 + nxt.start()] if nxt else text[start:]


def test_writes_results_txt(built):
    root, path, text = built
    assert path == root / "results" / "final" / "RESULTS.txt"
    for n, title in enumerate(
        (
            "DATASET SUMMARY",
            "PRIMARY RESULTS",
            "PAIRED COMPARISONS",
            "SECONDARY AND EXPLORATORY",
            "FIGURES",
            "DATA FILES",
            "PROVENANCE",
            "DEVIATIONS",
        ),
        start=1,
    ):
        assert f"\n{n}. {title}" in text
    assert "SYNTHETIC DATA ROOT" in text


def test_every_results_row_matches_model_comparison_csv(built):
    root, _, text = built
    comp = pd.read_csv(root / "results" / "final" / "model_comparison.csv")
    perm = json.loads(
        (
            root / "results" / "final" / "permutation_lr_handcrafted_primary.json"
        ).read_text()
    )
    rows = {}
    for line in _section(text, 2).splitlines():
        m = re.match(r"\s+(\S+/\S+)(?: \[[^\]]+\])?\s+(\d+)\s+(.*)$", line)
        if m and "/" in m.group(1) and not line.strip().startswith("from"):
            rows.setdefault(m.group(1), []).append((int(m.group(2)), m.group(3)))
    checked = 0
    for _, r in comp.iterrows():
        key = f"{r['model']}/{r['feature_set']}"
        n, rest = rows[key].pop(0)  # primary rows come first, then full
        nums = [float(x) for x in re.findall(NUM, rest)]
        expected = [
            r[c]
            for c in (
                "macro_f1",
                "macro_f1_ci_low",
                "macro_f1_ci_high",
                "accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high",
                "majority_baseline",
                "balanced_accuracy",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
            )
        ]
        assert n == r["n_participants"]
        assert nums[:13] == pytest.approx(expected, abs=5e-4), key
        if (r["analysis"], r["model"], r["feature_set"]) == (
            "primary",
            "lr_handcrafted",
            "all",
        ):
            assert nums[13] == pytest.approx(perm["p_value"], abs=5e-5)
            assert f"(N = {perm['n']})" in rest
        else:
            assert len(nums) == 13 and rest.endswith("not run")
        checked += 1
    assert checked == len(comp)


def test_paired_rows_match_model_comparison_csv(built):
    root, _, text = built
    comp = pd.read_csv(root / "results" / "final" / "model_comparison.csv")
    section = _section(text, 3)
    paired = comp[comp["delta_macro_f1_vs_reference"].notna()]
    assert len(paired) > 0
    for analysis in config.evaluation.analyses:
        block = section.split(f"-- {analysis} --")[1].split("--")[0]
        for _, r in paired[paired["analysis"] == analysis].iterrows():
            line = next(
                ln
                for ln in block.splitlines()
                if ln.split()[:1] == [f"{r['model']}/{r['feature_set']}"]
            )
            nums = [float(x) for x in re.findall(NUM, line)]
            assert nums == pytest.approx(
                [
                    r["delta_macro_f1_vs_reference"],
                    r["delta_macro_f1_ci_low"],
                    r["delta_macro_f1_ci_high"],
                ],
                abs=5e-4,
            )


def test_quoted_sections_are_verbatim(built):
    root, _, text = built
    q = (root / "results" / "questionnaire" / "report.txt").read_text(encoding="utf-8")
    alpha = next(ln for ln in q.splitlines() if ln.startswith("Cronbach's alpha"))
    assert f"  {alpha}" in _section(text, 1)
    sec = (root / "results" / "final" / "secondary.txt").read_text(encoding="utf-8")
    section4 = _section(text, 4)
    for line in sec.splitlines():
        if line:
            assert f"  {line}" in section4
    design = report.DESIGN.read_text(encoding="utf-8")
    fraction = next(ln for ln in design.splitlines() if ln.startswith("Middle-band"))
    assert f"  {fraction}" in _section(text, 1)
    protocol = report.PROTOCOL.read_text(encoding="utf-8")
    deviations = protocol.split(report.DEVIATIONS_HEADING)[1].strip().splitlines()[0]
    assert f"  {deviations}" in _section(text, 8)


def test_hashes_and_figures(built):
    root, _, text = built
    meta = root / "metadata"
    for name in ("folds.csv", "labels.csv"):
        digest = hashlib.sha256((meta / name).read_bytes()).hexdigest()
        assert f"{name}: {digest}" in _section(text, 6)
    figs = _section(text, 5)
    for png in (root / "results").glob("final/*.png"):
        assert f"results/final/{png.name}" in figs
    assert "results/questionnaire/score_distribution.png" in figs


def test_build_never_recomputes_metrics():
    source = report.__loader__.get_source(report.__name__)
    assert "sklearn" not in source and "numpy" not in source
    for banned in ("f1_score(", "bootstrap_ci(", "paired_diff_ci(", "metrics("):
        assert banned not in source


def test_missing_sources_are_reported_not_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(config.paths, "data_root", tmp_path)
    monkeypatch.setattr(config.paths, "output_root", tmp_path)
    monkeypatch.setattr(config.labeling, "output_csv", None)
    text = report.build().read_text(encoding="utf-8")
    assert "not found" in text and "model_comparison.csv" in text
    assert "folds.csv: not found" in text
