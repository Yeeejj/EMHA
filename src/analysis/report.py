"""
RESULTS.txt — one plain-text report of every result — Stage H5.

Assembled from existing files only; no metric is recomputed. Text sources are
quoted line for line, CSV/JSON values are only formatted. The only things
computed here are the SHA-256 hashes of folds.csv and labels.csv and the git
lookups.

Sources (a missing source is reported as missing, never filled in):

    results/questionnaire/report.txt         N, Cronbach's alpha, balance, band
    DOCS/decisions/analysis_design.txt       design, middle-band fraction
    results/final/model_comparison.txt/.csv  primary + full tables, paired diffs
    results/final/permutation_*.json         permutation p-values
    results/final/secondary.txt              secondary + exploratory analyses
    results/final/**.png, results/questionnaire/*.png   figure list
    metadata/folds.csv, labels.csv           SHA-256 (and whether the protocol
                                             text records the same hash)
    git: tag protocol-frozen, HEAD commit, uncommitted changes
    EVALUATION_PROTOCOL.md section 9         Deviations, verbatim

    python -m src.analysis.report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data.dataloader import labels_csv_path
from src.training.evaluator import output_dir
from src.training.run_ensemble import FEATURE_SET as ENSEMBLE_FS
from src.training.run_ensemble import MODEL as ENSEMBLE
from src.training.splits import folds_path
from src.utils.config import config
from src.utils.protocol_guard import PROTOCOL_TAG, is_synthetic_root

_REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = _REPO_ROOT / "EVALUATION_PROTOCOL.md"
DESIGN = _REPO_ROOT / "DOCS" / "decisions" / "analysis_design.txt"
DEVIATIONS_HEADING = "## 9. Deviations"
# results/questionnaire/report.txt is quoted whole except this per-item block.
ITEM_BLOCK = "Alpha if item deleted:"
DESIGN_KEYS = ("Date", "Design", "Middle-band fraction")
RULE = "=" * 78


# ─── sources ──────────────────────────────────────────────────────────────────


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _quote(text: str | None, path: Path, keys=None, skip_block=None) -> list:
    """Lines of a text source, indented; keys keeps only lines starting with
    them; skip_block drops the block from that line to the next blank line."""
    if text is None:
        return [f"  (not found: {path})"]
    lines, skipping = [], False
    for ln in text.splitlines():
        skipping = (skipping or ln == skip_block) and bool(ln.strip())
        if not skipping and (keys is None or ln.startswith(keys)):
            lines.append(ln)
    return [f"  {ln}" if ln else "" for ln in lines]


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def deviations(protocol_text: str | None) -> list:
    """Section 9 of EVALUATION_PROTOCOL.md, verbatim."""
    if protocol_text is None:
        return [f"  (not found: {PROTOCOL})"]
    lines = protocol_text.splitlines()
    if DEVIATIONS_HEADING not in lines:
        return [f"  ('{DEVIATIONS_HEADING}' not found in {PROTOCOL.name})"]
    start = lines.index(DEVIATIONS_HEADING) + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines)
    )
    body = list(lines[start:end])
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return [f"  {ln}" if ln else "" for ln in body]


def permutation_results(final: Path) -> dict:
    """(analysis, model, feature_set) -> permutation JSON."""
    out = {}
    for path in sorted(final.glob("permutation_*.json")):
        r = json.loads(path.read_text())
        out[(r["analysis"], r["model"], r["feature_set"])] = r
    return out


# ─── tables ───────────────────────────────────────────────────────────────────


def _ci(row, m: str) -> str:
    return f"{row[m]:.3f} [{row[f'{m}_ci_low']:.3f}, {row[f'{m}_ci_high']:.3f}]"


def results_table(comp: pd.DataFrame, analysis: str, perms: dict) -> list:
    part = comp[comp["analysis"] == analysis]
    if part.empty:
        return [f"  no {analysis} results in model_comparison.csv"]
    lines = [
        "  model/feature_set              n   macro-F1 [CI]          "
        "accuracy [CI]          majority  balanced acc. [CI]     "
        "ROC-AUC [CI]           permutation"
    ]
    for _, r in part.iterrows():
        key = (analysis, r["model"], r["feature_set"])
        perm = perms.get(key)
        p = f"p = {perm['p_value']:.4f} (N = {perm['n']})" if perm else "not run"
        name = f"{r['model']}/{r['feature_set']}"
        if (r["model"], r["feature_set"]) == (ENSEMBLE, ENSEMBLE_FS):
            name += " [PRIMARY CLAIM]" if analysis == "primary" else ""
        elif str(r["exploratory"]) == "True":
            name += " [exploratory]"
        lines.append(
            f"  {name:28s} {int(r['n_participants']):>4d}  {_ci(r, 'macro_f1')}  "
            f"{_ci(r, 'accuracy')}  {r['majority_baseline']:.3f}     "
            f"{_ci(r, 'balanced_accuracy')}  {_ci(r, 'roc_auc')}  {p}"
        )
    return lines


def paired_table(comp: pd.DataFrame, analysis: str) -> list:
    part = comp[
        (comp["analysis"] == analysis) & comp["delta_macro_f1_vs_reference"].notna()
    ]
    if part.empty:
        return [f"  no paired comparisons for {analysis}"]
    lines = []
    for _, r in part.iterrows():
        lines.append(
            f"  {r['model'] + '/' + r['feature_set']:28s} - {r['reference']:20s} "
            f"{r['delta_macro_f1_vs_reference']:+.3f} "
            f"[{r['delta_macro_f1_ci_low']:+.3f}, {r['delta_macro_f1_ci_high']:+.3f}]"
        )
    return lines


def figures(results: Path) -> list:
    """Figure paths relative to the output root; per-crop overlays summarized."""
    root = results.parent
    pngs = sorted(results.glob("final/**/*.png")) + sorted(
        results.glob("questionnaire/*.png")
    )
    grids = [p for p in pngs if p.parent.parent.name != config.gradcam.output_subdir]
    overlays = {}
    for p in pngs:
        if p.parent.parent.name == config.gradcam.output_subdir:
            overlays.setdefault(p.parent, []).append(p)
    lines = [f"  {p.relative_to(root).as_posix()}" for p in grids]
    for folder, items in sorted(overlays.items()):
        lines.append(
            f"  {folder.relative_to(root).as_posix()}/  "
            f"({len(items)} per-crop Grad-CAM overlays)"
        )
    return lines or ["  no figures found"]


# ─── build ────────────────────────────────────────────────────────────────────


def build() -> Path:
    """Write results/final/RESULTS.txt from the existing result files."""
    results = config.paths.results_dir
    final = output_dir()
    comp_path = final / "model_comparison.csv"
    comp = pd.read_csv(comp_path) if comp_path.is_file() else None
    perms = permutation_results(final)
    protocol_text = _read(PROTOCOL)

    tag_commit = _git("rev-list", "-n", "1", PROTOCOL_TAG)
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    synthetic = is_synthetic_root(config.paths.data_root)

    out = [
        RULE,
        "EMHA RESULTS (assembled by src.analysis.report from existing files; "
        "no metric recomputed)",
        RULE,
    ]
    if synthetic:
        out.append("SYNTHETIC DATA ROOT: these are pipeline checks, NOT results.")
    if not tag_commit:
        out.append(
            f"WARNING: git tag '{PROTOCOL_TAG}' not found; the protocol is not "
            "frozen and no number below may be reported as a result."
        )
    out.append("")

    q_path = results / "questionnaire" / "report.txt"
    rel = q_path.relative_to(results.parent).as_posix()
    out += ["1. DATASET SUMMARY", f"  from {rel} (alpha-if-item-deleted omitted):"]
    out += _quote(_read(q_path), q_path, skip_block=ITEM_BLOCK)
    out += [f"  from {DESIGN.relative_to(_REPO_ROOT).as_posix()}:"]
    out += _quote(_read(DESIGN), DESIGN, DESIGN_KEYS)
    out.append("")

    out += ["2. PRIMARY RESULTS (participant level, pooled out-of-fold)"]
    mc_txt = final / "model_comparison.txt"
    out += _quote(_read(mc_txt), mc_txt)[:7] if mc_txt.is_file() else []
    if comp is None:
        out += [f"  (not found: {comp_path})"]
    else:
        out += [""] + results_table(comp, "primary", perms)
        out += ["", "  Full sample (secondary):"]
        out += results_table(comp, "full", perms)
    out.append("")

    out += ["3. PAIRED COMPARISONS (macro-F1 difference vs reference [CI])"]
    if comp is not None:
        for analysis in config.evaluation.analyses:
            out += [f"  -- {analysis} --"] + paired_table(comp, analysis)
    out.append("")

    sec = final / "secondary.txt"
    out += ["4. SECONDARY AND EXPLORATORY RESULTS (verbatim secondary.txt)"]
    out += _quote(_read(sec), sec)
    out.append("")

    out += ["5. FIGURES"] + figures(results) + [""]

    out += ["6. DATA FILES (SHA-256)"]
    for name, path in (
        ("folds.csv", folds_path(config)),
        ("labels.csv", Path(labels_csv_path(config))),
    ):
        digest = sha256(path)
        if digest is None:
            out.append(f"  {name}: not found ({path})")
            continue
        recorded = protocol_text is not None and digest in protocol_text
        out.append(
            f"  {name}: {digest}  "
            f"({'matches' if recorded else 'NOT recorded in'} {PROTOCOL.name})"
        )
    out.append("")

    out += [
        "7. PROVENANCE",
        f"  protocol tag {PROTOCOL_TAG}: {tag_commit or 'NOT FOUND'}",
        f"  git HEAD: {head or 'unknown'}"
        + ("  (uncommitted changes present)" if dirty else ""),
        f"  data root: {config.paths.data_root}"
        + ("  (synthetic)" if synthetic else ""),
        "",
        "8. DEVIATIONS FROM EVALUATION_PROTOCOL.md (section 9, verbatim)",
    ]
    out += deviations(protocol_text)
    out.append("")

    final.mkdir(parents=True, exist_ok=True)
    path = final / "RESULTS.txt"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Written: {path}")
    return path


def main() -> int:
    argparse.ArgumentParser(description="Assemble RESULTS.txt (Stage H5).").parse_args()
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
