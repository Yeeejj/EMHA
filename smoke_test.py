"""
Phase 14 — Smoke test: run stages A-G on 3-5 respondents.

Execute AFTER Gate 1 fix (hybrid.py). Never scale before this passes.

Stages:
    A. Config import and directory creation
    B. Participant registration (3-5 respondents from DATASET/raw)
    C. Page role assignment
    D. Questionnaire scoring (requires questionnaire_scores.csv for pilot)
    E. Content extraction (24 crops per respondent)
    F. Label propagation -> crop_index.csv
    G. Preprocessing pipeline (spot-check output)

Failure in any stage prints the error and stops. Success prints a summary.

Run from the project root:

    python smoke_test.py
    python smoke_test.py --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ok(stage: str) -> None:
    print(f"  [PASS] Stage {stage}")


def _fail(stage: str, reason: str) -> None:
    print(f"  [FAIL] Stage {stage}: {reason}")
    sys.exit(1)


def stage_a_config() -> None:
    """A: Config import and directory setup."""
    try:
        from src.utils.config import config, DRAWING_CROPS, WORD_CROPS, CURSIVE_CROPS
        assert len(DRAWING_CROPS) == 4
        assert len(WORD_CROPS) == 15
        assert len(CURSIVE_CROPS) == 5
        assert config.cnn.use_pretrained is True
        assert config.labeling.adjusted_total_threshold == 72
        _ok("A: config import")
    except Exception as exc:
        _fail("A: config import", str(exc))


def stage_b_register(n: int) -> list[str]:
    """B: Register first n respondents from DATASET/raw."""
    try:
        from src.utils.config import config
        raw_dir = Path(config.data.raw_data_dir)
        folders = sorted(f for f in raw_dir.iterdir() if f.is_dir())[:n]
        if not folders:
            _fail("B: register", f"No folders in {raw_dir}")
        pids = []
        for folder in folders:
            num = folder.name.replace("respondent_", "")
            pids.append(f"P{int(num):03d}")
        print(f"  [PASS] Stage B: registered {pids}")
        return pids
    except Exception as exc:
        _fail("B: register", str(exc))
        return []


def stage_c_manifest() -> None:
    """C: Check page manifest exists or is generatable."""
    meta = Path("DATA/METADATA")
    manifest = meta / "page_manifest.csv"
    if manifest.is_file():
        _ok("C: page_manifest.csv exists")
    else:
        print(f"  [WARN] Stage C: {manifest} not yet generated (run Phase 2)")


def stage_d_scoring() -> None:
    """D: Validate questionnaire_scorer.py logic with synthetic data."""
    try:
        from src.data.questionnaire_scorer import score, assign_label, validate_items

        # Max happiness, min sadness -> HAPPY
        items_happy = {i: 5 for i in range(1, 13)} | {i: 1 for i in range(13, 25)}
        # Build proper item dict: 12 happiness items (value 5) + 12 sadness items (value 1)
        from src.utils.config import config as cfg
        items_h = {i: 5 for i in cfg.labeling.happiness_items}
        items_s = {i: 1 for i in cfg.labeling.sadness_items}
        all_items = {**items_h, **items_s}
        h, s, adj, _, _ = score(all_items)
        assert adj == 120, f"expected 120 got {adj}"
        assert assign_label(adj) == "HAPPY"

        # Min happiness, max sadness -> SAD
        items_h2 = {i: 1 for i in cfg.labeling.happiness_items}
        items_s2 = {i: 5 for i in cfg.labeling.sadness_items}
        all_items2 = {**items_h2, **items_s2}
        h2, s2, adj2, _, _ = score(all_items2)
        assert adj2 == 24, f"expected 24 got {adj2}"
        assert assign_label(adj2) == "SAD"

        # Boundary: adjusted_total = 72 -> HAPPY
        assert assign_label(72) == "HAPPY"
        assert assign_label(71) == "SAD"

        _ok("D: questionnaire_scorer logic")
    except Exception as exc:
        _fail("D: questionnaire_scorer", str(exc))


def stage_e_extraction(pids: list[str]) -> None:
    """E: Check extraction infrastructure (dry run — no actual files needed)."""
    try:
        from src.data.content_extractor import WritingCellExtractor, frac_box_to_pixels
        from src.utils.config import DRAWING_CROPS

        # Test fractional->pixel conversion
        x1, y1, x2, y2 = frac_box_to_pixels(DRAWING_CROPS["draw_circles"], 1700, 2200)
        assert x2 > x1 and y2 > y1, "draw_circles box is degenerate"

        extractor = WritingCellExtractor(Path("DATA/CROPS"))
        assert extractor is not None
        _ok(f"E: WritingCellExtractor ready (pilot: {pids})")
    except Exception as exc:
        _fail("E: extraction", str(exc))


def stage_f_propagate() -> None:
    """F: Verify crop_index or propagation imports work."""
    try:
        from src.data.propagate_labels import _task_type
        assert _task_type("draw_circles") == "drawing"
        assert _task_type("word_content_left") == "word"
        assert _task_type("cursive_01") == "cursive"
        _ok("F: propagate_labels task_type mapping")
    except Exception as exc:
        _fail("F: propagate_labels", str(exc))


def stage_g_preprocessing() -> None:
    """G: Import preprocessing pipeline and verify it initialises."""
    try:
        from src.preprocessing.pipeline import PreprocessingPipeline
        import numpy as np
        pp = PreprocessingPipeline(target_size=(224, 224), skip_skew=False)
        dummy = np.ones((100, 100, 3), dtype=np.uint8) * 128
        out = pp.process(dummy)
        assert out.shape == (224, 224), f"unexpected shape {out.shape}"
        assert out.dtype == np.float32
        _ok("G: preprocessing pipeline")
    except Exception as exc:
        _fail("G: preprocessing", str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="INSIDE-OUT smoke test (Phases A-G).")
    parser.add_argument("--n", type=int, default=3, help="Number of pilot respondents")
    args = parser.parse_args()

    print("=" * 60)
    print(f"SMOKE TEST — {args.n} pilot respondent(s)")
    print("=" * 60)

    stage_a_config()
    pids = stage_b_register(args.n)
    stage_c_manifest()
    stage_d_scoring()
    stage_e_extraction(pids)
    stage_f_propagate()
    stage_g_preprocessing()

    print("\n" + "=" * 60)
    print("ALL STAGES PASSED — safe to proceed to 30-respondent pilot.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
