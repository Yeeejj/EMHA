"""
Phase 7 — Run preprocessing: apply image pipeline to all included crops.

Reads DATA/METADATA/crop_index.csv (after Phase 6 QC) and processes every
row where excluded is False.

Pipeline order (all params from config.preprocessing):
    grayscale -> Otsu binarization -> morphological denoising
    -> skew correction (skipped for draw_* task codes)
    -> resize to 224x224 -> normalize to [0, 1]

Processed images are saved as uint8 PNG to:
    DATA/PROCESSED/{participant_id}/{participant_id}_{task_code}.png
    (mirrors DATA/CROPS structure — Section A pipeline)

Every attempt is recorded in DATA/METADATA/preprocessing_log.csv.

Run from the project root:

    python -m src.preprocessing.run_preprocessing
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import config

LOG_FIELDS = [
    "participant_id",
    "task_code",
    "input_path",
    "output_path",
    "status",
    "error_message",
]


def imread_unicode(path: Path) -> "np.ndarray | None":
    """Read an image via numpy buffer (handles non-ASCII Windows paths)."""
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Write a PNG via numpy buffer (handles non-ASCII Windows paths)."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def _should_skip_skew(task_code: str) -> bool:
    prefixes = config.preprocessing.skip_skew_prefixes
    return any(task_code.startswith(p) for p in prefixes)


def _make_pipelines() -> tuple[PreprocessingPipeline, PreprocessingPipeline]:
    """Return (pipeline_deskew, pipeline_skip) built from config."""
    cfg = config.preprocessing
    kwargs = dict(
        target_size=cfg.target_size,
        binarize_threshold=cfg.binarize_threshold,
        denoise_kernel_size=cfg.denoise_kernel_size,
    )
    return (
        PreprocessingPipeline(**kwargs, skip_skew=False),
        PreprocessingPipeline(**kwargs, skip_skew=True),
    )


def process_row(
    row: dict,
    processed_dir: Path,
    pipeline_deskew: PreprocessingPipeline,
    pipeline_skip: PreprocessingPipeline,
) -> dict:
    """Process one manifest row and return a log dict."""
    pid = row["participant_id"]
    task_code = row["task_code"]
    input_path = Path(row["crop_path"])

    out_dir = processed_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{pid}_{task_code}.png"

    log: dict = {
        "participant_id": pid,
        "task_code": task_code,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "",
        "error_message": "",
    }

    image = imread_unicode(input_path)
    if image is None:
        log["status"] = "FAIL"
        log["error_message"] = "could not read input image"
        return log

    try:
        pipeline = pipeline_skip if _should_skip_skew(task_code) else pipeline_deskew
        processed_float = pipeline.process(image)
        processed_uint8 = (processed_float * 255).astype(np.uint8)
        if not imwrite_unicode(output_path, processed_uint8):
            raise OSError("cv2.imencode / tofile failed")
    except Exception as exc:
        log["status"] = "FAIL"
        log["error_message"] = str(exc)
        return log

    log["status"] = "SUCCESS"
    return log


def run_preprocessing(
    manifest_path: Path,
    processed_dir: Path,
    log_path: Path,
) -> None:
    print("=" * 60)
    print("Phase 7 - RUN preprocessing")
    print("=" * 60)

    if not manifest_path.is_file():
        print(f"ERROR: samples manifest not found: {manifest_path}")
        sys.exit(1)

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))

    included = [r for r in all_rows if not _parse_bool(r.get("excluded", "False"))]
    skipped = len(all_rows) - len(included)

    print(f"Crop index     : {manifest_path}")
    print(f"Total rows     : {len(all_rows)}")
    print(f"Included       : {len(included)}")
    print(f"Skipped (excl) : {skipped}")

    pipeline_deskew, pipeline_skip = _make_pipelines()

    log_rows: list[dict] = []
    for row in included:
        log_rows.append(process_row(row, processed_dir, pipeline_deskew, pipeline_skip))

    succeeded = sum(1 for r in log_rows if r["status"] == "SUCCESS")
    failed = len(log_rows) - succeeded

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Preprocessing log: {log_path}")
    print(f"Processed (ok) : {succeeded}")
    print(f"Failed         : {failed}")
    print(f"Skipped (excl) : {skipped}")

    if failed:
        print("\nFailures:")
        for r in log_rows:
            if r["status"] == "FAIL":
                print(
                    f"  {r['participant_id']} {r['task_code']}: "
                    f"{r['error_message']}"
                )

    if succeeded:
        _print_example_pairs(log_rows)


def _print_example_pairs(log_rows: list[dict]) -> None:
    """Print one before/after pair for a draw_ task and a cursive_ task."""
    draw_ex = next(
        (
            r
            for r in log_rows
            if r["status"] == "SUCCESS" and r["task_code"].startswith("draw_")
        ),
        None,
    )
    cursive_ex = next(
        (
            r
            for r in log_rows
            if r["status"] == "SUCCESS" and r["task_code"].startswith("cursive_")
        ),
        None,
    )
    print("\nBefore/after pairs:")
    for label, ex in (
        ("draw_ (skip_skew=True)", draw_ex),
        ("cursive_ (skip_skew=False)", cursive_ex),
    ):
        if ex:
            print(f"\n  {label}")
            print(f"    input : {ex['input_path']}")
            print(f"    output: {ex['output_path']}")
        else:
            print(f"\n  {label}: no successful example found")


def main() -> int:
    meta = Path(config.data.metadata_dir)
    run_preprocessing(
        manifest_path=meta / "crop_index.csv",
        processed_dir=Path(config.data.processed_dir),
        log_path=meta / "preprocessing_log.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
