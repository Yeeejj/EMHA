"""
Crop manifest, verifier, and QC contact sheets — Stage C.

Indexes all 24 crops per participant (already produced by the protected
src/cropping/p3/p4 scripts) into the manifest the rest of the pipeline
reads. Opens every crop read-only; never writes, moves, or renames
anything under raw-3Page or raw-4Page (CLAUDE.md Non-Negotiable 2) — the
only outputs are DATASET/metadata/crop_manifest.csv, qc_log.csv, and
results/qc/ contact sheets.

Run from the project root:

    python -m src.data.crop_manifest
    python -m src.data.crop_manifest --batch 001-050
    python -m src.data.crop_manifest --apply-qc-status   # after reviewing the diff
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import config

CODE_PATTERN = re.compile(r"(?<!\d)(\d{3})(?!\d)")

DRAWING_CELLS = ("D1", "D2", "D3", "D4")
WORD_CELLS = tuple(f"W{n}_{s}" for n in range(1, 6) for s in ("LH", "RH", "UC"))
CURSIVE_CELLS = tuple(f"CS{k}" for k in range(1, 6))
ALL_CELLS = DRAWING_CELLS + WORD_CELLS + CURSIVE_CELLS

DRAWING_NAMES = {"D1": "circles", "D2": "dots", "D3": "person", "D4": "house"}
WORD_NAMES = {
    1: "content",
    2: "melancholic",
    3: "optimistic",
    4: "disconnected",
    5: "vibrant",
}

MANIFEST_FIELDS = [
    "participant_id",
    "cell",
    "task_family",
    "task",
    "item",
    "style",
    "path",
    "width_px",
    "height_px",
    "mode",
    "ink_ratio",
    "mean_ink_darkness",
    "edge_ink",
    "baseline_angle_deg",
    "sha256",
]


def _extract_code(stem: str) -> str:
    match = CODE_PATTERN.search(stem)
    return match.group(1) if match else ""


def _derive_fields(cell: str) -> tuple:
    """Return (task_family, task, item, style) for a cell code."""
    if cell.startswith("D"):
        k = int(cell[1:])
        return "drawing", DRAWING_NAMES[cell], str(k), ""
    if cell.startswith("CS"):
        k = int(cell[2:])
        return "cursive", "cursive", str(k), ""
    n_str, style = cell.split("_")
    n = int(n_str[1:])
    return "word", WORD_NAMES[n], str(n), style


def _expected_filename(participant_id: str, cell: str) -> str:
    if cell.startswith("D"):
        return f"EMHA-P3_DrawingExercise_{participant_id}_{cell}.png"
    return f"EMHA-P4_WritingExercise_{participant_id}_{cell}.png"


def _touches_two_or_more_borders(ink_mask: np.ndarray) -> bool:
    borders = [
        ink_mask[0, :].any(),
        ink_mask[-1, :].any(),
        ink_mask[:, 0].any(),
        ink_mask[:, -1].any(),
    ]
    return sum(bool(b) for b in borders) >= 2


def measure_baseline_angle(img: np.ndarray) -> float:
    """Baseline angle in degrees via a robust line fit.

    Finds the lowest ink pixel in each of 20 column bands, then fits a
    line through those points with the Theil-Sen estimator (median of all
    pairwise slopes) -- robust to the occasional stray ink pixel or gap.
    img is never rotated or modified. Positive angle means the baseline
    descends left-to-right (y increases with x, image coordinates).
    Returns 0.0 if fewer than 2 bands contain ink.
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    height, width = arr.shape
    ink_mask = arr < config.crop.ink_threshold

    n_bands = 20
    band_width = max(1, width // n_bands)
    xs = []
    ys = []
    for start in range(0, width, band_width):
        end = min(start + band_width, width)
        band = ink_mask[:, start:end]
        rows_with_ink = np.where(band.any(axis=1))[0]
        if rows_with_ink.size == 0:
            continue
        xs.append((start + end) / 2.0)
        ys.append(float(rows_with_ink.max()))

    if len(xs) < 2:
        return 0.0

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    slopes = []
    n = len(xs_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs_arr[j] - xs_arr[i]
            if dx != 0:
                slopes.append((ys_arr[j] - ys_arr[i]) / dx)

    if not slopes:
        return 0.0
    slope = float(np.median(slopes))
    return float(np.degrees(np.arctan(slope)))


def _describe_crop(path: Path, participant_id: str, cell: str) -> dict:
    """Open path read-only and describe it. Never writes to path."""
    task_family, task, item, style = _derive_fields(cell)

    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            gray = np.array(img.convert("L"))  # in-memory only, never saved
    except Exception:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "participant_id": participant_id,
            "cell": cell,
            "task_family": task_family,
            "task": task,
            "item": item,
            "style": style,
            "path": str(path),
            "width_px": None,
            "height_px": None,
            "mode": "UNREADABLE",
            "ink_ratio": None,
            "mean_ink_darkness": None,
            "edge_ink": None,
            "baseline_angle_deg": None,
            "sha256": sha256,
        }

    ink_mask = gray < config.crop.ink_threshold
    ink_ratio = float(ink_mask.mean())
    mean_ink_darkness = float((255 - gray[ink_mask]).mean()) if ink_mask.any() else 0.0
    edge_ink = _touches_two_or_more_borders(ink_mask)

    baseline_angle_deg = None
    if task_family in ("word", "cursive"):
        baseline_angle_deg = round(measure_baseline_angle(gray), 4)

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "participant_id": participant_id,
        "cell": cell,
        "task_family": task_family,
        "task": task,
        "item": item,
        "style": style,
        "path": str(path),
        "width_px": width,
        "height_px": height,
        "mode": mode,
        "ink_ratio": round(ink_ratio, 6),
        "mean_ink_darkness": round(mean_ink_darkness, 4),
        "edge_ink": edge_ink,
        "baseline_angle_deg": baseline_angle_deg,
        "sha256": sha256,
    }


def _scan_cells(root: Path, cells: tuple) -> list:
    rows = []
    for cell in cells:
        cell_dir = Path(root) / cell
        if not cell_dir.is_dir():
            continue
        for path in sorted(p for p in cell_dir.iterdir() if p.is_file()):
            participant_id = _extract_code(path.stem)
            rows.append(_describe_crop(path, participant_id, cell))
    return rows


def build_manifest() -> pd.DataFrame:
    """Index all crops under raw-3Page/raw-4Page into one DataFrame. Read-only."""
    rows = _scan_cells(config.paths.raw3_dir, DRAWING_CELLS)
    rows += _scan_cells(config.paths.raw4_dir, WORD_CELLS + CURSIVE_CELLS)
    return pd.DataFrame(rows, columns=MANIFEST_FIELDS)


def _row_problem_ids(df: pd.DataFrame) -> set:
    """participant_ids with at least one row-level problem (name/size/unreadable)."""
    cfg = config.crop
    bad_ids = set()
    for _, row in df.iterrows():
        if row["mode"] == "UNREADABLE":
            bad_ids.add(row["participant_id"])
            continue
        if Path(row["path"]).name != _expected_filename(
            row["participant_id"], row["cell"]
        ):
            bad_ids.add(row["participant_id"])
        expected_size = cfg.expected_size_px.get(row["cell"])
        if expected_size and (row["width_px"], row["height_px"]) != expected_size:
            bad_ids.add(row["participant_id"])
    return bad_ids


def verify(df: pd.DataFrame) -> list:
    """Return human-readable problems with df; empty list means clean.

    Checks: missing or extra cells per participant, bad file names, size
    mismatches, unreadable files, and participant_ids with crops but
    missing from labels.csv. Non-grayscale mode and edge_ink are
    informational only (every real crop is RGB; converting to grayscale
    happens downstream) and are NOT reported as problems here -- see the
    manifest's mode/edge_ink columns instead.
    """
    cfg = config.crop
    errors = []
    all_cells = set(cfg.expected_size_px.keys())

    if not df.empty:
        for pid, group in df.groupby("participant_id"):
            if pid == "":
                continue
            cells_present = set(group["cell"])
            for cell in sorted(all_cells - cells_present):
                errors.append(f"participant {pid!r}: missing cell {cell!r}")
            for cell in sorted(cells_present - all_cells):
                errors.append(f"participant {pid!r}: unexpected cell {cell!r}")

    for _, row in df.iterrows():
        if row["mode"] == "UNREADABLE":
            errors.append(f"unreadable file: {row['path']!r}")
            continue

        expected_name = _expected_filename(row["participant_id"], row["cell"])
        if Path(row["path"]).name != expected_name:
            errors.append(
                f"bad file name: {row['path']!r} (expected {expected_name!r})"
            )

        expected_size = cfg.expected_size_px.get(row["cell"])
        if expected_size and (row["width_px"], row["height_px"]) != expected_size:
            got = (row["width_px"], row["height_px"])
            errors.append(
                f"size mismatch for {row['path']!r}: "
                f"got {got}, expected {expected_size}"
            )

    labels_path = config.paths.metadata_dir / "labels.csv"
    if labels_path.is_file() and not df.empty:
        labels = pd.read_csv(labels_path, dtype={"participant_id": str})
        known_ids = set(labels["participant_id"])
        manifest_ids = set(df["participant_id"]) - {""}
        for pid in sorted(manifest_ids - known_ids):
            errors.append(
                f"participant_id {pid!r} has crops but is missing from labels.csv"
            )

    return errors


def contact_sheets(df: pd.DataFrame, n: int = 20) -> list:
    """Save one QC contact sheet per task family to results/qc/. Read-only on crops."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    out_dir = config.paths.results_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.training.seed)

    written = []
    families = (
        ("drawing", DRAWING_CELLS),
        ("word", WORD_CELLS),
        ("cursive", CURSIVE_CELLS),
    )

    for family, cells in families:
        family_df = df[df["task_family"] == family]
        participants = sorted(family_df["participant_id"].unique())
        if not participants:
            continue
        sample = sorted(rng.sample(participants, k=min(n, len(participants))))

        n_rows, n_cols = len(sample), len(cells)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows), squeeze=False
        )

        for i, pid in enumerate(sample):
            for j, cell in enumerate(cells):
                ax = axes[i][j]
                ax.axis("off")
                if i == 0:
                    ax.set_title(cell, fontsize=7)
                if j == 0:
                    ax.text(
                        -0.1,
                        0.5,
                        pid,
                        transform=ax.transAxes,
                        fontsize=7,
                        ha="right",
                        va="center",
                    )
                match = family_df[
                    (family_df["participant_id"] == pid) & (family_df["cell"] == cell)
                ]
                if match.empty or match.iloc[0]["mode"] == "UNREADABLE":
                    continue
                try:
                    with Image.open(match.iloc[0]["path"]) as img:
                        ax.imshow(img, cmap="gray")
                except Exception:
                    continue

        fig.suptitle(f"QC contact sheet — {family}", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"contact_sheet_{family}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)

    return written


def _filter_batch(df: pd.DataFrame, batch: str) -> pd.DataFrame:
    start_s, end_s = batch.split("-")
    start, end = int(start_s), int(end_s)
    numeric_id = pd.to_numeric(df["participant_id"], errors="coerce")
    return df[numeric_id.between(start, end)].reset_index(drop=True)


def _propose_qc_status(df: pd.DataFrame) -> set:
    """participant_ids eligible for qc_passed: all 24 cells, no row-level problem."""
    all_cells = set(config.crop.expected_size_px.keys())
    bad_ids = _row_problem_ids(df)
    eligible = set()
    for pid, group in df.groupby("participant_id"):
        if pid == "" or pid in bad_ids:
            continue
        if set(group["cell"]) >= all_cells:
            eligible.add(pid)
    return eligible


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index, verify, and QC crops under raw-3Page/raw-4Page."
    )
    parser.add_argument("--batch", help="Participant code range, e.g. 001-050")
    parser.add_argument(
        "--apply-qc-status",
        action="store_true",
        help="Write qc_passed to participants.csv (default: show the diff only).",
    )
    args = parser.parse_args()

    df = build_manifest()
    if args.batch:
        df = _filter_batch(df, args.batch)

    print(f"Crops scanned : {len(df)}")

    errors = verify(df)

    manifest_path = config.paths.metadata_dir / "crop_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Manifest written: {manifest_path}")

    qc_log = df.copy()
    qc_log["dropped"] = False
    qc_log["notes"] = ""
    qc_log_path = config.paths.metadata_dir / "qc_log.csv"
    qc_log.to_csv(qc_log_path, index=False)
    print(f"QC log written  : {qc_log_path}")

    print(f"Problems        : {len(errors)}")
    for problem in errors:
        print(f"  - {problem}")

    sheets = contact_sheets(df)
    for sheet in sheets:
        print(f"Contact sheet   : {sheet}")

    participants_path = config.paths.metadata_dir / "participants.csv"
    if not participants_path.is_file():
        print(f"\nNOTE: {participants_path} not found; skipping qc_status proposal.")
    elif not df.empty:
        from src.data.collector import ParticipantRegistry

        participants = ParticipantRegistry.load(participants_path)
        eligible = _propose_qc_status(df)
        to_update = participants[
            participants["participant_id"].isin(eligible)
            & (participants["status"] != "qc_passed")
        ]

        if len(to_update):
            print("\nProposed participants.csv status changes:")
            for _, row in to_update.iterrows():
                print(f"  {row['participant_id']}: {row['status']} -> qc_passed")

            if args.apply_qc_status:
                participants.loc[
                    participants["participant_id"].isin(eligible), "status"
                ] = "qc_passed"
                participants.to_csv(participants_path, index=False)
                print(f"\nparticipants.csv updated: {len(to_update)} participant(s).")
            else:
                print("\nDry run: rerun with --apply-qc-status to write this.")
        else:
            print("\nNo participants.csv status changes needed.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
