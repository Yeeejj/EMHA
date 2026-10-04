"""
Raw scan validator and checksum ledger — Stage C.

Walks raw-3Page/raw-4Page read-only, records one row per page scan and per
crop file (dimensions, DPI, mode, SHA-256), and flags problems. Never
writes, moves, or renames anything under raw-3Page or raw-4Page (CLAUDE.md
Non-Negotiable 2) — the only output is DATASET/metadata/raw_manifest.csv.

Run from the project root:

    python -m src.data.ingest
    python -m src.data.ingest --batch 001-050
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils.config import config

CODE_PATTERN = re.compile(r"(?<!\d)(\d{3})(?!\d)")

MANIFEST_FIELDS = [
    "participant_id",
    "kind",
    "cell",
    "path",
    "width_px",
    "height_px",
    "dpi",
    "mode",
    "sha256",
]


def _extract_code(stem: str) -> str:
    match = CODE_PATTERN.search(stem)
    return match.group(1) if match else ""


def _describe_file(path: Path, kind: str, cell: str) -> dict:
    """Open path read-only and describe it. Never writes to path."""
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
        dpi_info = img.info.get("dpi")
        dpi = round(dpi_info[0], 4) if dpi_info else None
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "participant_id": _extract_code(path.stem),
        "kind": kind,
        "cell": cell,
        "path": str(path),
        "width_px": width,
        "height_px": height,
        "dpi": dpi,
        "mode": mode,
        "sha256": sha256,
    }


def _scan_root(root: Path, scan_kind: str) -> list:
    rows = []
    if not root.is_dir():
        return rows

    for path in sorted(p for p in root.iterdir() if p.is_file()):
        rows.append(_describe_file(path, kind=scan_kind, cell=""))

    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        cell = subdir.name
        for path in sorted(p for p in subdir.iterdir() if p.is_file()):
            rows.append(_describe_file(path, kind="crop", cell=cell))

    return rows


def scan_raw(raw3_dir: Path, raw4_dir: Path) -> pd.DataFrame:
    """One row per page scan (P3, P4) and per crop file. Read-only."""
    rows = _scan_root(Path(raw3_dir), "p3_scan") + _scan_root(Path(raw4_dir), "p4_scan")
    return pd.DataFrame(rows, columns=MANIFEST_FIELDS)


def _expected_filename(participant_id: str, kind: str, cell: str) -> str:
    if kind == "p3_scan":
        return f"EMHA-P3_DrawingExercise_{participant_id}.png"
    if kind == "p4_scan":
        return f"EMHA-P4_WritingExercise_{participant_id}.png"
    if cell.startswith("D"):
        return f"EMHA-P3_DrawingExercise_{participant_id}_{cell}.png"
    return f"EMHA-P4_WritingExercise_{participant_id}_{cell}.png"


def validate_scans(df: pd.DataFrame) -> list:
    """Return human-readable problems with df; empty list means clean.

    Checks: missing P3 or P4 scan per participant, unexpected file names,
    bilevel (mode "1") or JPEG files, DPI not expected_dpi (scans only --
    crops carry no DPI metadata), and page sizes (scans only) more than
    size_tolerance off the median for that page kind.
    """
    cfg = config.ingest
    errors = []

    if df.empty:
        return errors

    for pid, group in df.groupby("participant_id"):
        if pid == "":
            continue
        kinds = set(group["kind"])
        if "p3_scan" not in kinds:
            errors.append(f"participant {pid!r}: missing P3 scan")
        if "p4_scan" not in kinds:
            errors.append(f"participant {pid!r}: missing P4 scan")

    for _, row in df.iterrows():
        name = Path(row["path"]).name
        expected = _expected_filename(row["participant_id"], row["kind"], row["cell"])
        if name != expected:
            errors.append(
                f"unexpected file name: {row['path']!r} (expected {expected!r})"
            )

        if row["mode"] == "1":
            errors.append(f"bilevel (mode '1') file: {row['path']!r}")
        if Path(row["path"]).suffix.lower() in (".jpg", ".jpeg"):
            errors.append(f"JPEG file: {row['path']!r}")

    scans = df[df["kind"].isin(["p3_scan", "p4_scan"])]
    for kind, group in scans.groupby("kind"):
        with_dpi = group[group["dpi"].notna()]
        for _, row in with_dpi.iterrows():
            if round(row["dpi"]) != cfg.expected_dpi:
                errors.append(
                    f"DPI {row['dpi']} != expected {cfg.expected_dpi}: {row['path']!r}"
                )

        for dim in ("width_px", "height_px"):
            median = group[dim].median()
            if median == 0:
                continue
            off = (group[dim] - median).abs() / median
            for _, row in group[off > cfg.size_tolerance].iterrows():
                errors.append(
                    f"page size off by >{cfg.size_tolerance:.0%} "
                    f"({dim}={row[dim]}, median={median:g}): {row['path']!r}"
                )

    return errors


def compare_hashes(old: pd.DataFrame, new: pd.DataFrame) -> list:
    """Return a report of every raw file whose SHA-256 changed between runs.

    Only compares rows present (by path) in both old and new; additions
    and removals are not reported here.
    """
    merged = old[["path", "sha256"]].merge(
        new[["path", "sha256"]], on="path", suffixes=("_old", "_new")
    )
    changed = merged[merged["sha256_old"] != merged["sha256_new"]]
    return [
        f"SHA-256 changed: {row['path']!r} "
        f"({row['sha256_old'][:12]}... -> {row['sha256_new'][:12]}...)"
        for _, row in changed.iterrows()
    ]


def _filter_batch(df: pd.DataFrame, batch: str) -> pd.DataFrame:
    start_s, end_s = batch.split("-")
    start, end = int(start_s), int(end_s)
    numeric_id = pd.to_numeric(df["participant_id"], errors="coerce")
    mask = numeric_id.between(start, end)
    return df[mask].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate raw-3Page/raw-4Page scans and record checksums."
    )
    parser.add_argument(
        "--batch", help="Participant code range, e.g. 001-050 (default: all)"
    )
    args = parser.parse_args()

    df = scan_raw(config.paths.raw3_dir, config.paths.raw4_dir)
    if args.batch:
        df = _filter_batch(df, args.batch)

    print(f"Files scanned : {len(df)}")

    manifest_path = config.paths.metadata_dir / "raw_manifest.csv"
    hash_changes = []
    if manifest_path.is_file():
        old_df = pd.read_csv(manifest_path, dtype={"participant_id": str})
        hash_changes = compare_hashes(old_df, df)

    problems = validate_scans(df) + hash_changes

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Manifest written: {manifest_path}")

    print(f"Problems      : {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
