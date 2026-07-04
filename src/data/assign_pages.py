"""
Phase 2 — Assign pages: map each participant's 4 PNGs to page roles.

Reads DATA/METADATA/participants.csv. For every participant the 4 PNG files
under DATA/STAGING/{folder_name}/PNG/ are sorted alphabetically, which equals
chronological page order per the EMHA timestamp naming. Sorted position maps
to a page role:

    index 0 -> page_cover
    index 1 -> page_selfreport
    index 2 -> page_drawing
    index 3 -> page_writing

PDFs are matched by sorted position, not by filename: the scanner stamps PNG
and PDF captures with separate timestamps, so the stems never match. Both
folders hold their pages in chronological (alphabetical) order, so the PDF at
sorted index i is the archival counterpart of the PNG at index i. If no PDF
exists at that position, pdf_path is left blank and a warning is logged. Each
participant must produce exactly 4 rows — any mismatch is flagged. The
manifest is written to DATA/METADATA/page_manifest.csv.

Run from the project root:

    python -m src.data.assign_pages
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from src.utils.config import config

PAGE_ROLES = ["page_cover", "page_selfreport", "page_drawing", "page_writing"]
EXPECTED_PNG_COUNT = len(PAGE_ROLES)


def read_participants(participants_path: Path) -> list[dict]:
    """Return participant rows (folder_name, participant_id, ...)."""
    with participants_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_rows_for_participant(participant: dict, staging_dir: Path) -> list[dict]:
    """Build manifest rows for one participant.

    Returns a list of row dicts. The list has 4 entries when the PNG folder
    holds exactly 4 files; otherwise it has as many rows as there are PNGs
    (possibly 0), and the caller flags the count mismatch.
    """
    participant_id = participant["participant_id"]
    folder_name = participant["folder_name"]

    png_dir = staging_dir / folder_name / "PNG"
    pdf_dir = staging_dir / folder_name / "PDF"

    if not png_dir.is_dir():
        print(
            f"WARNING: {participant_id}: PNG folder missing ({png_dir}); "
            "0 pages assigned."
        )
        return []

    png_files = sorted(p for p in png_dir.iterdir() if p.is_file())

    # PDFs are paired by sorted position (chronological order), since PNG and
    # PDF stems carry independent scanner timestamps and never match.
    if pdf_dir.is_dir():
        pdf_files = sorted(p for p in pdf_dir.iterdir() if p.is_file())
    else:
        pdf_files = []
        print(
            f"WARNING: {participant_id}: PDF folder missing ({pdf_dir}); "
            "pdf_path left blank for all pages."
        )

    rows: list[dict] = []
    for index, png_path in enumerate(png_files):
        # Roles only exist for the first 4 positions; extras get a blank
        # role so the mismatch is visible rather than crashing.
        role = PAGE_ROLES[index] if index < len(PAGE_ROLES) else ""

        if index < len(pdf_files):
            pdf_str = str(pdf_files[index])
        else:
            pdf_str = ""
            print(
                f"WARNING: {participant_id}: no PDF at position {index} "
                f"for {png_path.name} ({len(pdf_files)} PDFs present)."
            )

        rows.append(
            {
                "participant_id": participant_id,
                "page_role": role,
                "png_path": str(png_path),
                "pdf_path": pdf_str,
            }
        )

    return rows


def assign_pages(
    participants_path: Path, staging_dir: Path, manifest_path: Path
) -> None:
    """Build the page-role manifest for all participants."""
    print("=" * 60)
    print("Phase 2 - ASSIGN pages")
    print("=" * 60)

    if not participants_path.is_file():
        print(f"ERROR: participants registry not found: {participants_path}")
        sys.exit(1)

    participants = read_participants(participants_path)
    print(f"Participants registry : {participants_path}")
    print(f"Participants          : {len(participants)}")

    all_rows: list[dict] = []
    flagged: list[tuple[str, int]] = []

    for participant in participants:
        rows = build_rows_for_participant(participant, staging_dir)
        if len(rows) != EXPECTED_PNG_COUNT:
            flagged.append((participant["participant_id"], len(rows)))
        all_rows.extend(rows)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["participant_id", "page_role", "png_path", "pdf_path"]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    missing_pdfs = sum(1 for r in all_rows if not r["pdf_path"])

    print(f"Manifest written      : {manifest_path}")
    print(f"Total rows            : {len(all_rows)}")
    print(f"Rows missing PDF path : {missing_pdfs}")
    print(f"Participants flagged  : {len(flagged)}")

    if flagged:
        print("\nFlagged participants (expected 4 rows):")
        for participant_id, count in flagged:
            print(f"  {participant_id}: {count} rows")


def main() -> int:
    metadata_dir = Path(config.data.metadata_dir)
    participants_path = metadata_dir / "participants.csv"
    manifest_path = metadata_dir / "page_manifest.csv"
    staging_dir = Path(config.data.staging_dir)
    assign_pages(participants_path, staging_dir, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
