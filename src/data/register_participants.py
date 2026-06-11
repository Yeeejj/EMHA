"""
Phase 1 — Register participants: assign barcode IDs to validated folders.

Reads DATA/METADATA/validation_report.csv and processes only rows whose
status is PASS. Each passing folder is assigned a participant ID via the
barcode rule (CLAUDE.md): strip the ``respondent_`` prefix, zero-pad the
number to 3 digits, and prepend ``P`` (respondent_1 -> P001,
respondent_42 -> P042).

Every generated ID is validated against ^P\\d{3}$ before writing, and
duplicate IDs are skipped with a warning so the registry never contains
duplicates. The result is written to DATA/METADATA/participants.csv.

Run from the project root:

    python -m src.data.register_participants
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import date
from pathlib import Path

from src.utils.config import config

PARTICIPANT_ID_PATTERN = re.compile(r"^P\d{3}$")
FOLDER_NAME_PATTERN = re.compile(r"^respondent_(\d+)$")


def make_participant_id(folder_name: str) -> str | None:
    """Convert a respondent folder name to a participant ID.

    Returns the ID (e.g. "P001") or None if the folder name does not match
    the expected ``respondent_<number>`` pattern.
    """
    match = FOLDER_NAME_PATTERN.match(folder_name)
    if not match:
        return None
    number = int(match.group(1))
    return f"P{number:03d}"


def read_passing_folders(report_path: Path) -> list[str]:
    """Return folder names from the validation report with status PASS."""
    with report_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [row["folder_name"] for row in reader if row.get("status") == "PASS"]


def register_participants(report_path: Path, output_path: Path) -> None:
    """Build the participant registry from passing validation rows."""
    print("=" * 60)
    print("Phase 1 - REGISTER participants")
    print("=" * 60)

    if not report_path.is_file():
        print(f"ERROR: validation report not found: {report_path}")
        sys.exit(1)

    intake_date = date.today().isoformat()
    passing = read_passing_folders(report_path)
    print(f"Validation report : {report_path}")
    print(f"Passing folders    : {len(passing)}")

    rows: list[dict] = []
    seen_ids: set[str] = set()
    registered = 0
    skipped_duplicates = 0

    for folder_name in passing:
        participant_id = make_participant_id(folder_name)

        if participant_id is None or not PARTICIPANT_ID_PATTERN.match(participant_id):
            print(
                f"WARNING: '{folder_name}' produced an invalid ID "
                f"({participant_id!r}); skipping."
            )
            continue

        if participant_id in seen_ids:
            print(
                f"WARNING: duplicate participant_id '{participant_id}' "
                f"from folder '{folder_name}'; skipping."
            )
            skipped_duplicates += 1
            continue

        seen_ids.add(participant_id)
        rows.append(
            {
                "folder_name": folder_name,
                "participant_id": participant_id,
                "intake_date": intake_date,
                "status": "ACTIVE",
            }
        )
        registered += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["folder_name", "participant_id", "intake_date", "status"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Registry written   : {output_path}")
    print(f"Total registered   : {registered}")
    print(f"Skipped duplicates : {skipped_duplicates}")


def main() -> int:
    report_path = Path(config.data.metadata_dir) / "validation_report.csv"
    output_path = Path(config.data.metadata_dir) / "participants.csv"
    register_participants(report_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
