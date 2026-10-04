"""
Participant registry — indexes which 3-digit participant codes have scans
or crops under raw-3Page and raw-4Page.

No optical codes are generated or decoded: participant_id is read directly
from scan/crop filenames (the one 3-digit run embedded in every EMHA
filename, e.g. EMHA-P3_DrawingExercise_001.png or
EMHA-P4_WritingExercise_001_W1_LH.png -> "001"). Writes no names or other
identifying fields — only the code, presence flags, and a pipeline status.

Run from the project root:

    python -m src.data.collector --build
    python -m src.data.collector --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.utils.config import config

# Exactly 3 digits, not part of a longer run (so the single-digit crop-slot
# suffix in e.g. "_D1" or "_W1_LH" is never mistaken for the code).
CODE_PATTERN = re.compile(r"(?<!\d)(\d{3})(?!\d)")
VALID_ID_PATTERN = re.compile(r"^\d{3}$")

PARTICIPANTS_FIELDS = ["participant_id", "has_p3_scan", "has_p4_scan", "status"]


def _codes_under(root: Path) -> set:
    """Return every 3-digit code found in any filename under root."""
    if not root.is_dir():
        return set()
    codes = set()
    for path in root.rglob("*"):
        if path.is_file():
            match = CODE_PATTERN.search(path.stem)
            if match:
                codes.add(match.group(1))
    return codes


class ParticipantRegistry:
    """Builds and validates DATASET/metadata/participants.csv.

    No optical-code scanning of any kind: participant_id is the 3-digit
    code embedded in every scan/crop filename. Only presence flags and
    pipeline status are recorded, never names or other identifying fields.
    """

    @staticmethod
    def build_from_files(raw3_dir: Path, raw4_dir: Path) -> pd.DataFrame:
        """Index every 3-digit code found under raw3_dir/raw4_dir.

        Writes DATASET/metadata/participants.csv (config.paths.metadata_dir)
        and returns the same DataFrame. Every row starts at status
        "indexed" — a later QC step is responsible for qc_passed/qc_failed.
        """
        p3_codes = _codes_under(Path(raw3_dir))
        p4_codes = _codes_under(Path(raw4_dir))
        all_codes = sorted(p3_codes | p4_codes)

        df = pd.DataFrame(
            {
                "participant_id": all_codes,
                "has_p3_scan": [code in p3_codes for code in all_codes],
                "has_p4_scan": [code in p4_codes for code in all_codes],
                "status": ["indexed"] * len(all_codes),
            }
        )

        out_path = config.paths.metadata_dir / "participants.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return df

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        """Load a participants.csv, keeping participant_id a zero-padded string."""
        return pd.read_csv(path, dtype={"participant_id": str})

    @staticmethod
    def validate(df: pd.DataFrame) -> list:
        """Return human-readable problems with df; empty list means clean.

        Flags participant_id values that don't match ^\\d{3}$ and any
        duplicate participant_id.
        """
        errors = []
        ids = df["participant_id"].astype(str)

        for pid in ids[~ids.str.match(VALID_ID_PATTERN)]:
            errors.append(
                f"invalid participant_id format: {pid!r} (expected ^\\d{{3}}$)"
            )

        for pid in ids[ids.duplicated()].unique():
            errors.append(f"duplicate participant_id: {pid!r}")

        return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Participant registry: index codes under raw-3Page/raw-4Page."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build participants.csv from scan/crop filenames.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the existing participants.csv.",
    )
    args = parser.parse_args()

    if not args.build and not args.check:
        parser.error("specify --build, --check, or both")

    participants_path = config.paths.metadata_dir / "participants.csv"

    if args.build:
        df = ParticipantRegistry.build_from_files(
            config.paths.raw3_dir, config.paths.raw4_dir
        )
        print(f"Participants indexed : {len(df)}")
        print(f"Registry written     : {participants_path}")

    if args.check:
        if not participants_path.is_file():
            print(f"ERROR: participants.csv not found: {participants_path}")
            return 1
        df = ParticipantRegistry.load(participants_path)
        errors = ParticipantRegistry.validate(df)
        print(f"Participants checked : {len(df)}")
        if errors:
            print(f"Problems found       : {len(errors)}")
            for err in errors:
                print(f"  - {err}")
            return 1
        print("No problems found.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
