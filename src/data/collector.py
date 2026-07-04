"""
Data Collection Module
Handles image capture and questionnaire management for handwriting samples.
"""

import os
import re
from pathlib import Path
from datetime import datetime

import pandas as pd


class DataCollector:
    """Manages collection of handwriting samples and participant data."""

    ID_PATTERN = re.compile(r"^P\d{3}$")

    def __init__(
        self,
        raw_data_path: str = "DATA/RAW",
        metadata_path: str = "DATA/METADATA"
    ):
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = Path(metadata_path)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.participants_csv = self.metadata_path / "participants.csv"

    def _get_existing_ids(self) -> set:
        """Get all existing participant IDs from both filesystem and CSV."""
        ids = set()

        # From filesystem
        for entry in self.raw_data_path.iterdir():
            if entry.is_dir() and self.ID_PATTERN.match(entry.name):
                ids.add(entry.name)

        # From participants.csv (authoritative source)
        if self.participants_csv.exists():
            try:
                df = pd.read_csv(self.participants_csv)
                if "participant_id" in df.columns:
                    ids.update(df["participant_id"].dropna().tolist())
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass

        return ids

    def generate_participant_id(self) -> str:
        """Generate a unique participant ID, validated against existing records."""
        existing_ids = self._get_existing_ids()

        # Find the highest existing number and increment
        max_num = 0
        for pid in existing_ids:
            match = re.match(r"^P(\d{3})$", pid)
            if match:
                max_num = max(max_num, int(match.group(1)))

        next_id = f"P{max_num + 1:03d}"

        # Safety check: ensure uniqueness
        while next_id in existing_ids:
            max_num += 1
            next_id = f"P{max_num + 1:03d}"

        return next_id

    def create_participant_folder(self, participant_id: str) -> Path:
        """Create a folder for a new participant."""
        if not self.ID_PATTERN.match(participant_id):
            raise ValueError(
                f"Invalid participant ID format: {participant_id}. "
                "Expected format: P001, P002, etc."
            )
        participant_path = self.raw_data_path / participant_id
        participant_path.mkdir(exist_ok=True)
        return participant_path

    def save_sample(
        self,
        participant_id: str,
        image_data,
        sample_num: int = 1
    ) -> str:
        """Save a handwriting sample for a participant."""
        participant_path = self.raw_data_path / participant_id
        participant_path.mkdir(exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{participant_id}_sample_{sample_num:02d}_{date_str}.png"
        filepath = participant_path / filename

        image_data.save(str(filepath))

        return str(filepath)


if __name__ == "__main__":
    collector = DataCollector()
    pid = collector.generate_participant_id()
    print(f"Generated participant ID: {pid}")
