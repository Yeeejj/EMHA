"""
Data Collection Module
Handles image capture and questionnaire management for handwriting samples.
"""

import os
from pathlib import Path
from datetime import datetime


class DataCollector:
    """Manages collection of handwriting samples and participant data."""

    def __init__(self, raw_data_path: str = "DATA/RAW"):
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def create_participant_folder(self, participant_id: str) -> Path:
        """Create a folder for a new participant."""
        participant_path = self.raw_data_path / participant_id
        participant_path.mkdir(exist_ok=True)
        return participant_path

    def generate_participant_id(self) -> str:
        """Generate a unique participant ID."""
        existing = list(self.raw_data_path.glob("P*"))
        next_num = len(existing) + 1
        return f"P{next_num:03d}"

    def save_sample(self, participant_id: str, image_data, sample_num: int = 1) -> str:
        """Save a handwriting sample for a participant."""
        participant_path = self.raw_data_path / participant_id
        participant_path.mkdir(exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{participant_id}_sample_{sample_num:02d}_{date_str}.png"
        filepath = participant_path / filename

        # TODO: Implement actual image saving logic
        # image_data.save(filepath)

        return str(filepath)


if __name__ == "__main__":
    collector = DataCollector()
    pid = collector.generate_participant_id()
    print(f"Generated participant ID: {pid}")
