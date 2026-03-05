"""
Labeling Module
Assigns emotion labels (HAPPY/SAD) based on questionnaire scores.
"""

import pandas as pd
from pathlib import Path
from typing import Literal


class EmotionLabeler:
    """Assigns emotion labels based on DASS and happiness scale scores."""

    # Default thresholds based on DASS-21 scoring
    DASS_THRESHOLDS = {
        'normal': (0, 9),
        'mild': (10, 13),
        'moderate': (14, 20),
        'severe': (21, 27),
        'extremely_severe': (28, float('inf'))
    }

    def __init__(
        self,
        happiness_threshold_high: float = 40.0,
        happiness_threshold_low: float = 30.0,
        dass_depression_threshold: float = 14.0
    ):
        self.happiness_high = happiness_threshold_high
        self.happiness_low = happiness_threshold_low
        self.dass_threshold = dass_depression_threshold

    def label_participant(
        self,
        dass_depression: float,
        happiness_score: float
    ) -> Literal["HAPPY", "SAD", "NEUTRAL"]:
        """
        Assign emotion label based on questionnaire scores.

        Args:
            dass_depression: DASS-21 depression subscale score
            happiness_score: Score from happiness questionnaire

        Returns:
            Emotion label: HAPPY, SAD, or NEUTRAL
        """
        # High happiness + low depression = HAPPY
        if happiness_score >= self.happiness_high and dass_depression < self.dass_threshold:
            return "HAPPY"

        # Low happiness + high depression = SAD
        if happiness_score < self.happiness_low or dass_depression >= self.dass_threshold:
            return "SAD"

        # Ambiguous cases
        return "NEUTRAL"

    def label_from_csv(self, scores_csv: str, output_csv: str) -> pd.DataFrame:
        """
        Label all participants from a scores CSV file.

        Args:
            scores_csv: Path to questionnaire scores CSV
            output_csv: Path to save labels CSV

        Returns:
            DataFrame with assigned labels
        """
        df = pd.read_csv(scores_csv)

        df['assigned_emotion'] = df.apply(
            lambda row: self.label_participant(
                row['dass_depression'],
                row['happiness_score']
            ),
            axis=1
        )

        df.to_csv(output_csv, index=False)
        return df


if __name__ == "__main__":
    labeler = EmotionLabeler()

    # Example usage
    test_cases = [
        (5, 48),   # Low depression, high happiness -> HAPPY
        (22, 18),  # High depression, low happiness -> SAD
        (10, 35),  # Moderate -> NEUTRAL
    ]

    for dass, happiness in test_cases:
        label = labeler.label_participant(dass, happiness)
        print(f"DASS: {dass}, Happiness: {happiness} -> {label}")
