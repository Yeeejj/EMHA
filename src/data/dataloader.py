"""
Data Loading Module
Handles loading and batching of preprocessed handwriting images for training.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np

# Uncomment when dependencies are installed:
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image


class HandwritingDataset:
    """
    Dataset class for loading handwriting samples.

    Expects folder structure:
        DATA/PROCESSED/
        ├── HAPPY/
        │   ├── sample1.png
        │   └── ...
        └── SAD/
            ├── sample1.png
            └── ...
    """

    EMOTION_TO_LABEL = {"HAPPY": 0, "SAD": 1}
    LABEL_TO_EMOTION = {0: "HAPPY", 1: "SAD"}

    def __init__(
        self,
        data_dir: str = "DATA/PROCESSED",
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan directories and load sample paths with labels."""
        for emotion, label in self.EMOTION_TO_LABEL.items():
            emotion_dir = self.data_dir / emotion
            if emotion_dir.exists():
                for img_path in emotion_dir.glob("*.png"):
                    self.samples.append((str(img_path), label))
                for img_path in emotion_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), label))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load and return a sample."""
        img_path, label = self.samples[idx]

        # TODO: Implement actual image loading
        # image = Image.open(img_path).convert('RGB')
        # image = image.resize(self.image_size)
        # image = np.array(image) / 255.0

        # Placeholder
        image = np.zeros((*self.image_size, 3))

        return image, label

    def get_class_distribution(self) -> dict:
        """Return count of samples per class."""
        distribution = {"HAPPY": 0, "SAD": 0}
        for _, label in self.samples:
            emotion = self.LABEL_TO_EMOTION[label]
            distribution[emotion] += 1
        return distribution


def create_data_loaders(
    data_dir: str = "DATA/SPLITS",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224)
) -> dict:
    """
    Create train, validation, and test data loaders.

    Returns:
        Dictionary with 'train', 'val', 'test' loaders
    """
    loaders = {}

    for split in ['train', 'val', 'test']:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            dataset = HandwritingDataset(
                data_dir=str(split_dir),
                image_size=image_size,
                augment=(split == 'train')
            )
            loaders[split] = dataset
            # TODO: Wrap in actual DataLoader when PyTorch is available
            # loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

    return loaders


if __name__ == "__main__":
    dataset = HandwritingDataset()
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
