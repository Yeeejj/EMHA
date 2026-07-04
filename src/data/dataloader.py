"""
Data Loading Module
Handles loading and batching of preprocessed handwriting images for training.
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class HandwritingDataset(Dataset):
    """
    PyTorch Dataset for loading handwriting samples.

    Expects folder structure:
        data_dir/
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
        transform: transforms.Compose = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan directories and load sample paths with labels."""
        for emotion, label in self.EMOTION_TO_LABEL.items():
            emotion_dir = self.data_dir / emotion
            if emotion_dir.exists():
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for img_path in emotion_dir.glob(ext):
                        self.samples.append((str(img_path), label))

        print(
            f"Loaded {len(self.samples)} samples "
            f"(HAPPY: {sum(1 for _, l in self.samples if l == 0)}, "
            f"SAD: {sum(1 for _, l in self.samples if l == 1)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and return a sample."""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def get_labels(self) -> List[int]:
        """Return list of all labels."""
        return [label for _, label in self.samples]

    def get_class_distribution(self) -> dict:
        """Return count of samples per class."""
        distribution = {"HAPPY": 0, "SAD": 0}
        for _, label in self.samples:
            emotion = self.LABEL_TO_EMOTION[label]
            distribution[emotion] += 1
        return distribution


def get_train_transform(image_size: Tuple[int, int] = (224, 224)):
    """Training transforms with handwriting-safe augmentations."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transform(image_size: Tuple[int, int] = (224, 224)):
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class TransformSubset(Dataset):
    """Subset of a dataset with a custom transform override."""

    def __init__(self, dataset: HandwritingDataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


def create_data_loaders(
    data_dir: str = "DATA/SPLITS",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0
) -> dict:
    """Create train, validation, and test data loaders."""
    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            transform = train_transform if split == "train" else val_transform
            dataset = HandwritingDataset(
                data_dir=str(split_dir),
                transform=transform,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
            )

    return loaders


if __name__ == "__main__":
    dataset = HandwritingDataset()
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
