"""
CNN Feature Extractor Module
Extracts spatial features from preprocessed handwriting images.
"""

from typing import Tuple

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN model for extracting features from handwriting images.

    Architecture:
    - 4 convolutional blocks with BatchNorm + ReLU + pooling
    - AdaptiveAvgPool to (1,1) for global features
    - FC layer to num_features dimensions
    - Also supports extracting spatial sequence features for HMM
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_features: int = 256,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_features = num_features

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature vector. Shape: (batch, num_features)."""
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial sequence features for HMM input.

        Returns feature maps averaged along height, producing a
        left-to-right sequence representing the spatial progression
        of handwriting.

        Input:  (batch, channels, H, W)
        Output: (batch, seq_len, 256) where seq_len = W // 8
        """
        feat_maps = self.conv_blocks(x)  # (batch, 256, H/8, W/8)
        # Average along height → left-to-right sequence
        # (batch, 256, W/8)
        seq = feat_maps.mean(dim=2)
        # Transpose to (batch, seq_len, 256)
        seq = seq.permute(0, 2, 1)
        return seq


class EmotionCNN(nn.Module):
    """Full CNN with classifier head for pretraining the feature extractor."""

    def __init__(
        self,
        input_channels: int = 1,
        num_features: int = 256,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.extractor = CNNFeatureExtractor(
            input_channels, num_features, dropout_rate
        )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        features = self.extractor(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global features (no grad). Shape: (batch, num_features)."""
        self.eval()
        with torch.no_grad():
            return self.extractor(x)

    def extract_sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial sequence features (no grad) for HMM."""
        self.eval()
        with torch.no_grad():
            return self.extractor.extract_spatial_features(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN().to(device)
    dummy = torch.randn(2, 1, 224, 224).to(device)

    out = model(dummy)
    print(f"Logits shape: {out.shape}")  # (2, 2)

    feats = model.extract_features(dummy)
    print(f"Global feature shape: {feats.shape}")  # (2, 256)

    seq_feats = model.extract_sequence_features(dummy)
    print(f"Sequence feature shape: {seq_feats.shape}")  # (2, 28, 256)
