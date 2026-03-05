"""
CNN Feature Extractor Module
Extracts spatial features from preprocessed handwriting images.
"""

from typing import Tuple

# Uncomment when PyTorch is installed:
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class CNNFeatureExtractor:
    """
    CNN model for extracting features from handwriting images.

    Architecture:
    - Input: (batch, 3, 224, 224) or (batch, 1, 224, 224) for grayscale
    - Conv layers with ReLU and MaxPooling
    - Output: Feature vector for HMM input
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_features: int = 256,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.input_channels = input_channels
        self.num_features = num_features
        self.image_size = image_size

        # TODO: Build actual CNN architecture
        # self.model = self._build_model()

    def _build_model(self):
        """
        Build CNN architecture.

        Example architecture:
        Conv2d(1, 32, 3) -> ReLU -> MaxPool
        Conv2d(32, 64, 3) -> ReLU -> MaxPool
        Conv2d(64, 128, 3) -> ReLU -> MaxPool
        Conv2d(128, 256, 3) -> ReLU -> AdaptiveAvgPool
        Flatten -> Linear(256, num_features)
        """
        # layers = nn.Sequential(
        #     nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #
        #     nn.Flatten(),
        #     nn.Linear(256, self.num_features)
        # )
        # return layers
        pass

    def forward(self, x):
        """Extract features from input images."""
        # return self.model(x)
        pass

    def extract_features(self, image):
        """Extract feature vector from a single image."""
        # self.model.eval()
        # with torch.no_grad():
        #     features = self.forward(image.unsqueeze(0))
        # return features.squeeze(0)
        pass


class PretrainedCNNExtractor:
    """
    Use pretrained CNN (ResNet, VGG, etc.) as feature extractor.
    Transfer learning approach.
    """

    def __init__(self, backbone: str = "resnet18", num_features: int = 256):
        self.backbone = backbone
        self.num_features = num_features
        # TODO: Load pretrained model and modify final layers

    def extract_features(self, image):
        """Extract features using pretrained backbone."""
        pass


if __name__ == "__main__":
    extractor = CNNFeatureExtractor(
        input_channels=1,
        num_features=256,
        image_size=(224, 224)
    )
    print(f"CNN Feature Extractor initialized")
    print(f"Output features: {extractor.num_features}")
