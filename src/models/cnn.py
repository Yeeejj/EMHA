"""
CNN Feature Extractor Module
Extracts spatial features from preprocessed handwriting images.

Two backends:
  CNNFeatureExtractor   — custom 4-block CNN (secondary, kept for reference)
  PretrainedCNNExtractor — ResNet18 backbone (primary per CLAUDE.md)

EmotionCNN selects the backend via use_pretrained (default True in config).
"""

import torch
import torch.nn as nn

# ─── custom 4-block CNN ───────────────────────────────────────────────────────


class CNNFeatureExtractor(nn.Module):
    """
    Custom 4-block CNN.

    Input:  (batch, input_channels, 224, 224)
    Output: (batch, num_features)  — global features
    Spatial output: (batch, 28, num_features)  — sequence for HMM
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
        """Global feature vector — (batch, num_features)."""
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        return self.fc(x)

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial sequence features for HMM.

        Input:  (batch, channels, H, W)
        Output: (batch, W//8, num_features)  — left-to-right sequence
        """
        feat_maps = self.conv_blocks(x)  # (B, 256, H/8, W/8)
        seq = feat_maps.mean(dim=2)  # (B, 256, W/8)
        return seq.permute(0, 2, 1)  # (B, W/8, 256)


# ─── ResNet18 backbone ────────────────────────────────────────────────────────


class PretrainedCNNExtractor(nn.Module):
    """
    ResNet18 backbone adapted for 1-channel grayscale input.

    The pretrained conv1 filters are averaged across the three colour
    channels to produce a single-channel initialisation that preserves
    the pretrained feature response magnitudes.

    Input:  (batch, 1, 224, 224)
    Output: (batch, num_features)  — global features
    Spatial output: (batch, 7, num_features)  — seq_len = 7 for 224×224 input
    """

    def __init__(
        self,
        num_features: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        try:
            import torchvision.models as models

            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            import torchvision.models as models

            backbone = models.resnet18(pretrained=True)

        # ── adapt conv1 for 1-channel input ──────────────────────────────
        w3 = backbone.conv1.weight.data  # (64, 3, 7, 7)
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight = nn.Parameter(w3.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv1

        # Split into (conv1..layer4) and avgpool; drop the fc head
        children = list(backbone.children())
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        self.feature_layers = nn.Sequential(*children[:-2])
        self.global_pool = children[-2]  # AdaptiveAvgPool2d

        if freeze_backbone:
            for p in self.feature_layers.parameters():
                p.requires_grad = False
            # conv1 weights were re-initialised — keep them trainable
            self.feature_layers[0].weight.requires_grad = True

        self.proj = nn.Linear(512, num_features)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global feature vector — (batch, num_features)."""
        maps = self.feature_layers(x)  # (B, 512, 7, 7)
        pooled = self.global_pool(maps).flatten(1)  # (B, 512)
        return self.drop(torch.relu(self.proj(pooled)))

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial sequence features for HMM.

        Input:  (batch, 1, 224, 224)
        Output: (batch, 7, num_features)
        """
        maps = self.feature_layers(x)  # (B, 512, 7, 7)
        seq = maps.mean(dim=2).permute(0, 2, 1)  # (B, 7, 512)
        B, T, _ = seq.shape
        out = torch.relu(self.proj(seq.reshape(B * T, 512)))
        return out.reshape(B, T, self.num_features)


# ─── EmotionCNN (full model with classifier head) ────────────────────────────


class EmotionCNN(nn.Module):
    """
    Full CNN with classifier head.

    Selects backend via use_pretrained:
      True  → PretrainedCNNExtractor (ResNet18, primary per CLAUDE.md)
      False → CNNFeatureExtractor   (custom 4-block CNN)
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_features: int = 256,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        use_pretrained: bool = False,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if use_pretrained:
            self.extractor = PretrainedCNNExtractor(
                num_features=num_features,
                freeze_backbone=freeze_backbone,
            )
        else:
            self.extractor = CNNFeatureExtractor(
                input_channels=input_channels,
                num_features=num_features,
                dropout_rate=dropout_rate,
            )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        return self.classifier(self.extractor(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global features, no grad — (batch, num_features)."""
        self.eval()
        with torch.no_grad():
            return self.extractor(x)

    def extract_sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial sequence features, no grad — (batch, seq_len, num_features)."""
        self.eval()
        with torch.no_grad():
            return self.extractor.extract_spatial_features(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 1, 224, 224).to(device)

    for pretrained in (False, True):
        label = "pretrained" if pretrained else "custom"
        model = EmotionCNN(use_pretrained=pretrained).to(device)
        logits = model(dummy)
        feats = model.extract_features(dummy)
        seq = model.extract_sequence_features(dummy)
        print(
            f"{label}: logits={tuple(logits.shape)} "
            f"feats={tuple(feats.shape)} "
            f"seq={tuple(seq.shape)}"
        )
