"""
CNN models — Stage F.

CNNFeatureExtractor(cfg) wraps one of two backbones (CNNConfig.backbone):

  "resnet18"  torchvision ResNet18 (primary). conv1 takes 1 channel: the
              pretrained RGB filters are SUMMED over the colour channels, as
              in timm's adapt_input_conv (in_chans=1). A gray image copied
              to 3 identical channels gives exactly the same conv1 response
              as the summed filter on the gray image; averaging would scale
              it by 1/3. With imagenet_input_norm, the eval-transform input
              (mean 0.5, std 0.5) is first rescaled to ImageNet grayscale
              statistics (mean/std of the ImageNet channel means/stds),
              which is what the frozen pretrained stem and its BatchNorm
              running statistics expect.
              Everything before the first of trainable_blocks (stem, layer1,
              layer2 by default) is frozen (requires_grad False) and its
              BatchNorm layers stay in eval mode even in train() — their
              ImageNet running statistics are never updated.
  "simple"    the old custom 4-block CNN, randomly initialised (ablation).

Global features: last feature map -> global average pool -> Linear(C, 256)
-> ReLU -> Dropout; EmotionCNN adds Linear(256, 2).

Sequence features (HMM input): the seq_layer feature map (layer3: 256
channels, stride 16) averaged over height -> (batch, W/16, 256) left-to-right
columns. The simple backbone gives (batch, W/8, 256).

Run from the project root to print parameter counts:

    python -m src.models.cnn
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data.transforms import NORMALIZE_MEAN, NORMALIZE_STD

RESNET_BLOCKS = ("layer1", "layer2", "layer3", "layer4")
RESNET_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
SIMPLE_CHANNELS = 256

_IMAGENET = ResNet18_Weights.IMAGENET1K_V1.transforms()
IMAGENET_GRAY_MEAN = float(sum(_IMAGENET.mean) / len(_IMAGENET.mean))
IMAGENET_GRAY_STD = float(sum(_IMAGENET.std) / len(_IMAGENET.std))


def one_channel_conv1(conv: nn.Conv2d) -> nn.Conv2d:
    """1-input-channel copy of an RGB conv with filters summed over channels."""
    new = nn.Conv2d(
        1,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _frozen_resnet_parts(trainable_blocks) -> list:
    """Names of ResNet parts to freeze: stem + blocks before the first trainable."""
    blocks = tuple(trainable_blocks)
    if not blocks or blocks != RESNET_BLOCKS[-len(blocks) :]:
        raise ValueError(
            f"trainable_blocks must be a non-empty suffix of {RESNET_BLOCKS}, "
            f"got {blocks}"
        )
    return ["conv1", "bn1"] + list(RESNET_BLOCKS[: len(RESNET_BLOCKS) - len(blocks)])


class _ResNetBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        weights = ResNet18_Weights[cfg.weights] if cfg.use_pretrained else None
        net = resnet18(weights=weights)
        self.conv1 = one_channel_conv1(net.conv1)
        self.bn1, self.relu, self.maxpool = net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.input_norm = cfg.imagenet_input_norm
        self.frozen_names = _frozen_resnet_parts(cfg.trainable_blocks)
        for name in self.frozen_names:
            for param in getattr(self, name).parameters():
                param.requires_grad_(False)
        if cfg.seq_layer not in ("layer3", "layer4"):
            raise ValueError(f"seq_layer must be layer3 or layer4, got {cfg.seq_layer}")
        self.seq_layer = cfg.seq_layer
        self.out_channels = RESNET_CHANNELS["layer4"]
        self.seq_channels = RESNET_CHANNELS[cfg.seq_layer]

    def train(self, mode: bool = True):
        super().train(mode)
        for name in self.frozen_names:
            getattr(self, name).eval()  # frozen BatchNorm keeps ImageNet stats
        return self

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_norm:
            return x
        unit = x * NORMALIZE_STD[0] + NORMALIZE_MEAN[0]
        return (unit - IMAGENET_GRAY_MEAN) / IMAGENET_GRAY_STD

    def feature_maps(self, x: torch.Tensor) -> dict:
        x = self.maxpool(self.relu(self.bn1(self.conv1(self._prepare(x)))))
        x = self.layer2(self.layer1(x))
        l3 = self.layer3(x)
        return {"layer3": l3, "layer4": self.layer4(l3)}

    def final_and_seq(self, x: torch.Tensor):
        maps = self.feature_maps(x)
        return maps["layer4"], maps[self.seq_layer]

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.layer4


class _SimpleBackbone(nn.Module):
    """The original custom 4-block CNN (random init), kept for the ablation."""

    def __init__(self, cfg):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(cfg.input_channels, 32, kernel_size=3, padding=1),
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
            nn.Conv2d(128, SIMPLE_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(SIMPLE_CHANNELS),
            nn.ReLU(),
        )
        self.frozen_names = []
        self.out_channels = SIMPLE_CHANNELS
        self.seq_channels = SIMPLE_CHANNELS

    def final_and_seq(self, x: torch.Tensor):
        maps = self.conv_blocks(x)
        return maps, maps

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.conv_blocks[-3]  # last Conv2d


class CNNFeatureExtractor(nn.Module):
    """Backbone + global head: forward -> (batch, num_features)."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.backbone == "resnet18":
            self.backbone = _ResNetBackbone(cfg)
        elif cfg.backbone == "simple":
            self.backbone = _SimpleBackbone(cfg)
        else:
            raise ValueError(f"backbone must be resnet18 or simple, got {cfg.backbone}")
        self.num_features = cfg.num_features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, cfg.num_features),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        final, _ = self.backbone.final_and_seq(x)
        return self.fc(self.global_pool(final))

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Column sequence: seq-layer map averaged over height -> (B, W', C)."""
        _, seq_map = self.backbone.final_and_seq(x)
        return seq_map.mean(dim=2).permute(0, 2, 1)

    @property
    def seq_channels(self) -> int:
        return self.backbone.seq_channels

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.backbone.gradcam_layer


class EmotionCNN(nn.Module):
    """CNNFeatureExtractor + Linear(num_features, num_classes) classifier."""

    def __init__(self, cfg):
        super().__init__()
        self.extractor = CNNFeatureExtractor(cfg)
        self.classifier = nn.Linear(cfg.num_features, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits — (batch, num_classes)."""
        return self.classifier(self.extractor(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global features, eval mode, no grad — (batch, num_features)."""
        self.eval()
        with torch.no_grad():
            return self.extractor(x)

    def extract_sequence_features(self, x: torch.Tensor) -> torch.Tensor:
        """Column sequence, eval mode, no grad — (batch, seq_len, seq_channels)."""
        self.eval()
        with torch.no_grad():
            return self.extractor.extract_spatial_features(x)


def parameter_counts(model: nn.Module) -> tuple:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> int:
    from dataclasses import replace

    from src.utils.config import config

    for backbone in ("resnet18", "simple"):
        cfg = replace(config.cnn, backbone=backbone, use_pretrained=False)
        total, trainable = parameter_counts(EmotionCNN(cfg))
        print(
            f"{backbone:9s} total {total:>11,d}  trainable {trainable:>11,d}"
            f"  ({100 * trainable / total:.1f}%)"
        )
    print("(counts are the same with or without pretrained weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
