"""Tests for src.models.cnn (use_pretrained=False unless marked network)."""

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from src.models.cnn import (
    CNNFeatureExtractor,
    EmotionCNN,
    IMAGENET_GRAY_MEAN,
    IMAGENET_GRAY_STD,
    one_channel_conv1,
    parameter_counts,
)
from src.utils.config import config

FAMILIES = ("drawing", "word", "cursive")


def _cfg(**kw):
    return replace(config.cnn, use_pretrained=False, **kw)


def _canvas(family, batch=2, channels=1):
    w, h = config.preprocessing.canvas_size[family]
    torch.manual_seed(0)
    return torch.randn(batch, channels, h, w)


def _down(n, k, s, p):
    return (n + 2 * p - k) // s + 1


def _resnet_layer3_width(w):
    w = _down(w, 7, 2, 3)  # conv1
    w = _down(w, 3, 2, 1)  # maxpool
    w = _down(w, 3, 2, 1)  # layer2
    return _down(w, 3, 2, 1)  # layer3


@pytest.mark.parametrize("family", FAMILIES)
def test_resnet_shapes_per_family_canvas(family):
    model = EmotionCNN(_cfg())
    x = _canvas(family)
    assert model(x).shape == (2, 2)
    assert model.extract_features(x).shape == (2, 256)
    seq = model.extract_sequence_features(x)
    assert seq.shape == (2, _resnet_layer3_width(x.shape[-1]), 256)
    assert CNNFeatureExtractor(_cfg())(x).shape == (2, 256)


@pytest.mark.parametrize("family", FAMILIES)
def test_simple_backbone_shapes(family):
    model = EmotionCNN(_cfg(backbone="simple"))
    x = _canvas(family)
    assert model(x).shape == (2, 2)
    assert model.extract_sequence_features(x).shape == (2, x.shape[-1] // 8, 256)


def test_layer3_sequence_is_16x_downsampled_and_256_channels():
    model = EmotionCNN(_cfg())
    seq = model.extract_sequence_features(torch.randn(1, 1, 64, 320))
    assert seq.shape == (1, 320 // 16, 256)


def test_seq_layer4_option():
    model = EmotionCNN(_cfg(seq_layer="layer4"))
    seq = model.extract_sequence_features(torch.randn(1, 1, 64, 320))
    assert seq.shape == (1, 320 // 32, 512)


def test_one_channel_input_only():
    model = EmotionCNN(_cfg())
    assert model.extractor.backbone.conv1.in_channels == 1
    model(torch.randn(2, 1, 58, 443))
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 3, 58, 443))


def test_frozen_parameters_have_requires_grad_false():
    model = EmotionCNN(_cfg())
    bb = model.extractor.backbone
    for name in ("conv1", "bn1", "layer1", "layer2"):
        assert all(not p.requires_grad for p in getattr(bb, name).parameters()), name
    for module in (bb.layer3, bb.layer4, model.extractor.fc, model.classifier):
        assert all(p.requires_grad for p in module.parameters())


def test_frozen_batchnorm_stays_in_eval_mode_and_stats_unchanged():
    model = EmotionCNN(_cfg())
    model.train()
    bb = model.extractor.backbone
    frozen_bn = [
        m
        for n in ("bn1", "layer1", "layer2")
        for m in getattr(bb, n).modules()
        if isinstance(m, nn.BatchNorm2d)
    ]
    live_bn = [m for m in bb.layer3.modules() if isinstance(m, nn.BatchNorm2d)]
    assert frozen_bn and all(not m.training for m in frozen_bn)
    assert live_bn and all(m.training for m in live_bn)
    before = [m.running_mean.clone() for m in frozen_bn]
    model(_canvas("word"))
    assert all(torch.equal(b, m.running_mean) for b, m in zip(before, frozen_bn))
    model.eval()
    assert all(not m.training for m in live_bn)


def test_backward_cpu_batch2_only_trainable_get_grads():
    model = EmotionCNN(_cfg())
    model.train()
    loss = nn.functional.cross_entropy(model(_canvas("cursive")), torch.tensor([0, 1]))
    loss.backward()
    bb = model.extractor.backbone
    assert bb.conv1.weight.grad is None and bb.layer2[0].conv1.weight.grad is None
    assert bb.layer3[0].conv1.weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_trainable_blocks_must_be_a_suffix():
    for bad in (("layer3",), ("layer2", "layer4"), ()):
        with pytest.raises(ValueError, match="suffix"):
            EmotionCNN(_cfg(trainable_blocks=bad))
    model = EmotionCNN(_cfg(trainable_blocks=("layer4",)))
    assert all(
        not p.requires_grad for p in model.extractor.backbone.layer3.parameters()
    )


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError, match="backbone"):
        EmotionCNN(_cfg(backbone="vgg"))


def test_input_rescaled_to_imagenet_gray_statistics():
    bb = EmotionCNN(_cfg()).extractor.backbone
    white = torch.ones(1, 1, 2, 2)  # eval transform output for pixel 255
    expected = (1.0 - IMAGENET_GRAY_MEAN) / IMAGENET_GRAY_STD
    assert torch.allclose(bb._prepare(white), torch.full_like(white, expected))
    raw = EmotionCNN(_cfg(imagenet_input_norm=False)).extractor.backbone
    assert torch.equal(raw._prepare(white), white)


def test_one_channel_conv1_sums_rgb_filters():
    rgb = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
    gray = one_channel_conv1(rgb)
    assert torch.allclose(gray.weight, rgb.weight.sum(dim=1, keepdim=True))
    x = torch.randn(1, 1, 8, 8)
    assert torch.allclose(gray(x), rgb(x.repeat(1, 3, 1, 1)), atol=1e-6)


def test_parameter_counts_reflect_freezing():
    total, trainable = parameter_counts(EmotionCNN(_cfg()))
    assert 0 < trainable < total
    s_total, s_trainable = parameter_counts(EmotionCNN(_cfg(backbone="simple")))
    assert s_total == s_trainable


def test_hybrid_still_builds_with_the_cnn():
    from src.models.hybrid import HybridCNNHMM

    hybrid = HybridCNNHMM(device=torch.device("cpu"))
    assert isinstance(hybrid.cnn, EmotionCNN)


@pytest.mark.network
def test_pretrained_conv1_is_summed_imagenet_filters():
    from torchvision.models import ResNet18_Weights, resnet18

    try:
        reference = resnet18(weights=ResNet18_Weights[config.cnn.weights])
    except Exception as exc:  # no network and no local cache
        pytest.skip(f"pretrained weights unavailable: {exc}")
    model = EmotionCNN(replace(config.cnn, use_pretrained=True))
    conv1 = model.extractor.backbone.conv1.weight
    assert conv1.shape == (64, 1, 7, 7)
    assert torch.allclose(conv1, reference.conv1.weight.sum(dim=1, keepdim=True))
    assert torch.equal(
        model.extractor.backbone.layer4[0].conv1.weight,
        reference.layer4[0].conv1.weight,
    )
