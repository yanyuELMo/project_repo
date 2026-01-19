from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models


class TemporalAvgResNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_resnet18(pretrained)
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = x.shape
        x = x.view(b * k, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, k, -1).mean(dim=1)  # temporal average
        logits = self.head(feats).squeeze(-1)
        return logits


def _build_resnet18(pretrained: bool) -> tuple[nn.Module, int]:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
    except AttributeError:
        resnet = models.resnet18(pretrained=pretrained)
    feat_dim = resnet.fc.in_features
    resnet.fc = nn.Identity()
    return resnet, feat_dim


def build_model(name: Literal["temporal_avg_resnet18"], pretrained: bool = True) -> nn.Module:
    if name == "temporal_avg_resnet18":
        return TemporalAvgResNet(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
