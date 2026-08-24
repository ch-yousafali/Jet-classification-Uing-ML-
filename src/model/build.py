"""Model construction helpers."""

from __future__ import annotations

import torch

from .cnn import JetImageCNN


def build_model(img_size: int = 40, device: str | torch.device = "cpu") -> JetImageCNN:
    model = JetImageCNN(img_size=img_size)
    return model.to(device)
