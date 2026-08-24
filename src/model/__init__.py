"""Model subpackage: JetImageCNN and construction helpers."""

from .build import build_model
from .cnn import ConvBlock, JetImageCNN

__all__ = ["build_model", "ConvBlock", "JetImageCNN"]
