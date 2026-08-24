"""Data subpackage: download, jet-image construction, and Dataset.

Convenience re-exports so `from src.data import X` keeps working for the
common names used by the training/evaluation scripts.
"""

from .dataset import DatasetConfig, JetImageDataset
from .download import SPLIT_URLS, download_split, ensure_splits
from .jet_image import IMG_RANGE, IMG_SIZE, build_jet_images, load_split_arrays

__all__ = [
    "DatasetConfig",
    "JetImageDataset",
    "SPLIT_URLS",
    "download_split",
    "ensure_splits",
    "IMG_SIZE",
    "IMG_RANGE",
    "build_jet_images",
    "load_split_arrays",
]
