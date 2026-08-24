"""PyTorch Dataset wrapping the Top Quark Tagging jet images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .download import download_split
from .jet_image import IMG_SIZE, build_jet_images, load_split_arrays


@dataclass
class DatasetConfig:
    data_dir: str = "data"
    img_size: int = IMG_SIZE
    max_events: int | None = None  # cap number of jets loaded (for smoke tests)


class JetImageDataset(Dataset):
    """PyTorch dataset that yields (jet_image, label) tensors.

    Jet images are built once from the constituent four-momenta and cached
    in memory. For the full 1.2M-event training set this needs ~1.2M * 40*40*4
    bytes ~= 7.7 GB of RAM; if that is too large, set `max_events` or move
    caching to disk (see notes in README).
    """

    def __init__(self, split: str, cfg: DatasetConfig | None = None):
        self.split = split
        self.cfg = cfg or DatasetConfig()
        path = download_split(split, self.cfg.data_dir)
        print(f"[data] Loading {split} from {path} ...")
        E, PX, PY, PZ, Eta, Phi, y = load_split_arrays(path, self.cfg.max_events)
        n = E.shape[0]
        print(f"[data] Loaded {n} jets. Building jet images ...")
        self.images = build_jet_images(
            E, PX, PY, PZ, Eta, Phi, img_size=self.cfg.img_size
        )
        self.labels = np.asarray(y, dtype=np.float32).reshape(-1)
        pos = int(self.labels.sum())
        print(f"[data] {self.split}: {pos} top / {n - pos} qcd")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.images[idx])
        y = torch.tensor(self.labels[idx])
        return x, y
