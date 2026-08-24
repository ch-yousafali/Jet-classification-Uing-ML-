"""Data loading and jet-image construction for the Top Quark Tagging dataset.

The Top Quark Tagging Reference Dataset (Kasieczka et al., 2019,
https://zenodo.org/record/2603256) is distributed as three HDF5 files
(train.h5, val.h5, test.h5). Each file stores, per jet:

  - E, PX, PY, PZ   : (200,) constituent four-momenta (zero-padded)
  - Eta, Phi        : (200,) constituent pseudorapidity / azimuth
  - is_signal_new   : (1,) label, 1 = top quark, 0 = QCD background
  - ttv             : (1,) split flag (redundant across files)
  - truth{E,PX,PY,PZ}: top-quark four-momentum

This module turns the list of constituents into a 2D "jet image"
(eta-phi histogram weighted by pT) which is fed to a CNN, following
the image-based approach described in the README (Approach 1 / DeepTop).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Zenodo download URLs for the three splits.
ZENODO_BASE = "https://zenodo.org/record/2603256/files"
SPLIT_URLS = {
    "train": f"{ZENODO_BASE}/train.h5",
    "val": f"{ZENODO_BASE}/val.h5",
    "test": f"{ZENODO_BASE}/test.h5",
}

# Image grid parameters. Anti-kT R=0.8 jets fit within |eta_rel|, |phi_rel| < 0.8.
IMG_SIZE = 40
IMG_RANGE = 0.8  # half-width of the eta-phi window in radians


def download_split(split: str, data_dir: str) -> str:
    """Download one split's HDF5 file from Zenodo if not already present.

    Returns the local path to the file. `split` must be one of
    train/val/test.
    """
    if split not in SPLIT_URLS:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_URLS)}")
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, f"{split}.h5")
    if os.path.exists(dest):
        print(f"[data] {dest} already exists, skipping download.")
        return dest
    url = SPLIT_URLS[split]
    print(f"[data] Downloading {split} from {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[data] Done ({os.path.getsize(dest) / 1e9:.2f} GB).")
    return dest


def ensure_splits(data_dir: str, splits=("train", "val", "test")) -> dict:
    """Download all requested splits and return {split: path}."""
    return {s: download_split(s, data_dir) for s in splits}


# --------------------------------------------------------------------------- #
# Jet-image construction
# --------------------------------------------------------------------------- #


def _delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Smallest signed angular difference on the azimuthal circle."""
    d = phi1 - phi2
    return (d + np.pi) % (2 * np.pi) - np.pi


def build_jet_images(
    E: np.ndarray,
    PX: np.ndarray,
    PY: np.ndarray,
    PZ: np.ndarray,
    Eta: np.ndarray,
    Phi: np.ndarray,
    img_size: int = IMG_SIZE,
    img_range: float = IMG_RANGE,
) -> np.ndarray:
    """Convert constituent four-momenta into pT-weighted eta-phi images.

    Inputs are (N, 200) arrays (zero-padded constituents). The jet axis is
    taken as the pT-weighted centroid of the constituents, and constituents
    are histogrammed into an `img_size` x `img_size` grid spanning
    [-img_range, img_range] in (eta_rel, phi_rel). Pixel value = sum of pT
    of constituents falling in that bin.

    Returns a float32 array of shape (N, 1, img_size, img_size).
    """
    N = E.shape[0]
    # pT of each constituent; zero-padded entries have E=0 -> pT=0.
    pt = np.sqrt(np.maximum(PX**2 + PY**2, 0.0))  # (N, 200)

    # Jet axis: pT-weighted mean of constituent eta/phi.
    # Guard against all-zero rows (should not happen, but be safe).
    pt_sum = pt.sum(axis=1, keepdims=True)
    pt_sum_safe = np.where(pt_sum > 0, pt_sum, 1.0)
    jet_eta = (pt * Eta).sum(axis=1, keepdims=True) / pt_sum_safe  # (N,1)
    jet_phi = (pt * Phi).sum(axis=1, keepdims=True) / pt_sum_safe

    eta_rel = Eta - jet_eta  # (N, 200)
    phi_rel = _delta_phi(Phi, jet_phi)

    # Bin indices in [0, img_size).
    bins = np.linspace(-img_range, img_range, img_size + 1)
    ix = np.digitize(eta_rel, bins) - 1  # (N, 200)
    iy = np.digitize(phi_rel, bins) - 1
    ix = np.clip(ix, 0, img_size - 1)
    iy = np.clip(iy, 0, img_size - 1)

    images = np.zeros((N, img_size, img_size), dtype=np.float32)
    # Use np.add.at for scatter-add of pT into the image grid.
    flat_idx = ix * img_size + iy  # (N, 200)
    # Flatten over constituents and scatter per-event.
    for n in range(N):
        np.add.at(images[n].ravel(), flat_idx[n], pt[n])
    # Log-compress and standardize per-image (mean 0, std 1), common in
    # jet-image literature. Keep a channel dim for the CNN.
    images = np.log1p(images)
    mean = images.mean(axis=(1, 2), keepdims=True)
    std = images.std(axis=(1, 2), keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    images = (images - mean) / std
    return images[:, None, :, :].astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


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
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            n = f["E"].shape[0]
            if self.cfg.max_events is not None:
                n = min(n, self.cfg.max_events)
            sl = slice(0, n)
            E = f["E"][sl]
            PX = f["PX"][sl]
            PY = f["PY"][sl]
            PZ = f["PZ"][sl]
            Eta = f["Eta"][sl]
            Phi = f["Phi"][sl]
            # Label key is `is_signal_new` in the canonical Zenodo files.
            if "is_signal_new" in keys:
                y = f["is_signal_new"][sl]
            else:
                # Fallback: some derived versions use a different name.
                raise KeyError(f"Could not find label key in {keys}")
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
