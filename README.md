# Jet Classification Using Machine Learning — CNN Baseline

A PyTorch implementation of a **Convolutional Neural Network (CNN) baseline**
for top-quark vs. QCD jet tagging on the
[Top Quark Tagging Reference Dataset](https://zenodo.org/record/2603256)
(Zenodo 2603256). Jets are rendered as 2D images in (η, φ) and classified
as originating from a top quark or from a light quark / gluon.

> The full research report (problem statement, dataset review, literature
> comparison, timeline, and references) lives in **[docs/THESIS.md](docs/THESIS.md)**.
> This README documents the model and the code only.

---

## Project structure

```
thesis-repo/
├── README.md
├── AGENTS.md            # notes for AI coding agents
├── requirements.txt     # pinned dependencies
├── .gitignore           # ignores data/, checkpoints/, .venv/
├── docs/
│   └── THESIS.md        # research report (problem, dataset, literature)
├── data/                # downloaded HDF5 files (gitignored, ~1.4 GB)
├── checkpoints/         # saved models + history (gitignored)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py  # re-exports the public API
│   │   ├── download.py  # Zenodo download + caching
│   │   ├── jet_image.py # raw array loading + jet-image construction
│   │   └── dataset.py   # JetImageDataset + DatasetConfig
│   ├── model/
│   │   ├── __init__.py  # re-exports the public API
│   │   ├── cnn.py       # ConvBlock + JetImageCNN
│   │   └── build.py     # build_model helper
│   ├── train.py         # training loop, AUC tracking, checkpointing
│   └── evaluate.py      # test-split evaluation
└── tests/
    ├── __init__.py
    ├── test_model.py    # CNN shape / param / gradient tests
    ├── test_data.py     # jet-image + kinematics + dataset tests
    └── test_train.py    # one training step smoke tests
```

## Setup

A Python virtual environment is used because the system `pip` is blocked
by PEP 668 on this machine.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## How to run

The dataset is downloaded automatically from Zenodo into `data/` on
first use, so no manual download step is needed.

Quick smoke test (caps each split to 4000 jets, 3 epochs, CPU-friendly):

```bash
.venv/bin/python -m src.train --max-events 4000 --epochs 3 --batch-size 256 --splits train val
```

Full training (1.2M training jets, 20 epochs):

```bash
.venv/bin/python -m src.train --epochs 20 --batch-size 512
```

Evaluate the best checkpoint on the held-out test split:

```bash
.venv/bin/python -m src.evaluate --ckpt checkpoints/cnn_best.pt
```

## Tests

The test suite covers the model, the data layer, and a single training
step. Tests that need the downloaded HDF5 files skip themselves
automatically when the files are absent, so the suite runs on a fresh
checkout without downloading 1.4 GB.

```bash
.venv/bin/python -m pytest tests/ -v
```

What is covered:

| File | What it checks |
|---|---|
| `tests/test_model.py` | `ConvBlock` and `JetImageCNN` output shapes, parameter count (~913k), presence of BatchNorm/Dropout, gradient flow through all parameters, determinism in eval mode, finiteness of outputs. |
| `tests/test_data.py` | `_delta_phi` wrap-around behaviour, `build_jet_images` shape / standardization / robustness to all-zero jets, `load_split_arrays` shapes and on-shell energy constraint (E² ≥ p²), `JetImageDataset` length and item format. |
| `tests/test_train.py` | `run_epoch` runs in train and eval modes, returns a valid AUC in [0,1], and loss decreases over repeated steps on separable synthetic data. |

## Current status

The CNN baseline pipeline runs end-to-end. On a small smoke subset
(4000 jets per split, 3 epochs, CPU only) it reaches:

- Validation AUC: **0.9275**
- Test AUC: **0.9248**
- Background rejection 1/ε_B at ε_S = 0.3: **~41**

These numbers are well below the published P-CNN AUC of 0.9803 (see
[docs/THESIS.md](docs/THESIS.md), Section 4) because the smoke run uses
only ~0.3% of the training set and three epochs. A full training run on
the 1.2M-event training set is the next step to get a comparable number;
on this CPU-only machine that run is slow, so it has not been completed
yet.

## Next steps

1. Run full training (20 epochs, full 1.2M training set) and record the
   test AUC alongside the published P-CNN / ResNeXt numbers.
2. Build the point-cloud / GNN model (Approach 2, ParticleNet-style) as
   a second model under `src/model/` so the two can be compared on the
   same test split.
3. Add a plotting script for ROC curves and the AUC comparison table.
