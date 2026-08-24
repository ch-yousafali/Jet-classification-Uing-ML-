# Project notes for agents

## Project
Jet classification (top-quark tagging) on the Top Quark Tagging Reference
Dataset (Zenodo 2603256) using a CNN baseline on jet images. PyTorch.

## Environment
- Python venv at `.venv/` (PEP 668 blocks system pip on this machine).
- Install: `.venv/bin/pip install -r requirements.txt`
- No GPU detected; runs on CPU. Code auto-detects CUDA if present.

## Commands
- Smoke test (small subset, no full download needed beyond test.h5 used for both splits):
  `.venv/bin/python -m src.train --max-events 4000 --epochs 3 --batch-size 256 --splits train val`
- Full training:
  `.venv/bin/python -m src.train --epochs 20 --batch-size 512`
- Evaluate on test split:
  `.venv/bin/python -m src.evaluate --ckpt checkpoints/cnn_best.pt`

## Dataset
- HDF5 files from https://zenodo.org/record/2603256 (train.h5, val.h5, test.h5).
- Keys: E, PX, PY, PZ, Eta, Phi (shape (N, 200)); is_signal_new (label);
  ttv (split); truth{E,PX,PY,PZ}.
- Downloaded automatically into `data/` by `src.data.download_split`.

## Layout
- `src/data.py` — download + jet-image construction + Dataset
- `src/model.py` — JetImageCNN
- `src/train.py` — training loop with AUC tracking + checkpointing
- `src/evaluate.py` — test-split evaluation (AUC, accuracy, 1/eps_B @ eps_S=0.3)
