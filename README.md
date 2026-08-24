# Jet Classification Using Machine Learning on CERN Open Data

**A Research Report on Baseline Reproduction and Extension Approaches**

Yousuf Ali | Computational Physics, Punjab University Lahore | 5th Semester

---

## 1. What is the Problem?

At CERN, a physics lab in Switzerland, particles are smashed together at very high speed. When they collide, many small particles fly out together in a group. This group is called a jet.

The question this project deals with is: can we look at a jet and guess what created it? Did it come from a top quark, which is a heavy, important particle, or from a normal, ordinary quark or gluon? This task is called jet tagging.

This matters because physicists use the answer to check if their theories about physics are correct, and to search for new, undiscovered particles. Machine learning has become the standard way to do this, replacing older methods that relied on manually picked physics variables.

## 2. What Dataset Was Found

The standard dataset that almost every paper in this field uses was found and reviewed. It is called the **Top Quark Tagging Reference Dataset**.

Simple facts about it:

- It has 2 million examples of jets, already labeled: this one is a top quark, or this one is not
- It is split into three parts: 1.2 million for training the model, 400k for checking progress during training, and 400k for the final test
- It was made using computer simulation, not real collision data. A program called PYTHIA8 created simulated collisions, and another program called Delphes simulated how a real detector would see them

Why this dataset matters: it is public and free, and almost everyone in this research area uses it. This means results from this project can be compared directly against real published numbers. If the model built in this project gets similar scores to famous papers, that is a sign it was done correctly.

This dataset was chosen over raw ATLAS or CMS Open Data for a practical reason: it is already cleaned, labeled, and split into training and testing sets. Raw detector data would need a lot of extra cleaning and preparation work before any model could even be trained on it.

Dataset link: https://zenodo.org/record/2603256

## 3. Two Ways to Solve This Problem

There are two main approaches used in published research for this problem:

### Approach 1: Turn the jet into a picture (CNN)

A jet can be turned into something like a black-and-white photo, imagine a grid where brighter pixels mean more energy was detected there. Then a normal image-recognition network, called a CNN or Convolutional Neural Network, is used to classify the picture as top quark or not.

This is simpler because CNNs are the most common, well-known type of neural network, with the most tutorials and existing code available. This is the planned starting point, since it is the easier entry point.

### Approach 2: Treat the jet as a group of points (GNN)

Instead of turning the jet into an image, it can be kept as a list of individual particles, like dots scattered in space. Then a Graph Neural Network, or GNN, is used. This is a newer, smarter kind of network built to understand relationships between points, not just pixels.

This approach performs better in every paper reviewed, because it does not lose information the way squeezing everything into a picture does. But it is harder to build.

## 4. What the Comparison of Published Results Showed

One specific paper was found, the one that introduced a model called ParticleNet, that tested many different models on this exact same dataset and reported their scores.

The main number used to judge how good a model is, is called AUC, a score between 0 and 1, where closer to 1 is better. The published numbers found:

| Model | Type | AUC |
|---|---|---|
| P-CNN | Image-based | 0.9803 |
| PFN | Theory-based | 0.9819 |
| ResNeXt-50 | Image-based | 0.9837 |
| ParticleNet-Lite | Graph-based | 0.9844 |
| ParticleNet | Graph-based | 0.9858 |

*Source: Qu & Gouskos, ParticleNet, arXiv:1902.08570, Table II.*

So the point-based approach is better, but only by a small amount, not a huge difference. This is useful, because it shows that even the simpler CNN approach gets decent results, so starting there is not a waste of time.

## 5. The Plan, Based on What Was Found

- Start by building the CNN version first, since it is simpler, well-documented, and has lots of existing code to learn from
- Once that works, try building the GNN version
- Compare both models on the same test data, and explain why one does better than the other

This gives the thesis an actual contribution: not just copying one paper, but comparing two approaches directly and explaining the result, most likely because graph-based models capture particle-to-particle relationships that image-based models lose when everything is squeezed into a picture.

## 6. Existing Code Found on GitHub

Real, working code that other researchers have already shared publicly was reviewed, so this project does not need to start from zero:

| Repository | What it contains |
|---|---|
| [hqucms/ParticleNet](https://github.com/hqucms/ParticleNet) | Official implementation of ParticleNet. Includes the model code and a full training example notebook on the Top Quark Tagging Reference Dataset. |
| [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer) | Official implementation of the Particle Transformer (ParT), the architecture that later surpassed ParticleNet on jet tagging benchmarks. Includes dataset download scripts for the TopLandscape (Top Quark Tagging) dataset. |
| [colizz/weaver-benchmark](https://github.com/colizz/weaver-benchmark) | A CMS Machine Learning group tutorial repository. Walks through training an MLP, a CNN-based tagger, and ParticleNet on the same top-tagging dataset, then compares their performance directly. |
| [niklai99/jet-tagging](https://github.com/niklai99/jet-tagging) | A research project applying machine learning jet-tagging techniques to CMS Open Data, useful as a reference for working with real (not simulated) detector data. |

The Weaver-benchmark repository is especially useful, since it already trains three different models, a simple model, a CNN, and ParticleNet, on the same dataset and compares them directly. This is basically the same comparison this project plans to do, so it works as a teaching example.

## 7. Proposed Working Timeline

![Timeline](timeline_chart.png)

Literature review runs in parallel with early technical work; the GNN model is planned after the CNN baseline is confirmed working.

| Task | Weeks |
|---|---|
| PyTorch fundamentals | 1-2 |
| Data loading + preprocessing | 3-4 |
| Baseline CNN (train/debug) | 4-6 |
| Literature review write-up | 2-7 |
| GNN / point-cloud model | 6-8 |
| Comparison + writing | 8-9 |

## 8. Summary

The Top Quark Tagging Reference Dataset is a well-established, public dataset with a large body of published results, ranging from image-based CNN models to graph-based models like ParticleNet and LorentzNet, and more recently transformer-based models like the Particle Transformer. Image-based models reach an AUC of roughly 0.98, while graph-based point-cloud models push this to about 0.985 to 0.986.

This research report has gone through the key papers, the real benchmark numbers, and the existing open-source code relevant to reproducing and extending this line of work. The next step is building the CNN baseline described in Section 3.

## References

| # | Reference | Type |
|---|---|---|
| 1 | Kasieczka, G., Plehn, T., Thompson, J., and Russell, M. (2019). Top Quark Tagging Reference Dataset. Zenodo. doi:10.5281/zenodo.2603256 | Dataset |
| 2 | Butter, A., Kasieczka, G., Plehn, T., et al. (2019). The Machine Learning Landscape of Top Taggers. SciPost Physics, 7, 014. arXiv:1902.09914 | Benchmark survey paper |
| 3 | Qu, H. and Gouskos, L. (2020). ParticleNet: Jet Tagging via Particle Clouds. Physical Review D, 101, 056019. arXiv:1902.08570 | Model paper (GNN baseline) |
| 4 | Kasieczka, G., Plehn, T., Russell, M., and Schell, T. (2017). Deep-learning Top Taggers or The End of QCD? JHEP. (DeepTop CNN tagger) | Model paper (CNN baseline) |
| 5 | Gong, S., Meng, Q., Zhang, J., et al. (2022). An efficient Lorentz equivariant graph neural network for jet tagging (LorentzNet). JHEP. arXiv:2201.08187 | Model paper (advanced GNN) |
| 6 | Butter, A., Kasieczka, G., Plehn, T., and Russell, M. (2018). Deep-learned Top Tagging with a Lorentz Layer. SciPost Physics, 5, 028. arXiv:1707.08966 | Dataset origin paper |

---

## 9. Implementation

This section documents the actual code that implements the CNN baseline
described in Section 3 (Approach 1). It is written in PyTorch and
organised as a small Python package so that the data, model, training,
and evaluation concerns are kept separate, the way a real codebase
should be.

### Project structure

```
thesis-repo/
├── README.md
├── AGENTS.md            # notes for AI coding agents
├── requirements.txt     # pinned dependencies
├── .gitignore           # ignores data/, checkpoints/, .venv/
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

### Setup

A Python virtual environment is used because the system `pip` is blocked
by PEP 668 on this machine.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### How to run

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

### Tests

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

### Current status

The CNN baseline pipeline runs end-to-end. On a small smoke subset
(4000 jets per split, 3 epochs, CPU only) it reaches:

- Validation AUC: **0.9275**
- Test AUC: **0.9248**
- Background rejection 1/ε_B at ε_S = 0.3: **~41**

These numbers are well below the published P-CNN AUC of 0.9803 (Section
4) because the smoke run uses only ~0.3% of the training set and three
epochs. A full training run on the 1.2M-event training set is the next
step to get a comparable number; on this CPU-only machine that run is
slow, so it has not been completed yet.

### Next steps

1. Run full training (20 epochs, full 1.2M training set) and record the
   test AUC alongside the published P-CNN / ResNeXt numbers.
2. Build the point-cloud / GNN model (Approach 2, ParticleNet-style) as
   a second model under `src/model/` so the two can be compared on the
   same test split.
3. Add a plotting script for ROC curves and the AUC comparison table.
