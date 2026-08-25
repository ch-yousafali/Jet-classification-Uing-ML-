# CNN vs. ParticleNet — Side-by-Side Comparison on the Top Quark Tagging Reference Dataset

All published numbers below are taken **only** from official sources:

- **ParticleNet paper**: Qu & Gouskos, "Jet Tagging via Particle Clouds,"
  Phys. Rev. D **101**, 056019 (2020), arXiv:1902.08570 — Table 2 (performance)
  and Table 4 (model complexity).
- **Top Tagging Landscape paper**: Butter, Kasieczka, Plehn et al.,
  "The Machine Learning Landscape of Top Taggers," SciPost Phys. **7**, 014 (2019),
  arXiv:1902.09914.

Every published model was trained on the **full 1.2M-jet training set** and
evaluated on the **full 400k-jet test set**, with results reported as the median
of 9 independent trainings (uncertainty = spread across those 9 runs). Our CNN
was **not** trained on the full set — see the caveat below.

---

## 1. Comparison Table

| Model | Type | Accuracy | AUC | 1/ε_B @ ε_S=50% | 1/ε_B @ ε_S=30% | Params | Source |
|---|---|---|---|---|---|---|---|
| **Our CNN (smoke, 3 ep)** | Jet-image 2D CNN | not logged | 0.9248 | not logged | ~41 | ~913k | This repo (README) |
| **Our CNN (ckpt, 1 ep)** | Jet-image 2D CNN | 0.8083 | 0.8912 | not logged | 16.8 | ~913k | This repo (eval run) |
| ResNeXt-50 | Jet-image 2D CNN (deep) | 0.936 | 0.9837 | 302 ± 5 | 1147 ± 58 | 1.46M | ParticleNet paper, Table 2/4 |
| P-CNN | 1D particle-sequence CNN | 0.930 | 0.9803 | 201 ± 4 | 759 ± 24 | 348k | ParticleNet paper, Table 2/4 |
| PFN | Particle-set (Deep Sets) | not reported | 0.9819 | 247 ± 3 | 888 ± 17 | 82k | ParticleNet paper, Table 2/4 |
| ParticleNet-Lite | Point-cloud / graph (EdgeConv) | 0.937 | 0.9844 | 325 ± 5 | 1262 ± 49 | 26k | ParticleNet paper, Table 2/4 |
| **ParticleNet** | Point-cloud / graph (EdgeConv) | 0.940 | 0.9858 | 397 ± 7 | 1615 ± 93 | 366k | ParticleNet paper, Table 2/4 |

Notes on the table:
- "not logged" = the metric was not recorded for that run; we do not estimate it.
- "not reported" = the source paper does not report that number (PFN accuracy is
  absent from the ParticleNet paper's Table 2).
- Our CNN's two rows come from **two different short runs**, explained in Section 2.

---

## 2. Important Caveat About Our CNN Numbers

Our CNN has **not** been fully trained. The numbers above come from two
CPU-only smoke runs on a tiny subset of the data, **not** the full 1.2M-jet
training set that every published model used:

- **"Our CNN (smoke, 3 ep)"** — the run documented in the repo README: 3 epochs,
  4000 jets per split (~0.3% of the training set). Results: val AUC 0.9275,
  test AUC 0.9248, 1/ε_B @ ε_S=0.3 ≈ 41. Accuracy was not logged for this run.
  The checkpoint from this run is **no longer on disk** (it was overwritten).

- **"Our CNN (ckpt, 1 ep)"** — the checkpoint currently saved at
  `checkpoints/cnn_best.pt` is from a **shorter 1-epoch** run (val AUC 0.9093).
  Re-evaluating it on 4000 test jets gives test AUC 0.8912, accuracy 0.8083,
  1/ε_B @ ε_S=0.3 = 16.8. These are lower than the README numbers because the
  checkpoint is from fewer epochs.

A full training run (20 epochs on the full 1.2M-jet set) is the planned next
step but has not been completed because this machine is CPU-only and the run
would be slow. **Until that run is done, our CNN numbers are not directly
comparable to the published numbers**, which all use the full dataset.

---

## 3. How Our CNN Is Architecturally Different From ParticleNet

Our CNN renders each jet as a fixed 2D image on a (η, φ) pixel grid and applies
standard 2D convolutions over that grid, exactly like an image classifier — it
forces the irregular, sparse spray of ~100 particles into a regular raster,
which discards per-particle information (any two particles landing in the same
pixel are summed) and leaves >90% of the pixels blank. ParticleNet instead
treats the jet as an unordered "particle cloud": it keeps each constituent as a
distinct point with its own kinematic features, builds a k-nearest-neighbor
graph in (Δη, Δφ) space, and applies EdgeConv operations that learn from the
*relationships between neighboring particles* rather than from fixed pixel
patches. Crucially, ParticleNet dynamically rebuilds that graph after each
EdgeConv block using the learned feature space, so the "neighborhood" evolves
layer by layer, and the whole architecture is permutation-invariant by
construction — something a grid-based CNN is not. In short, the CNN learns
spatial patterns of energy deposition on a grid, while ParticleNet learns
particle-to-particle relational structure directly, which is why it reaches
higher AUC and dramatically higher background rejection (1615 vs. ~41 at our
smoke-test scale, and 1147 for the strongest image-based model, ResNeXt-50).

---

## 4. Does Our CNN Underperform? — Yes, and That Is Expected

Our CNN **underperforms every published model in the table**, including the
image-based ResNeXt-50 and P-CNN, by a wide margin. This is expected and does
not indicate a bug, for two reasons:

1. **Training scale.** Our numbers are from a 4000-jet, 3-epoch smoke run
   (~0.3% of the training data). Every published number uses the full 1.2M-jet
   training set with a tuned learning-rate schedule over many epochs. A
   fully-trained image CNN on this dataset reaches AUC ~0.98 (see ResNeXt-50
   and P-CNN), so the gap would narrow substantially with full training — but
   it would still fall short of ParticleNet.

2. **Architecture.** Even with full training, a grid-based CNN is expected to
   trail a point-cloud/graph model like ParticleNet, because the image
   representation is lossy (pixel binning, sparsity) and lacks the
   permutation-invariant, relational reasoning that EdgeConv provides. The
   ParticleNet paper confirms this: its strongest image-based baseline
   (ResNeXt-50, AUC 0.9837) is still beaten by ParticleNet (AUC 0.9858) with
   4× fewer parameters, and ParticleNet's background rejection at ε_S=30% is
   ~40% higher than ResNeXt-50's.

The contribution of this thesis is **the CNN-vs-GNN comparison itself** —
building both a jet-image CNN and a ParticleNet-style point-cloud model on the
same dataset and explaining *why* the graph approach wins — not beating the
state of the art with a CNN baseline.

---

## 5. Sources

| # | Source | What it provides |
|---|---|---|
| 1 | Qu & Gouskos, "Jet Tagging via Particle Clouds," PRD 101, 056019 (2020), arXiv:1902.08570 | Table 2 (accuracy, AUC, 1/ε_B @ ε_S=50%/30%) and Table 4 (params, inference time) for ResNeXt-50, P-CNN, PFN, ParticleNet-Lite, ParticleNet |
| 2 | Butter, Kasieczka, Plehn et al., "The Machine Learning Landscape of Top Taggers," SciPost Phys. 7, 014 (2019), arXiv:1902.09914 | Broad survey of top taggers on the same dataset; corroborates the AUC ordering above |
| 3 | This repo — `README.md`, `checkpoints/cnn_history.json`, `src/evaluate.py` | Our CNN's smoke-test results and the re-evaluation of the saved checkpoint |

No ParticleNet/Weaver numbers were estimated or invented. Any metric not
published in the sources above is marked "not reported" / "not logged" rather
than guessed.
