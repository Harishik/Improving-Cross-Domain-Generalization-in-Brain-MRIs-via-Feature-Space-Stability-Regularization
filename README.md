<div align="center">

https://github.com/user-attachments/assets/86d0c26b-c0d3-4b89-9203-61b8f636cbaf

# Improving Cross-Domain Generalization in Brain MRIs via Feature Space Stability Regularization

**FSSR** — a lightweight, model-agnostic training objective that keeps CNN feature representations stable under MRI-safe intensity perturbations, so brain tumor classifiers hold up on unseen clinical data.

[![Paper](https://img.shields.io/badge/Paper-Mathematics%202026-1f6feb)](https://doi.org/10.3390/math14061082)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fmath14061082-blue)](https://doi.org/10.3390/math14061082)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Paper License](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

[Paper](https://doi.org/10.3390/math14061082) · [Method](#method) · [Results](#results) · [Getting started](#getting-started) · [Citation](#citation)

</div>

---

## Overview

Deep learning models for brain tumor classification from MRI often reach high accuracy on the dataset they were trained on, then drop sharply on data from other hospitals. Scanners, acquisition protocols and intensity distributions all differ, and the learned features shift with them.

Most existing fixes scale up the architecture or regularize parameters. Neither directly constrains how stable the **learned feature representations** are. FSSR does. It adds an auxiliary loss that pulls together the embedding of an MRI slice and the embedding of an intensity-perturbed copy, trained alongside standard cross-entropy.

- **Model-agnostic.** Works with any backbone that exposes a feature vector. Evaluated on ResNet-18, ResNet-34 and DenseNet-121.
- **Lightweight.** One extra forward pass per batch and a single scalar weight λ. No architectural changes, no target-domain data.
- **Zero-shot generalization.** Trained only on the Kaggle Brain MRI dataset and evaluated, without retraining or fine-tuning, on the fully unseen **BRISC-2025** dataset.

## Method

For an input image $x$ and an MRI-safe perturbation $\tilde{x} = \mathcal{A}(x)$, a backbone $f_\theta$ produces embeddings $z = f_\theta(x)$ and $\tilde{z} = f_\theta(\tilde{x})$. A linear head $g_\phi$ classifies $z$. The training objective is:

$$
\mathcal{L}_{\text{FSSR}} \;=\; \underbrace{\mathcal{L}_{\text{CE}}\big(g_\phi(z),\, y\big)}_{\text{supervision}} \;+\; \lambda \cdot \underbrace{\frac{1}{B}\sum_{i=1}^{B} \big\lVert z_i - \tilde{z}_i \big\rVert_2}_{\text{feature stability}}
$$

```text
            ┌──────────────┐     z      ┌────────────┐
  x  ─────▶ │              │ ─────────▶ │ classifier │ ──▶ CE(·, y)
            │   backbone   │            └────────────┘
  A(x) ───▶ │  (shared θ)  │ ── z̃ ──┐
            └──────────────┘        └──▶ λ · ‖z − z̃‖₂
```

**MRI-safe perturbations** $\mathcal{A}$ (deliberately mild, intensity-only, no geometric changes):

| Perturbation | Setting |
| --- | --- |
| Intensity scaling | uniform factor in [0.9, 1.1] |
| Gaussian noise | σ = 1% of the image standard deviation |
| Spatial smoothing | 3×3 average pooling, applied with p = 0.5 |

λ = 0.05 for all backbones, selected from a sweep over {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}. The ablation in the paper shows the full combination (scale + noise + smoothing) beats every single component and every pair.

### Training setup

| Setting | Value |
| --- | --- |
| Input | 224 × 224, grayscale, area-based resizing |
| Initialization | Random (trained from scratch) |
| Optimizer | AdamW, learning rate 1e-4 |
| Batch size | 32 |
| Epochs | up to 25, early stopping with patience 5 |
| FSSR weight λ | 0.05 |

## Results

All models are trained only on Kaggle Brain MRI and evaluated zero-shot on BRISC-2025. Domain gap = Kaggle accuracy − BRISC-2025 accuracy.

| Backbone | Method | Kaggle Acc | Kaggle F1 | BRISC Acc | BRISC F1 | Domain gap ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | Baseline (CE) | 95.50 | 95.15 | 90.50 | 89.38 | 5.00 |
| | SimCLR | 94.28 | 93.90 | 85.40 | 81.72 | 8.88 |
| | MixStyle | 91.30 | 90.72 | 85.60 | 84.29 | 5.70 |
| | **FSSR (ours)** | 95.27 | 94.91 | 90.50 | **89.49** | **4.77** |
| ResNet-34 | Baseline (CE) | 95.73 | 95.42 | 85.50 | 80.12 | 10.23 |
| | SimCLR | 93.52 | 93.15 | 86.00 | 83.47 | 7.52 |
| | MixStyle | 93.90 | 93.55 | 87.00 | 84.54 | 6.90 |
| | **FSSR (ours)** | **97.71** | **97.55** | **93.70** | **92.62** | **4.01** |
| DenseNet-121 | Baseline (CE) | 96.03 | 95.75 | 94.20 | 94.32 | 1.83 |
| | SimCLR | 93.75 | 93.40 | 91.10 | 90.56 | 2.65 |
| | MixStyle | 92.37 | 92.12 | 89.10 | 89.06 | 3.27 |
| | **FSSR (ours)** | **97.64** | **97.47** | **96.70** | **96.87** | **0.94** |

**Highlights**

- **+8.20 pp** BRISC-2025 accuracy and **+12.50 pp** macro-F1 for ResNet-34 (accuracy gain 95% CI [6.10, 10.30], p < 0.001).
- **96.70%** accuracy and **96.87%** macro-F1 on unseen data with DenseNet-121, a domain gap of only **0.94%**.
- Better calibration on the target domain: ResNet-34 ECE drops from 0.1166 to 0.0400, and DenseNet-121 from 0.0325 to 0.0166.

**Feature stability:** mean ‖z − z̃‖₂ deviation under perturbation.

| Backbone | Baseline mean / std | FSSR mean / std | Δ mean | Δ std |
| --- | --- | --- | ---: | ---: |
| ResNet-18 | 4.513 / 5.182 | 3.602 / 2.467 | −20.2% | −52.4% |
| ResNet-34 | 4.506 / 5.136 | 3.499 / 2.313 | −22.3% | −55.0% |
| DenseNet-121 | 4.565 / 5.157 | 3.474 / 2.168 | −23.9% | −58.0% |

Full tables, ablations, λ sweeps, confusion matrices and calibration analysis are in the [paper](https://doi.org/10.3390/math14061082).

## Datasets

| Dataset | Role | Classes |
| --- | --- | --- |
| Kaggle Brain MRI | Training and in-domain evaluation (stratified 80/20 split, 3-fold CV) | glioma · meningioma · pituitary · no tumor |
| BRISC-2025 | Fully unseen external test set (zero-shot) | glioma · meningioma · pituitary · no tumor |

Images are converted to single-channel grayscale and resized to 224 × 224. No intensity normalization is baked into the pipeline, so the stability loss sees raw intensity variation.

## Repository structure

```text
.
├── src/
│   └── main.py          # End-to-end pipeline (exported from the Colab notebook)
├── requirements.txt
├── CITATION.cff
└── README.md
```

`src/main.py` is organised as numbered steps that mirror the experiments in the paper:

| Step | What it does |
| --- | --- |
| 1 – 2 | Environment checks, Kaggle ZIP audit and extraction |
| BRISC scan | Class discovery and integrity checks for BRISC-2025 |
| 5A – 5C | Label discovery, stratified 80/20 split, 3-fold `StratifiedKFold` manifests |
| 6 – 7.5 | Intensity and resolution audit; exploratory augmentations with an SSIM/PSNR check |
| 8 | `Dataset` / `DataLoader` (grayscale, 224 × 224) |
| 9 | 1-channel ResNet-18, ResNet-34 and DenseNet-121 backbones that return `(logits, features)` |
| 10 – 10.5 | FSSR loss, gradient-flow check across the λ grid, micro-overfit sanity checks |
| 11 | Feature-stability measurement: CE baseline vs FSSR |
| 12 | Multi-backbone cross-validation with per-epoch logs (final perturbation set with 3×3 average-pool smoothing) |
| 14 – 14H | Final training (AdamW, up to 25 epochs, early stopping, AMP), full metrics, confusion matrices and error heatmaps (300 DPI TIFF) |
| 15 | BRISC-2025 cross-validation and zero-shot external evaluation |

## Getting started

The pipeline was developed in **Google Colab** with a GPU runtime and datasets stored on Google Drive.

```bash
git clone https://github.com/Harishik/Improving-Cross-Domain-Generalization-in-Brain-MRIs-via-Feature-Space-Stability-Regularization.git
cd Improving-Cross-Domain-Generalization-in-Brain-MRIs-via-Feature-Space-Stability-Regularization
pip install -r requirements.txt
```

1. Download the Kaggle Brain MRI dataset as a ZIP and the BRISC-2025 classification set.
2. Place them where the script expects them, or edit the path constants at the top of each step:

   ```python
   KAGGLE_ZIP = "/content/drive/MyDrive/project dataset/kaggle MRI.zip"
   BRISC_ROOT = "/content/drive/MyDrive/project dataset/brisc_2025"
   SAVE_ROOT  = "/content/drive/MyDrive/project dataset/outputs_fssr/"
   ```

3. Run the steps in order. In Colab, paste the sections into cells; locally, run `python src/main.py` once the paths point to your data.

The core of the method is small enough to drop into any training loop:

```python
def fssr_losses(model, x, y, lam):
    logits, f = model(x, return_features=True)
    logits_aug, f_aug = model(augment_batch_light(x), return_features=True)

    ce = F.cross_entropy(logits, y)
    stab = torch.norm(f - f_aug, p=2, dim=1).mean()   # feature stability
    return ce, stab, ce + lam * stab
```

## Authors

| | Author | Affiliation |
| --- | --- | --- |
| <a href="https://github.com/kakon2002"><img src="https://github.com/kakon2002.png" width="56" alt="kakon2002"/></a> | **Shawon Chakrabarty Kakon** · [@kakon2002](https://github.com/kakon2002) · [ORCID](https://orcid.org/0009-0005-5548-9805) | Dept. of Artificial Intelligence and Big Data, Woosong University, Daejeon, Republic of Korea |
| <a href="https://github.com/Harishik"><img src="https://github.com/Harishik.png" width="56" alt="Harishik"/></a> | **Harishik Dev Singh Jamwal** · [@Harishik](https://github.com/Harishik) · [ORCID](https://orcid.org/0009-0007-6781-2803) | Dept. of Artificial Intelligence and Big Data, Woosong University, Daejeon, Republic of Korea |
| | **Saurabh Singh** · [ORCID](https://orcid.org/0000-0003-1118-9569) | Dept. of Artificial Intelligence and Big Data, Woosong University, Daejeon, Republic of Korea |

## Citation

If you use this work, please cite:

```bibtex
@article{kakon2026fssr,
  title   = {Improving Cross-Domain Generalization in Brain MRIs via Feature Space Stability Regularization},
  author  = {Kakon, Shawon Chakrabarty and Jamwal, Harishik Dev Singh and Singh, Saurabh},
  journal = {Mathematics},
  volume  = {14},
  number  = {6},
  pages   = {1082},
  year    = {2026},
  doi     = {10.3390/math14061082}
}
```

> Kakon, S. C., Jamwal, H. D. S., & Singh, S. (2026). Improving Cross-Domain Generalization in Brain MRIs via Feature Space Stability Regularization. *Mathematics*, 14(6), 1082. https://doi.org/10.3390/math14061082

## License

The paper is published open access under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Datasets are subject to their own licenses.
