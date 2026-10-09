<div align="center">

<!-- INTRO VIDEO -->

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
| Gaussian blur | 3×3 kernel, σ = 0.8, applied with p = 0.5 |

λ is selected per backbone from a stability sweep over {0, 0.01, 0.05, 0.1} (e.g. λ = 0.05 for ResNet-18, λ = 0.002 for DenseNet-121).

## Results

**Source domain (Kaggle Brain MRI):** the best configuration reaches **97.71% accuracy** and **97.55% macro-F1**.

**Zero-shot target domain (BRISC-2025):**

| | |
| --- | --- |
| Accuracy gain from FSSR | up to **+8.20 pp** |
| Macro-F1 gain from FSSR | up to **+12.50 pp** |
| DenseNet-121 + FSSR | **96.70% accuracy**, **96.87% macro-F1** |
| Source → target domain gap | only **0.94%** |

Feature-space analysis shows FSSR consistently lowers both the mean and the variance of feature deviation under perturbation across all three backbones. Confusion matrices show less class confusion and steadier recall on the harder tumor categories. Full tables and figures are in the [paper](https://doi.org/10.3390/math14061082).

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
| 6 – 7.5 | Intensity and resolution audit; definition and quantitative check (SSIM/PSNR) of the MRI-safe augmentations |
| 8 | `Dataset` / `DataLoader` (grayscale, 224 × 224) |
| 9 | 1-channel ResNet-18, ResNet-34 and DenseNet-121 backbones that return `(logits, features)` |
| 10 – 10.5 | FSSR loss, gradient-flow check across the λ grid, micro-overfit sanity checks |
| 11 | Feature-stability measurement: CE baseline vs FSSR |
| 12 | Multi-backbone cross-validation with per-epoch logs |
| 14 – 14H | Final training (up to 25 epochs, AMP), full metrics, confusion matrices and error heatmaps (300 DPI TIFF) |
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
