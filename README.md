# BTI-Net: Bidirectional Decoder-Level Task Interaction for Multi-Task Medical Image Analysis

This repository contains the official implementation submitted for peer review.
Authors and institutional affiliations will be disclosed upon acceptance.

## Overview

BTI-Net jointly learns segmentation and classification from medical images by
establishing bidirectional cross-task communication at every decoder level.
Two core modules are proposed:

- **TIM (Task Interaction Module):** Bidirectional decoder-level interaction
  between segmentation and classification via channel-wise gated operations,
  with features propagated progressively across all four decoder resolutions.
- **UPA (Uncertainty Proxy Attention):** Per-instance, per-level adaptive gate
  using three uncertainty signals (cross-task alignment, segmentation gradient
  energy, classification activation spread), trained without external annotations
  or Bayesian overhead.

Evaluated on BUSI (breast ultrasound), HAM10000 (dermoscopy), and BRISC (brain MRI).

## Repository Structure

```
├── model.py              # BTI-Net model assembly (EfficientNetB4 + TIM + UPA)
├── modules.py            # TIM, UPA, MSCF, AttentionGate, ResidualBlock
├── train.py              # Two-stage training pipeline
├── loss.py               # Focal Tversky + Focal Cross-Entropy + gate loss
├── config.py             # Hyperparameters and dataset paths
├── busi_dataloader.py    # BUSI dataset loader
├── ham_dataloader.py     # HAM10000 dataset loader
└── brisc_dataloader.py   # BRISC dataset loader
```

## Requirements

```
tensorflow >= 2.10
numpy
opencv-python
scikit-learn
```

## Datasets

Download the datasets and set the paths in `config.py`:

- **BUSI:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- **HAM10000:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **BRISC:** https://www.nature.com/articles/s41597-026-06753-y

## Training

```bash
# Train on BUSI
python train.py --dataset busi

# Train on HAM10000
python train.py --dataset ham

# Train on BRISC
python train.py --dataset brisc
```

Training runs in two stages automatically:

- **Stage 1:** Full network trained on segmentation + classification losses
  (up to 50 epochs, early stopping on validation IoU).
- **Stage 2:** All layers frozen except UPA gate networks; gate fine-tuned on
  difficulty-aware loss (up to 15 epochs).

## Hyperparameters

All hyperparameters are in `config.py`. Key values:

| Parameter | Value |
|---|---|
| Input resolution | 224 × 224 |
| Encoder | EfficientNetB4 (ImageNet) |
| Decoder channels | 384 / 192 / 96 / 48 |
| Batch size | 8 |
| Initial LR | 3 × 10⁻⁴ |
| Focal Tversky γ | 0.75 |
| Focal CE γ | 2.0 |
| TIM modulation τ | 0.7 |

## License

This code will be released under the MIT License.
