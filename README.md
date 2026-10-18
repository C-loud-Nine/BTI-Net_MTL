# BTI-Net: Bidirectional Decoder-Level Task Interaction for Multi-Task Medical Image Analysis

This repository contains the official implementation submitted for peer review.
Authors and institutional affiliations will be disclosed upon acceptance.

## Overview

BTI-Net jointly learns segmentation and classification from medical images by
establishing bidirectional cross-task communication at every decoder level.
Three modules are proposed:

- **TIM (Task Interaction Module):** bidirectional decoder-level interaction
  between segmentation and classification through channel-wise gated
  operations, applied at all four decoder resolutions.
- **SRG (Supervised Reliability Gate):** a per-instance, per-level gate driven
  by three signals (cross-task alignment, segmentation spatial-gradient
  magnitude, and log(1 + variance) of the enhanced classification vector).
  The two coefficients come from **independent sigmoids**, so they do not sum
  to 1. None of the three signals is a predictive-uncertainty estimate: all are
  computed from a single deterministic forward pass, with no ensemble, no
  dropout sampling and no Bayesian overhead. The gate is trained in a second
  stage against an explicit supervision target derived from per-sample
  segmentation IoU and classification confidence, which is what "supervised"
  refers to.
- **MSCF (Multi-Scale Context Fusion):** parallel dilated separable
  convolutions with instance-adaptive scale weighting.

Evaluated on BUSI (breast ultrasound), HAM10000 (dermoscopy) and BRISC
(brain MRI). Reported numbers live in the paper and are not duplicated here;
`evaluate.py` regenerates them from a checkpoint.

## Repository structure

```
|-- model.py              # BTI-Net assembly (EfficientNetB4 + TIM + SRG)
|-- modules.py            # TIM, SRG, MSCF, AttentionGate, ResidualBlock
|-- train.py              # two-stage training pipeline
|-- loss.py               # focal Dice + focal cross-entropy + gate supervision
|-- config.py             # hyperparameters and dataset paths
|-- busi_dataloader.py    # BUSI loader (numpy arrays)
|-- ham_dataloader.py     # HAM10000 loader (tf.data)
|-- brisc_dataloader.py   # BRISC loader (tf.data)
`-- evaluate.py           # per-image evaluation and gate export
```

## Requirements

```
tensorflow >= 2.10
numpy
opencv-python
scikit-learn
pandas
```

## Datasets

Download and set the paths in `config.py`:

- **BUSI:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- **HAM10000:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **BRISC:** https://www.nature.com/articles/s41597-026-06753-y

Splits are stratified by class label with `random_state = 42`;
BUSI uses `config.TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT = 0.70 / 0.15 / 0.15`,
giving 546 / 117 / 117 images.

**BRISC mask coverage.** BRISC ships 6,000 classification images against 4,793
mask pairs. A tumour-labelled image with no mask file is substituted with an
all-zero mask in `brisc_dataloader.py`, so the overlap loss and every overlap
metric treat it as a confirmed background-only case. This depresses mean IoU
and mean Dice by an amount that depends on how many such cases land in the
test split. `evaluate.py` reports the count of zero-mask test images
separately so the effect is visible. The preferred behaviour, keeping the
classification loss for those cases and excluding them from the overlap
computation, changes the reported BRISC segmentation numbers and therefore
requires a re-run; the released code is left as it was run.

## Training

```bash
python train.py --dataset busi
python train.py --dataset ham
python train.py --dataset brisc
```

Training runs in two stages automatically:

- **Stage 1:** the full network is trained on the segmentation and
  classification losses, up to 50 epochs, with model selection on
  `config.MONITOR = "val_combined"` (0.7 x mean IoU + 0.3 x accuracy),
  maximised.
- **Stage 2:** every layer is frozen except the four SRG gate networks, which
  are fine-tuned for up to 15 epochs on the difficulty-aware gate loss with
  `STAGE2_LAMBDA_GATE = 1.0`. The same gate weight is used for all three
  modalities; there is no per-modality value in this repository. Stage 2
  monitors `val_loss`.

Two properties of the gate supervision are worth stating explicitly:

- the segmentation target is a **within-batch rank**: min-max normalisation
  forces the hardest sample in each batch to 1.0 and the easiest to 0.0,
  whatever their absolute IoU, so the gate learns relative, batch-conditional
  difficulty;
- the classification signal is the **maximum softmax probability**, not the
  probability of the true class, so a confidently wrong prediction counts as
  confident.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/busi_stage2.keras  --dataset busi
python evaluate.py --checkpoint checkpoints/ham_stage2.keras   --dataset ham
python evaluate.py --checkpoint checkpoints/brisc_stage2.keras --dataset brisc
```

`evaluate.py` is additive: it trains nothing and modifies nothing. It writes
`predictions_<tag>.csv` (one row per test image, including all eight gate
coefficients), `confusion_<tag>.csv` and `summary_<tag>.json` to
`config.RESULTS_DIR`.

It is built so the published aggregates can be checked rather than trusted:

- accuracy is recomputed as `100 x (integer correct) / N`, with the grid step
  `100/N` printed, so any reported accuracy must be a multiple of it;
- IoU and Dice come from the same per-image counts at the same threshold, so
  `Dice = 2 IoU / (1 + IoU)` holds per image and the aggregate obeys
  `mean Dice <= 2 mIoU / (1 + mIoU)`. The summary prints that ceiling and
  flags a violation;
- macro-F1 and balanced accuracy are reported, because weighted recall is
  numerically identical to accuracy and so says nothing about minority classes;
- the eight gate coefficients are recomputed as real model outputs through
  `SupervisedReliabilityGate.compute_gates`. They are **not** read back from
  the `_last_w_seg` / `_last_w_clf` layer attributes, which hold the most
  recent call only and therefore cover one batch rather than the test set.

## Hyperparameters

All values live in `config.py`. Key ones:

| Parameter | Value |
|---|---|
| Input resolution | 224 x 224 |
| Encoder | EfficientNetB4 (ImageNet), first 50 layers frozen |
| Decoder channels | 384 / 192 / 96 / 48 |
| Batch size (train / val) | 8 / 16 |
| Initial LR | 3.0e-4, cosine annealing to 1.5e-6 |
| Optimizer | Adam, eps = 1e-7, global clipnorm 1.0 |
| Focal Dice gamma | 0.75 |
| Focal CE gamma | 2.0 |
| TIM modulation tau | 0.7, giving mu in [1, 1.7] |
| Stage-2 epochs / LR / patience | 15 / 3.0e-4 / 5 |
| Stage-2 gate weight | 1.0 (all modalities) |

Note that `l2_lambda=1e-5` is accepted by `enhanced_bti_model` but never used:
there is no `kernel_regularizer` anywhere in `model.py` or `modules.py`, and
the optimizer is constructed without `weight_decay`. No L2 penalty is applied.
The signature keeps the argument for compatibility and the fact is noted
inline.

## License

This code will be released under the MIT License.
