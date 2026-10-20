# BTI-Net: Bidirectional Decoder-Level Task Interaction for Multi-Task Medical Image Analysis

This repository contains the official implementation submitted for peer review.

## Overview

BTI-Net jointly learns segmentation and classification from medical images by
establishing bidirectional cross-task communication at every decoder level.
SRG is the contribution of this work; TIM and MSCF are retained from the
preliminary study (arXiv:2603.01295).

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
  stage against targets derived from soft segmentation IoU and maximum
  classification confidence, which is what "supervised" refers to.
- **MSCF (Multi-Scale Context Fusion):** parallel dilated separable
  convolutions with instance-adaptive scale weighting.

Evaluated on BUSI (breast ultrasound), HAM10000 (dermoscopy) and BRISC
(brain MRI). Reported numbers live in the paper and are not duplicated here;
`evaluate.py` computes the per-run metrics behind them from a checkpoint.

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
|-- evaluate.py           # per-image evaluation and gate export
|-- audit_duplicates.py   # exact / near-duplicate audit against the BUSI manifest
|-- complexity.py         # parameter and FLOP count
`-- requirements.txt
```

## Requirements

```
tensorflow == 2.13 (the Stage 1 task-weight behaviour described below is that of TF 2.13)
numpy
opencv-python
albumentations
scikit-learn
pandas
```

Exact versions: `pip install -r requirements.txt`.

## Datasets

Download and set the paths in `config.py`:

- **BUSI:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- **HAM10000:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **BRISC:** https://www.nature.com/articles/s41597-026-06753-y

All partitions are fixed with `random_state = 42`; only the training seed
varies between runs. BUSI is stratified by class label with
`config.TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT = 0.70 / 0.15 / 0.15`, giving
545 / 118 / 117 images (`results/split_manifest_busi.csv`; test first, then validation from the remainder). HAM10000 is split
70 / 15 / 15 by lesion identifier, not by image or class, with disjointness
asserted (`results/ham_split_manifest.csv`). BRISC uses its official
5,000 / 1,000 split with a 15% stratified validation carve-out from the
training pool.

**BRISC mask coverage.** BRISC ships 6,000 classification images; its
segmentation release provides 4,793 masks for the tumour classes, and the 1,207
non-tumorous images have none. `brisc_dataloader.py` loads an all-zero mask for
any image without a mask file and logs the count per class, warning for any
tumour-labelled image that lacks one.

## Controls and audits

| Purpose | How |
|:--|:--|
| Gate controls | `config.GATE_MODE`: `srg` (reported), `off` (TIM output used directly), `fixed` (0.5 / 0.5), `constant` (two learned per-level constants); auxiliary objective off with `STAGE2_LAMBDA_GATE = 0`. `off` and `fixed` have no gate parameters, so Stage 2 is skipped for them |
| Seeds | `python train.py --dataset busi --seed 1`; the split seed stays `config.RANDOM_STATE`; outputs are `checkpoints/final_model_ft_<dataset>_<gate_mode>_lg<lambda>_s<seed>.keras` (`stage1_...` for `off` and `fixed`) and `results/test_results_<dataset>_<gate_mode>_lg<lambda>_s<seed>.txt`, and each run is appended to `results/run_ledger.csv` |
| Input scale | `config.INPUT_SCALE`: `unit` (reported runs; the Keras EfficientNet-B4 rescales again) or `raw` |
| Task weights | Stage 1 weights stay at the compile-time 0.82 / 0.18 in TF 2.13; the callback in `train.py` does not reach the compiled loss. Stage 2 uses unit task weights |
| Duplicate audit (BUSI) | `python audit_duplicates.py --root <BUSI dir> --manifest results/split_manifest_busi.csv` |
| HAM10000 partition | `results/ham_split_manifest.csv`, lesion-disjoint, with lesion counts in the log |
| Class-wise metrics | `evaluate.py` reports macro-F1, balanced accuracy and per-class precision, recall and F1 |
| Cost | `python complexity.py` (parameters, SRG share, FLOPs at 224 x 224) |

## Training

```bash
python train.py --dataset busi
python train.py --dataset ham
python train.py --dataset brisc
```

Training runs in two stages automatically:

- **Stage 1:** all layers except the first 50 encoder layers are trained on
  the segmentation and classification losses, up to 50 epochs; early stopping
  (patience 22, min_delta 0.004) monitors `config.MONITOR = "val_combined"`
  (0.7 x IoU + 0.3 x accuracy, IoU per `config.SELECTION_IOU`, maximised) and
  restores the best weights when it stops.
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
python evaluate.py --checkpoint checkpoints/final_model_ft_busi_srg_lg1_s42.keras  --dataset busi
python evaluate.py --checkpoint checkpoints/final_model_ft_ham_srg_lg1_s42.keras   --dataset ham
python evaluate.py --checkpoint checkpoints/final_model_ft_brisc_srg_lg1_s42.keras --dataset brisc
```

`evaluate.py` is additive: it trains nothing and modifies nothing. It writes
`predictions_<tag>.csv` (one row per test image with its file name, including
all eight gate coefficients), `confusion_<tag>.csv` and `summary_<tag>.json` to
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
| Batch size | 8 (training, validation and test pipelines); `evaluate.py --batch-size` (default 16) applies to BUSI only |
| Initial LR | 3.0e-4, cosine annealing with plateau reduction (factor 0.55, patience 7) to 1.5e-6 |
| Optimizer | Adam, beta1 = 0.91 in Stage 1 (Adam default 0.9 in Stage 2), eps = 1e-7, global clipnorm 1.0 |
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
