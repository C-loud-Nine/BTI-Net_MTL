# BTI-Net: Bidirectional Decoder-Level Task Interaction with Supervised Reliability Gating

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation.

---

## Overview

Multi-task models for medical images typically share an encoder, so the
segmentation and classification branches stop exchanging information once their
decoders separate. BTI-Net restores that exchange during decoding and makes its
strength an explicit, supervised quantity.

- **TIM** (Task Interaction Module) exchanges information between the
  segmentation decoder and the classification branch at each of four decoder
  levels.
- **SRG** (Supervised Reliability Gate) produces one scalar coefficient per
  task, per level and per image, interpolating between pre-interaction and
  post-interaction features. It is driven by three signals: cross-task
  alignment, segmentation spatial complexity, and classification activation
  spread. Unlike blending weights learned only through the task objectives, the
  gate is trained with an auxiliary objective against per-sample task
  performance, in a second stage with the backbone frozen.
- **MSCF** (Multi-Scale Context Fusion) applies three dilated separable
  convolutions with softmax scale competition.

Evaluated on BUSI (breast ultrasound), HAM10000 (dermoscopy) and BRISC
(brain MRI).

---

## Results

| Dataset | IoU | Dice | Accuracy |
|---|---|---|---|
| BUSI | 74.50 | 85.30 | 93.16 |
| HAM10000 | 89.80 | 94.20 | 87.84 |
| BRISC | 76.74 | 84.50 | 99.10 |

Mean over three seeds. Segmentation is foreground IoU and Dice at a threshold
of 0.5, computed on positive-mask cases; classification is over all test cases.

### Component ablation (BUSI)

| MSCF | TIM | SRG | IoU | Accuracy |
|:---:|:---:|:---:|---|---|
| – | – | – | 67.43 | 84.62 |
| ✓ | – | – | 69.95 | 88.89 |
| – | ✓ | – | 69.20 | 86.32 |
| ✓ | ✓ | – | 72.14 | 90.84 |
| – | ✓ | ✓ | 70.48 | 91.47 |
| **✓** | **✓** | **✓** | **74.50** | **93.16** |

Rows 4 and 6 differ only in SRG and isolate it: +2.36 IoU and +2.32 accuracy.

---

## Installation

```bash
git clone https://github.com/C-loud-Nine/BTI-Net.git
cd BTI-Net
pip install -r requirements.txt
```

## Data

Download each dataset and set the paths in `config.py`.

- **BUSI:** https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- **HAM10000:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **BRISC:** https://www.nature.com/articles/s41597-026-06753-y

```
data/
├── busi/      # benign/ malignant/ normal/
├── ham/       # HAM10000_images_part_1/, part_2/, segmentations/, metadata.csv
└── brisc/     # classification_task/, segmentation_task/
```

HAM10000 partitions are made lesion-wise, with an explicit check that no lesion
appears in more than one partition. BRISC uses the official train and test
partition, with a stratified validation split carved from the training pool.
BUSI carries no patient identifiers, so its partitions are image-level.

## Usage

```bash
# training runs both stages automatically
python train.py --dataset busi
python train.py --dataset ham
python train.py --dataset brisc

# reported metrics
python evaluate.py --dataset busi \
                   --weights checkpoints/final_model_ft_busi.keras

# add the gate-as-failure-signal analysis
python evaluate.py --dataset busi \
                   --weights checkpoints/final_model_ft_busi.keras \
                   --failure-detection
```

**Stage 1** trains the full network under the segmentation and classification
losses. **Stage 2** freezes every parameter except the four gate networks and
fine-tunes them against the auxiliary gate objective.

`evaluate.py` produces the numbers in the tables above. `MeanIoU` is also
printed during training, but it averages foreground and background IoU on
unthresholded outputs, so use `evaluate.py` for reporting.

---

## Repository layout

```
├── config.py             # hyperparameters and dataset paths
├── modules.py            # MSCF, DualPathAttention, ResidualBlock, AttentionGate, TIM, SRG
├── model.py              # EfficientNetB4 encoder, 4-level decoder, dual heads
├── loss.py               # focal Tversky + boundary + texture; focal CE; gate loss
├── train.py              # two-stage training pipeline
├── evaluate.py           # reported metrics and failure-detection analysis
├── busi_dataloader.py    # BUSI
├── ham_dataloader.py     # HAM10000 (lesion-wise partitions)
└── brisc_dataloader.py   # BRISC (official split)
```

## Key hyperparameters

| Parameter | Value |
|---|---|
| Input resolution | 224 × 224 |
| Encoder | EfficientNetB4 (ImageNet) |
| Decoder channels | 384 / 192 / 96 / 48 |
| Batch size | 8 |
| Stage 1 / Stage 2 epochs | 50 / 15 |
| Initial learning rate | 3 × 10⁻⁴ |
| Focal Tversky γ | 0.75 |
| Focal cross-entropy γ | 2.0 |
| TIM modulation τ | 0.7 |
| Stage 2 gate weight | 1.0 |

All values are in `config.py`.

---

## Notes on the implementation

Stated so that the code and the paper can be read together.

- **The TIM modulation is one-sided.** It is `1 + 0.7 * sigmoid(.) * sigmoid(.)`,
  so the factor lies in `[1, 1.7]`: a channel is left unchanged or amplified,
  never attenuated. Reversion toward the original representation is achieved by
  SRG driving its coefficient toward zero, not by the modulation itself.
- **The gate is supervised toward sample difficulty**, using inverted
  per-sample soft-IoU and prediction confidence as targets. Difficulty is not
  the same quantity as the benefit of interaction for that sample; a
  counterfactual target would align more closely with the mechanism and is
  identified in the paper as the natural next step.
- **Gate targets are min-max normalised within the mini-batch**, so a sample's
  target depends on its batch composition.
- **The three gate signals are activation statistics**, not estimates of
  predictive uncertainty.
- **The boundary loss term is computed on binarised masks.**

## Citation

```bibtex
@article{btinet2026,
  title   = {{BTI-Net}: Bidirectional decoder-level task interaction with
             supervised reliability gating for multi-task medical image analysis},
  author  = {Al Shafi, Abdullah and Zunayed, Md Kawsar Mahmud Khan and
             Ahmmed, Safin and Hossain, Sk Imran and Mephu Nguifo, Engelbert},
  journal = {arXiv preprint arXiv:2606.29102},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
