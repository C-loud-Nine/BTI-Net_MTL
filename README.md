# BTI-Net

Code for "BTI-Net: Bidirectional Task Interaction With Counterfactually Supervised Gating for Joint Lesion
Segmentation and Classification".

## Setup

```bash
git clone https://github.com/C-loud-Nine/BTI-Net_MTL.git
cd BTI-Net_MTL
pip install -r requirements.txt
```

## Data

```bash
python scripts/download_datasets.py
```

Downloads BUSI, BRISC and HAM10000 into `data/raw/` (Kaggle API token required). The partitions are in `splits/`.

## Training and testing

```bash
python train_btinet.py BUSI 42 results/btinet/BUSI/full/seed42                 # BTI-Net
python train_btinet.py BUSI 42 results/ablations/BUSI/no_tim/seed42 no_tim     # an ablation
python train_compared.py BUSI mtan 42 results/multitask/BUSI/mtan/seed42       # a compared method
bash scripts/run_experiments.sh                                                # every run
```

Each run writes `results.json`, the training history, the per-image test predictions and the confusion matrix.

| Code | Contents |
|---|---|
| `btinet/model.py` | BTI-Net |
| `btinet/protocol.py` | training protocol shared by all models |
| `btinet/evaluation.py` | validation score, test metrics, complexity |
| `methods/` | the compared methods |

## Citation

```bibtex
@misc{shafi2026btinet,
  author  = {Shafi, Abdullah Al and Zunayed, Md Kawsar Mahmud Khan and Suma, Sumaiya Rahim and Ahmmed, Safin and
             Hossain, Sk. Imran and Mephu Nguifo, Engelbert},
  title   = {{BTI-Net}: Bidirectional Task Interaction With Counterfactually Supervised Gating for Joint Lesion
             Segmentation and Classification},
  year    = {2026}
}
```

## License

MIT. The datasets keep their own licences.
