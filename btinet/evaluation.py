"""Validation score, test metrics and model complexity."""
import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from sklearn.metrics import roc_auc_score

from .protocol import CUDA, DEV, amp

EVAL_BATCH = 32
THR = 0.5
_STRUCT = generate_binary_structure(2, 1)


def to_device(d, idx):
    X = torch.from_numpy(np.ascontiguousarray(d["X"][idx])).to(DEV).permute(0, 3, 1, 2).contiguous()
    M = torch.from_numpy((np.asarray(d["M"][idx]) > 0).astype(np.uint8)).to(DEV)[:, None]
    Y = torch.from_numpy(np.asarray(d["Y"][idx], dtype=np.int64)).to(DEV)
    return X, M, Y


def channels_last(x):
    return x.contiguous(memory_format=torch.channels_last) if CUDA else x


def _bacc(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


@torch.no_grad()
def val_metrics(model, Xv, Mv, Yv, tasks=("seg", "clf")):
    """Validation score 0.7 pooled IoU + 0.3 balanced accuracy (the pooled IoU or the balanced accuracy alone for a
    single-task model)."""
    model.eval()
    inter = union = torch.zeros((), device=DEV)
    pr = []
    for i in range(0, len(Xv), EVAL_BATCH):
        x, m = channels_last(Xv[i:i + EVAL_BATCH].float()), Mv[i:i + EVAL_BATCH].float()
        with amp():
            o = model(x)
        pb = (o["seg"].float() > THR).float()
        inter = inter + (pb * m).sum()
        union = union + ((pb + m) > 0).float().sum()
        pr.append(o["clf"].argmax(1))
    j = float(inter / union.clamp_min(1))
    ba = _bacc(torch.cat(pr).cpu().numpy(), Yv.cpu().numpy())
    score = {("seg",): j, ("clf",): ba}.get(tuple(tasks), 0.7 * j + 0.3 * ba)
    return {"score": score, "IoU pooled": 100 * j, "Bal. Acc": 100 * ba}


@torch.no_grad()
def predict(model, X):
    model.eval()
    segs, clfs = [], []
    for i in range(0, len(X), EVAL_BATCH):
        with amp():
            o = model(channels_last(X[i:i + EVAL_BATCH].float()))
        segs.append(o["seg"].float()[:, 0].cpu())
        clfs.append(o["clf"].float().cpu())
    return torch.cat(segs).numpy(), torch.cat(clfs).numpy()


def hd95(gt, pr):
    """95th-percentile Hausdorff distance in pixels; an empty prediction scores the image diagonal."""
    if not pr.any():
        return float(np.hypot(*gt.shape))
    sg = gt & ~binary_erosion(gt, _STRUCT, border_value=0)
    sp = pr & ~binary_erosion(pr, _STRUCT, border_value=0)
    d = np.concatenate([distance_transform_edt(~sg)[sp], distance_transform_edt(~sp)[sg]])
    return float(np.percentile(d, 95))


def test_metrics(seg_prob, clf_prob, masks, y, files, classes):
    """IoU, Dice and HD95 on lesion-positive images; Acc, macro-F1, balanced accuracy and macro one-vs-rest AUC on all."""
    t = masks.reshape(len(masks), -1) > 0
    p = seg_prob.reshape(len(seg_prob), -1) > THR
    tp, fp, fn = (t & p).sum(1), (~t & p).sum(1), (t & ~p).sum(1)
    pos = t.any(1)
    iou = np.where(pos, tp / np.maximum(tp + fp + fn, 1), np.nan)
    dice = np.where(pos, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    h = np.array([hd95(masks[i] > 0, seg_prob[i] > THR) if pos[i] else np.nan for i in range(len(y))])
    pred = clf_prob.argmax(1)
    df = pd.DataFrame({"file": files, "true": [classes[c] for c in y], "pred": [classes[c] for c in pred],
                       "correct": (pred == y).astype(int), "iou": iou, "dice": dice, "hd95": h})
    for k, c in enumerate(classes):
        df["p_" + c] = clf_prob[:, k]
    K = len(classes)
    cm = np.zeros((K, K), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    rec = np.array([cm[k, k] / max(cm[k].sum(), 1) for k in range(K)])
    prec = np.array([cm[k, k] / max(cm[:, k].sum(), 1) for k in range(K)])
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    auc = roc_auc_score(y, clf_prob, multi_class="ovr", average="macro", labels=list(range(K)))
    s = {"IoU": 100 * np.nanmean(iou), "Dice": 100 * np.nanmean(dice), "HD95": float(np.nanmean(h)),
         "Acc": 100 * float((pred == y).mean()), "Macro-F1": 100 * f1.mean(), "Bal. Acc": 100 * rec.mean(), "AUC": 100 * auc}
    return df, s, cm


def evaluate(model, X, M, Y, classes, files, groups, out):
    """Test metrics; writes the per-image predictions and the confusion matrix to out/."""
    seg, clf = predict(model, X)
    df, s, cm = test_metrics(seg, clf, M[:, 0].cpu().numpy() > 0, Y.cpu().numpy(), files, classes)
    df["group"] = groups
    df.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out, "test_confusion.csv"))
    return s


@torch.no_grad()
def complexity(model):
    """Parameters (M) and multiply-accumulate operations (G) for one 224 x 224 image."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x = channels_last(torch.rand(1, 3, 224, 224, device=DEV) * 255)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return {"params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3), "GFLOPs": round(fc.get_total_flops() / 2e9, 2)}
