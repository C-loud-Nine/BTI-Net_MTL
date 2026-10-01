"""
Evaluation for BTI-Net: computes every metric reported in the paper.

Segmentation : foreground IoU, Dice, sensitivity, precision (threshold 0.5)
Classification: accuracy, macro-F1, weighted F1, AUC, per-class report
Failure detection: AUROC and average precision for the SRG coefficients,
                   compared against the maximum-softmax baseline

Images whose ground-truth mask is empty (BUSI 'normal', BRISC 'no_tumor') are
excluded from the segmentation metrics and counted separately, since foreground
overlap is undefined for them. They are retained for classification.

Usage
-----
    python evaluate.py --dataset busi  --weights checkpoints/busi_stage2.h5
    python evaluate.py --dataset ham   --weights checkpoints/ham_stage2.h5
    python evaluate.py --dataset brisc --weights checkpoints/brisc_stage2.h5
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, classification_report,
                             confusion_matrix)

import config
from model import enhanced_bti_model


# ---------------------------------------------------------------- segmentation
def _binarise(x, threshold=0.5):
    return (np.asarray(x).squeeze() > threshold)


def per_image_segmentation(y_true, y_pred, threshold=0.5):
    """Foreground IoU, Dice, sensitivity and precision for one image."""
    t = _binarise(y_true, 0.5)
    p = _binarise(y_pred, threshold)
    if t.sum() == 0:
        return (np.nan,) * 4
    inter = np.logical_and(t, p).sum()
    union = np.logical_or(t, p).sum()
    iou = inter / union if union > 0 else np.nan
    dice = 2.0 * inter / (t.sum() + p.sum()) if (t.sum() + p.sum()) > 0 else np.nan
    sens = inter / t.sum()
    prec = inter / p.sum() if p.sum() > 0 else np.nan
    return iou, dice, sens, prec


def evaluate_segmentation(masks_true, masks_pred, threshold=0.5):
    ious, dices, senss, precs, per_image_iou = [], [], [], [], []
    n_empty = 0
    for t, p in zip(masks_true, masks_pred):
        if _binarise(t, 0.5).sum() == 0:
            n_empty += 1
            per_image_iou.append(np.nan)
            continue
        i, d, s, pr = per_image_segmentation(t, p, threshold)
        ious.append(i); dices.append(d); senss.append(s); precs.append(pr)
        per_image_iou.append(i)
    return {
        'iou': float(np.nanmean(ious) * 100),
        'dice': float(np.nanmean(dices) * 100),
        'median_dice': float(np.nanmedian(dices) * 100),
        'sensitivity': float(np.nanmean(senss) * 100),
        'precision': float(np.nanmean(precs) * 100),
        'n_evaluated': len(ious),
        'n_empty_masks_excluded': n_empty,
        'threshold': threshold,
    }, np.array(per_image_iou)


# -------------------------------------------------------------- classification
def evaluate_classification(y_true, y_prob, class_names):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    n_classes = y_prob.shape[1]
    try:
        auc = (roc_auc_score(y_true, y_prob[:, 1]) if n_classes == 2
               else roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except ValueError:
        auc = float('nan')
    return {
        'accuracy': float(accuracy_score(y_true, y_pred) * 100),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro') * 100),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted') * 100),
        'auc': float(auc * 100),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'per_class': classification_report(y_true, y_pred, target_names=class_names,
                                           output_dict=True, zero_division=0),
    }


# ----------------------------------------------------------- failure detection
def evaluate_failure_detection(model, images, per_image_iou, y_true, y_prob,
                               iou_threshold=0.5, batch_size=16):
    """Score the SRG coefficients as single-pass failure signals.

    A segmentation failure is an image whose foreground IoU falls below
    `iou_threshold`; a classification failure is a wrong label. These labels
    score coefficients that have already been computed; they never tune them.

    The maximum-softmax baseline (1 - max probability) is reported alongside,
    because it is the standard single-pass comparator.
    """
    srg_layers = [model.get_layer(n) for n in config.SRG_LAYER_NAMES]
    _ = model.predict(images, batch_size=batch_size, verbose=0)

    results = {'segmentation': {}, 'classification': {}}

    seg_fail = (per_image_iou < iou_threshold).astype(int)
    valid = ~np.isnan(per_image_iou)
    for name, layer in zip(config.SRG_LAYER_NAMES, srg_layers):
        w = getattr(layer, '_last_w_seg', None)
        if w is None:
            continue
        w = np.asarray(w).reshape(-1)
        if len(w) != len(seg_fail):
            continue
        try:
            results['segmentation'][name] = {
                'auroc': float(roc_auc_score(seg_fail[valid], w[valid])),
                'ap': float(average_precision_score(seg_fail[valid], w[valid])),
            }
        except ValueError:
            pass

    y_true = np.asarray(y_true).reshape(-1)
    clf_fail = (np.asarray(y_prob).argmax(axis=1) != y_true).astype(int)
    for name, layer in zip(config.SRG_LAYER_NAMES, srg_layers):
        w = getattr(layer, '_last_w_clf', None)
        if w is None:
            continue
        w = np.asarray(w).reshape(-1)
        if len(w) != len(clf_fail):
            continue
        try:
            results['classification'][name] = {
                'auroc': float(roc_auc_score(clf_fail, w)),
                'ap': float(average_precision_score(clf_fail, w)),
            }
        except ValueError:
            pass

    one_minus_conf = 1.0 - np.asarray(y_prob).max(axis=1)
    try:
        results['classification']['baseline_1_minus_conf'] = {
            'auroc': float(roc_auc_score(clf_fail, one_minus_conf)),
            'ap': float(average_precision_score(clf_fail, one_minus_conf)),
        }
    except ValueError:
        pass

    results['n_seg_failures'] = int(seg_fail[valid].sum())
    results['n_clf_failures'] = int(clf_fail.sum())
    return results


# ----------------------------------------------------------------------- main
def _load_dataset(name):
    """Return (X, masks, labels, class_names, n_classes) for the test split.

    The three loaders differ: BUSI returns numpy arrays, while HAM10000 and
    BRISC return tf.data pipelines, so the latter are materialised here.
    """
    if name == 'busi':
        import busi_dataloader as dl
        d = dl.prepare_datasets()
        return (d['X_test'], d['mask_test'], d['y_test'],
                config.BUSI_CATEGORIES, config.BUSI_NUM_CLF_CLASSES)

    if name == 'ham':
        import ham_dataloader as dl
        d = dl.prepare_ham_datasets()
        cats, n = config.HAM_CATEGORIES, config.HAM_NUM_CLF_CLASSES
    elif name == 'brisc':
        import brisc_dataloader as dl
        d = dl.prepare_brics_datasets()
        cats, n = config.BRISC_CATEGORIES, config.BRISC_NUM_CLF_CLASSES
    else:
        raise ValueError(f"unknown dataset '{name}'")

    xs, ms, ys = [], [], []
    for batch in d['test_dataset']:
        inputs, targets = batch[0], batch[1]
        xs.append(inputs.numpy())
        seg = targets['segmentation_output'] if isinstance(targets, dict) else targets[0]
        clf = targets['classification_output'] if isinstance(targets, dict) else targets[1]
        ms.append(np.asarray(seg))
        ys.append(np.asarray(clf))
    X = np.concatenate(xs); M = np.concatenate(ms); Y = np.concatenate(ys)
    if Y.ndim > 1 and Y.shape[-1] > 1:
        Y = Y.argmax(axis=-1)
    return X, M, Y.reshape(-1), cats, n


def _load_model(path, n_classes):
    """Load either a full saved model (.keras) or a weights file (.h5).

    train.py saves a complete model with `model.save(... .keras)`, so that
    path is loaded directly. A weights-only file is applied to a freshly built
    architecture instead.
    """
    if path.endswith('.keras') or os.path.isdir(path):
        return tf.keras.models.load_model(path, compile=False)
    model = enhanced_bti_model(input_size=config.INPUT_SIZE,
                               num_seg_classes=config.NUM_SEG_CLASSES,
                               num_clf_classes=n_classes,
                               dropout_rate=config.DROPOUT_RATE)
    model.load_weights(path)
    return model


def main():
    ap = argparse.ArgumentParser(description="BTI-Net evaluation")
    ap.add_argument('--dataset', required=True, choices=['busi', 'ham', 'brisc'])
    ap.add_argument('--weights', required=True,
                    help='checkpoints/final_model_ft_<dataset>.keras from train.py, '
                         'or a weights-only .h5 file')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--failure-detection', action='store_true',
                    help='also score the SRG coefficients as failure signals')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    X_test, mask_test, y_test, categories, n_classes = _load_dataset(args.dataset)

    model = _load_model(args.weights, n_classes)

    seg_pred, clf_pred = model.predict(X_test,
                                       batch_size=config.VAL_BATCH_SIZE, verbose=0)

    seg_metrics, per_image_iou = evaluate_segmentation(
        mask_test, seg_pred, args.threshold)
    clf_metrics = evaluate_classification(y_test, clf_pred, categories)

    results = {
        'dataset': args.dataset,
        'n_test': int(len(X_test)),
        'segmentation': seg_metrics,
        'classification': clf_metrics,
    }

    if args.failure_detection:
        results['failure_detection'] = evaluate_failure_detection(
            model, X_test, per_image_iou, y_test, clf_pred)

    print(f"\n{args.dataset.upper()}  (n = {results['n_test']})")
    print("-" * 44)
    print(f"  IoU            {seg_metrics['iou']:.2f}")
    print(f"  Dice           {seg_metrics['dice']:.2f}   (median {seg_metrics['median_dice']:.2f})")
    print(f"  Sensitivity    {seg_metrics['sensitivity']:.2f}")
    print(f"  Precision      {seg_metrics['precision']:.2f}")
    print(f"  Accuracy       {clf_metrics['accuracy']:.2f}")
    print(f"  Macro F1       {clf_metrics['macro_f1']:.2f}")
    print(f"  Weighted F1    {clf_metrics['weighted_f1']:.2f}")
    print(f"  AUC            {clf_metrics['auc']:.2f}")
    print(f"  (excluded {seg_metrics['n_empty_masks_excluded']} empty-mask cases from segmentation)")

    out = args.out or os.path.join(config.RESULTS_DIR, f'eval_{args.dataset}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {out}")


if __name__ == '__main__':
    main()
