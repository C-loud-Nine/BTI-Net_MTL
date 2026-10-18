"""
evaluate.py - per-image evaluation and gate export for BTI-Net.

ADDITIVE. Trains nothing and changes nothing in the model or the data
pipeline. Works on all three modalities (BUSI, HAM10000, BRISC).

This replaces the pattern of reading gate coefficients back from the layer
attributes `_last_w_seg` / `_last_w_clf` after model.predict(). Those are not
dataset-length accumulators: they hold the MOST RECENT CALL only, so any table
or figure built from them covers one batch, not the test set. Here the eight
coefficients are recomputed as real model outputs through
SupervisedReliabilityGate.compute_gates and collected per batch, with an
assertion that every length equals N.

What it writes to config.RESULTS_DIR:
  predictions_<tag>.csv   one row per test image, with all 8 gate values
  confusion_<tag>.csv     confusion matrix, entries sum to N
  summary_<tag>.json      aggregates, each stated with its denominator

Why each column exists:
  * Accuracy must equal 100 * (integer correct) / N, so every reported
    accuracy has to be a multiple of 100/N. Both numbers are printed.
  * IoU and Dice come from the SAME per-image TP/FP/FN at the SAME threshold,
    so the per-image identity Dice = 2*IoU/(1+IoU) holds and the aggregate
    obeys mean Dice <= 2*mIoU/(1+mIoU). The summary prints that ceiling.
  * Macro-F1 and balanced accuracy are reported because weighted recall is
    numerically identical to accuracy and therefore answers nothing about
    minority classes.
  * Images with an all-zero mask are counted separately. On BRISC this matters:
    tumour-labelled images with no mask file are substituted with a zero mask
    upstream (see brisc_dataloader.py), which depresses the overlap metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/busi_stage2.keras --dataset busi
    python evaluate.py --checkpoint checkpoints/ham_stage2.keras  --dataset ham
    python evaluate.py --checkpoint checkpoints/brisc_stage2.keras --dataset brisc
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

import config
import modules

THRESHOLD = 0.5
EPS = 1e-7

CUSTOM_OBJECTS = {
    "MultiScaleContextFusion": modules.MultiScaleContextFusion,
    "DualPathAttention": modules.DualPathAttention,
    "ResidualBlock": modules.ResidualBlock,
    "AttentionGate": modules.AttentionGate,
    "TaskInteractionModule": modules.TaskInteractionModule,
    "SupervisedReliabilityGate": modules.SupervisedReliabilityGate,
}

DATASET_INFO = {
    "busi":  (config.BUSI_CATEGORIES,  config.BUSI_NUM_CLF_CLASSES),
    "ham":   (config.HAM_CATEGORIES,   config.HAM_NUM_CLF_CLASSES),
    "brisc": (config.BRISC_CATEGORIES, config.BRISC_NUM_CLF_CLASSES),
}


def load_test_batches(dataset, batch_size):
    """Yield (images, masks, labels) batches for the test split.

    BUSI returns numpy arrays from its loader; HAM10000 and BRISC return
    tf.data pipelines. Both shapes are normalised to numpy batches here so the
    rest of the script is identical across modalities.
    """
    if dataset == "busi":
        import busi_dataloader as dl
        d = dl.prepare_datasets()
        X, m, y = d["X_test"], d["mask_test"], np.asarray(d["y_test"]).astype(int)
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size], m[i:i + batch_size], y[i:i + batch_size]
        return

    if dataset == "ham":
        import ham_dataloader as dl
        ds = dl.prepare_ham_datasets()["test_dataset"]
    elif dataset == "brisc":
        import brisc_dataloader as dl
        ds = dl.prepare_brics_datasets()["test_dataset"]
    else:
        raise ValueError(dataset)

    print("[note] this modality uses a tf.data pipeline, so the evaluation "
          f"batch size is the one it was built with (config.BATCH_SIZE = "
          f"{config.BATCH_SIZE}), not --batch-size.")
    for images, targets in ds:
        yield (images.numpy(),
               targets["segmentation_output"].numpy(),
               targets["classification_output"].numpy().astype(int).reshape(-1))


def build_gate_export_model(model):
    """Expose the eight SRG coefficients as real model outputs.

    Uses SupervisedReliabilityGate.compute_gates on the layer's own input
    tensors, so the exported values are exactly the ones the blend used.

    Returns (export_model, level_names); level_names is empty when the gate
    layers cannot be located, in which case the gate columns are omitted
    rather than invented.
    """
    outs, names = [], []
    for nm in config.SRG_LAYER_NAMES:
        try:
            layer = model.get_layer(nm)
        except ValueError:
            print(f"[warn] gate layer '{nm}' not found; gates not exported")
            return model, []
        ins = layer.input
        if not isinstance(ins, (list, tuple)) or len(ins) != 4:
            print(f"[warn] unexpected inputs on '{nm}'; gates not exported")
            return model, []
        outs.append(layer.compute_gates(*ins))
        names.append(nm)
    return tf.keras.Model(model.inputs, list(model.outputs) + outs), names


def per_image_overlap(y_true, y_pred_prob, threshold=THRESHOLD):
    """Per-image TP/FP/FN/IoU/Dice over foreground, from ONE set of counts.

    An all-zero ground-truth mask yields NaN rather than 0.0 or 1.0: it has no
    foreground to overlap, and forcing a value is what makes aggregate IoU and
    Dice drift apart.
    """
    t = (y_true > 0.5).astype(np.float64).reshape(len(y_true), -1)
    p = (y_pred_prob > threshold).astype(np.float64).reshape(len(y_pred_prob), -1)
    tp = (t * p).sum(axis=1)
    fp = ((1.0 - t) * p).sum(axis=1)
    fn = (t * (1.0 - p)).sum(axis=1)
    has_fg = t.sum(axis=1) > 0
    iou = np.where(has_fg, tp / np.maximum(tp + fp + fn, EPS), np.nan)
    dice = np.where(has_fg, 2.0 * tp / np.maximum(2.0 * tp + fp + fn, EPS), np.nan)
    return tp, fp, fn, iou, dice, has_fg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, choices=["busi", "ham", "brisc"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch-size", type=int, default=config.VAL_BATCH_SIZE,
                    help="BUSI only; HAM and BRISC use the tf.data batch size.")
    args = ap.parse_args()
    tag = args.tag or f"{args.dataset}_bs{args.batch_size}"
    cls, n_cls = DATASET_INFO[args.dataset]

    print(f"[load] {args.checkpoint}")
    model = tf.keras.models.load_model(
        args.checkpoint, compile=False, custom_objects=CUSTOM_OBJECTS)
    export, gate_names = build_gate_export_model(model)

    rows_y, rows_pred, rows_prob = [], [], []
    acc_tp, acc_fp, acc_fn, acc_iou, acc_dice, acc_fg = [], [], [], [], [], []
    acc_gates = [[] for _ in gate_names]

    for images, masks, labels in load_test_batches(args.dataset, args.batch_size):
        out = export.predict_on_batch(images)
        seg = np.asarray(out[0])
        if seg.ndim == 4:
            seg = seg[..., 0]
        clf = np.asarray(out[1])
        m = np.asarray(masks)
        if m.ndim == 4:
            m = m[..., 0]

        tp, fp, fn, iou, dice, fg = per_image_overlap(m, seg)
        acc_tp.append(tp); acc_fp.append(fp); acc_fn.append(fn)
        acc_iou.append(iou); acc_dice.append(dice); acc_fg.append(fg)
        rows_y.append(labels); rows_pred.append(clf.argmax(axis=1)); rows_prob.append(clf)
        for j in range(len(gate_names)):
            acc_gates[j].append(np.asarray(out[2 + j]).reshape(len(images), -1))

    y_true = np.concatenate(rows_y)
    clf_pred = np.concatenate(rows_pred)
    clf_prob = np.concatenate(rows_prob)
    tp = np.concatenate(acc_tp); fp = np.concatenate(acc_fp); fn = np.concatenate(acc_fn)
    iou = np.concatenate(acc_iou); dice = np.concatenate(acc_dice)
    has_fg = np.concatenate(acc_fg)
    gates = [np.concatenate(g) for g in acc_gates]

    n = len(y_true)
    # Hard guarantee against the last-batch truncation this script exists to fix.
    assert len(clf_pred) == n and len(iou) == n, "metric length != N"
    for nm, g in zip(gate_names, gates):
        assert len(g) == n, f"gate '{nm}' has {len(g)} rows, expected {n}"
    print(f"[data] {args.dataset}: test images N = {n}  "
          f"(grid step = {100.0 / n:.4f} accuracy points)")

    correct = (clf_pred == y_true)
    n_correct = int(correct.sum())
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ---- per-image CSV ---------------------------------------------------
    path = os.path.join(config.RESULTS_DIR, f"predictions_{tag}.csv")
    gcols = [f"{nm}_w_{k}" for nm in gate_names for k in ("seg", "clf")]
    with open(path, "w") as f:
        f.write("index,true_class,pred_class,correct,max_prob," +
                ",".join(f"p_{c}" for c in cls) +
                ",tp,fp,fn,iou,dice,has_foreground" +
                ("," + ",".join(gcols) if gcols else "") + "\n")
        for i in range(n):
            row = [str(i), cls[y_true[i]], cls[clf_pred[i]], str(int(correct[i])),
                   f"{clf_prob[i].max():.6f}"]
            row += [f"{v:.6f}" for v in clf_prob[i]]
            row += [f"{tp[i]:.0f}", f"{fp[i]:.0f}", f"{fn[i]:.0f}",
                    f"{iou[i]:.6f}", f"{dice[i]:.6f}", str(int(has_fg[i]))]
            for g in gates:
                row += [f"{v:.6f}" for v in g[i]]
            f.write(",".join(row) + "\n")
    print(f"[write] {path}")

    # ---- confusion matrix ------------------------------------------------
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t_, p_ in zip(y_true, clf_pred):
        cm[t_, p_] += 1
    assert cm.sum() == n, "confusion matrix does not sum to N"
    cpath = os.path.join(config.RESULTS_DIR, f"confusion_{tag}.csv")
    with open(cpath, "w") as f:
        f.write("true\\pred," + ",".join(cls) + "\n")
        for i, c in enumerate(cls):
            f.write(c + "," + ",".join(str(v) for v in cm[i]) + "\n")
    print(f"[write] {cpath}")

    # ---- aggregates, each with its denominator ---------------------------
    per_class, f1s, recs = {}, [], []
    for i, c in enumerate(cls):
        tp_c = int(cm[i, i])
        fn_c = int(cm[i].sum()) - tp_c
        fp_c = int(cm[:, i].sum()) - tp_c
        prec = tp_c / max(tp_c + fp_c, 1)
        rec = tp_c / max(tp_c + fn_c, 1)
        f1 = 2 * prec * rec / max(prec + rec, EPS)
        per_class[c] = {"support": int(cm[i].sum()), "precision": prec,
                        "recall": rec, "f1": f1}
        f1s.append(f1)
        recs.append(rec)

    n_fg = int(has_fg.sum())
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "eval_batch_size": args.batch_size if args.dataset == "busi" else config.BATCH_SIZE,
        "threshold": THRESHOLD,
        "n_test_images": n,
        "n_correct": n_correct,
        "accuracy_percent": 100.0 * n_correct / n,
        "accuracy_grid_step_percent": 100.0 / n,
        "macro_f1_percent": 100.0 * float(np.mean(f1s)),
        "balanced_accuracy_percent": 100.0 * float(np.mean(recs)),
        "n_positive_mask_images": n_fg,
        "n_zero_mask_images": int(n - n_fg),
        "mean_iou_percent": float(np.nanmean(iou) * 100.0),
        "mean_dice_percent": float(np.nanmean(dice) * 100.0),
        "median_dice_percent": float(np.nanmedian(dice) * 100.0),
        "per_class": per_class,
        "gate_layers_exported": gate_names,
    }
    m_ = summary["mean_iou_percent"] / 100.0
    summary["max_mean_dice_implied_by_mean_iou_percent"] = 100.0 * 2 * m_ / (1 + m_)
    summary["dice_within_ceiling"] = bool(
        summary["mean_dice_percent"]
        <= summary["max_mean_dice_implied_by_mean_iou_percent"] + 1e-9)
    if gate_names:
        summary["mean_gate_coefficients"] = {
            nm: {"w_seg": float(g[:, 0].mean()), "w_clf": float(g[:, 1].mean())}
            for nm, g in zip(gate_names, gates)}

    spath = os.path.join(config.RESULTS_DIR, f"summary_{tag}.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ RECONSTRUCTION CHECK ================")
    print(f"  accuracy    = {n_correct}/{n} = {summary['accuracy_percent']:.4f}%")
    print(f"  grid step   = {summary['accuracy_grid_step_percent']:.4f} points")
    print(f"                any reported accuracy must be a multiple of this")
    print(f"  macro-F1    = {summary['macro_f1_percent']:.4f}%")
    print(f"  balanced acc= {summary['balanced_accuracy_percent']:.4f}%")
    print(f"  mean IoU    = {summary['mean_iou_percent']:.4f}% over {n_fg} positive-mask images")
    print(f"  zero-mask images excluded from overlap: {n - n_fg}")
    print(f"  mean Dice   = {summary['mean_dice_percent']:.4f}%  "
          f"(ceiling {summary['max_mean_dice_implied_by_mean_iou_percent']:.4f}%, "
          f"{'OK' if summary['dice_within_ceiling'] else 'VIOLATED'})")
    print(f"  gate layers exported: {gate_names or 'none'}")
    print("======================================================")
    print(f"[write] {spath}")


if __name__ == "__main__":
    main()
