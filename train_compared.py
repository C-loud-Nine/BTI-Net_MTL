"""Compared methods: training and testing of one run, under the protocol of BTI-Net.

    python train_compared.py <BUSI|BRISC|HAM10000> <method> <seed> <output folder>

method: mtan, mti_net, densemtl, mtanet, mtl_oca, aiw (multi-task methods);
        unet, att_unet, unet_pp, transunet, swin_unet, missformer, stl_seg, stl_clf (references).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from btinet import evaluation as ev
from btinet import protocol as pr
from btinet.common import DATASETS, save_json
from methods import multitask, references


def main(ds, method, seed, out):
    seed = int(seed)
    os.makedirs(out, exist_ok=True)
    pr.seed_everything(seed)
    if method in multitask.LABEL:
        build, label, tasks = multitask.build, multitask.LABEL[method], ("seg", "clf")
    else:
        build, label, tasks = references.build, references.LABEL[method], references.TASKS[method]
    classes = DATASETS[ds]["classes"]
    d = pr.load(ds)
    tr, va, te = (np.asarray(d["IDX"][s]) for s in ("train", "val", "test"))
    Xg, Mg, Yg = pr.training_set(d, seed)
    Xv, Mv, Yv = ev.to_device(d, va)
    print("%s %s seed %d | train %d (x4 = %d), val %d, test %d" % (ds, label, seed, len(tr), len(Xg), len(va), len(te)), flush=True)
    model = build(method, len(classes)).to(pr.DEV)
    if pr.CUDA:
        model = model.to(memory_format=torch.channels_last)

    def step(x, m, y):
        with pr.amp():
            o = model(x)
        return pr.task_loss(o, m, y, tasks)

    step_c = pr.Compiled(step)
    t0 = time.time()
    hist, info = pr.fit(model, step_c, Xg, Mg, Yg, lambda mdl: ev.val_metrics(mdl, Xv, Mv, Yv, tasks), seed, compiled=(step_c,))
    del Xg, Mg, Yg
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    calib = None
    if method == "aiw":
        calib = {k: [float(v) for v in vals] for k, vals in model.calibrate(d["X"][tr], 24).items()}
    Xt, Mt, Yt = ev.to_device(d, te)
    test = ev.evaluate(model, Xt, Mt, Yt, classes, list(d["FILES"][te]), d["GROUP"][te], out)
    res = {"dataset": ds, "method": method, "label": label, "tasks": list(tasks), "seed": seed, **info,
           "n": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
           "validation": ev.val_metrics(model, Xv, Mv, Yv, tasks), "aiw_calibration": calib, "test": test,
           "complexity": ev.complexity(model), "minutes": round((time.time() - t0) / 60, 1)}
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out, "weights.pt"))
    save_json(res, os.path.join(out, "results.json"))
    print("Test: IoU %.2f | Dice %.2f | HD95 %.1f | Acc %.2f | BAcc %.2f | AUC %.2f"
          % tuple(test[k] for k in ("IoU", "Dice", "HD95", "Acc", "Bal. Acc", "AUC")), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
