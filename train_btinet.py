"""BTI-Net: training and testing of one run.

    python train_btinet.py <BUSI|BRISC|HAM10000> <seed> <output folder> [ablation]

ablation (Table V): no_tim (b), no_c2s (c: seg->cls only), no_s2c (d: cls->seg only), gate_one (e), gate_const (f),
no_gsup (g), gate_aiw (h), no_gctx_pres (i).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from btinet import evaluation as ev
from btinet import protocol as pr
from btinet.common import DATASETS, save_json
from btinet.model import BTINet, calibrate_aiw

LAMBDA = 1.0


class counterfactual_mode:
    """Rollouts without gradient, in single precision, with dropout off and BN in inference mode."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.mods = [(m, m.training) for m in self.model.modules()
                     if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd))]
        for m, _ in self.mods:
            m.eval()
        self.model._cf = True
        self.ng = torch.no_grad(); self.ng.__enter__()
        self.ac = torch.autocast("cuda" if pr.CUDA else "cpu", enabled=False); self.ac.__enter__()
        return self

    def __exit__(self, *a):
        self.ac.__exit__(*a); self.ng.__exit__(*a)
        for m, tr in self.mods:
            m.train(tr)
        self.model._cf = False
        return False


def benefit(model, o, m, y, l, k):
    """Per-image benefit of the exchange at level l, stream k: multi-task loss with w <- 0 minus with w <- 1."""
    e = {kk: vv.detach().float() for kk, vv in o["e"].items()}
    oo = {"cache": [{"d_in": c["d_in"].detach().float(), "pm_in": None if c["pm_in"] is None else c["pm_in"].detach().float()}
                    for c in o["cache"]], "douts": [v.detach().float() for v in o["douts"]],
          "f_fin": [v.detach().float() for v in o["f_fin"]], "w": o["w"]}
    B = y.shape[0]
    res = []
    for v in (torch.zeros(B, device=m.device), torch.ones(B, device=m.device)):
        seg, clf = model.rollout(e, oo, l, k, v, m.shape[-2:])
        res.append(pr.SEG_W * pr.seg_loss_vec(m, seg.float()) + pr.CLF_W * pr.clf_loss_vec(y, clf.float()))
    return res[0] - res[1]


@torch.no_grad()
def gate_statistics(model, X, M, Y, bs=16):
    """On validation images, in inference mode: share of image-level pairs with positive benefit, AUC of the
    coefficients against the sign of the benefit, mean coefficients overall and per level."""
    model.eval()
    rec = {0: ([], []), 1: ([], [])}
    per_level = {}
    for i in range(0, len(X), bs):
        x, m, y = ev.channels_last(X[i:i + bs].float()), M[i:i + bs].float(), Y[i:i + bs]
        with pr.amp():
            o = model(x, keep=True)
        for l in range(4):
            for k in (0, 1):
                with torch.autocast("cuda" if pr.CUDA else "cpu", enabled=False):
                    dl = benefit(model, o, m, y, l, k)
                w = o["w"][l][:, k].float()
                rec[k][0].append(w.cpu()); rec[k][1].append(dl.cpu())
                per_level.setdefault("mean w %s L%d" % ("seg" if k == 0 else "cls", l + 1), []).append(w.mean().item())
    out = {}
    for k, name in ((0, "seg"), (1, "cls")):
        w, dl = torch.cat(rec[k][0]).numpy(), torch.cat(rec[k][1]).numpy()
        t = (dl > 0).astype(float)
        out["share benefit>0 " + name] = float(t.mean())
        out["gate AUC " + name] = float(roc_auc_score(t, w)) if 0 < t.mean() < 1 else float("nan")
        out["mean w " + name] = float(w.mean())
    out.update({k: float(np.mean(v)) for k, v in per_level.items()})
    return out


def main(ds, seed, out, ablation=None):
    seed = int(seed)
    os.makedirs(out, exist_ok=True)
    pr.seed_everything(seed)
    classes = DATASETS[ds]["classes"]
    d = pr.load(ds)
    tr, va, te = (np.asarray(d["IDX"][s]) for s in ("train", "val", "test"))
    Xg, Mg, Yg = pr.training_set(d, seed)
    Xv, Mv, Yv = ev.to_device(d, va)
    print("%s BTI-Net%s seed %d | train %d (x4 = %d), val %d, test %d"
          % (ds, " [%s]" % ablation if ablation else "", seed, len(tr), len(Xg), len(va), len(te)), flush=True)
    model = BTINet(len(classes), ablation).to(pr.DEV)
    empty = DATASETS[ds]["empty"]
    model.set_empty(classes.index(empty) if empty else -1)
    if pr.CUDA:
        model = model.to(memory_format=torch.channels_last)
    levels = list(range(4)) if model.supervised else []

    def fwd(x, m, y):
        with pr.amp():
            o = model(x, keep=bool(levels))
        return pr.task_loss(o, m, y), o

    fwd_c = pr.Compiled(fwd)
    pairs = [(l, k) for l in levels for k in (0, 1)]
    warm = pr.WARMUP * min(max(1, len(Xg) // pr.BATCH), pr.WARMUP_BATCHES)
    count = {"step": 0}

    def step(x, m, y):
        task, o = fwd_c(x, m, y)
        if not levels:
            return task
        i = count["step"]
        count["step"] = i + 1
        if i < warm:
            return task
        l, k = pairs[(i - warm) % len(pairs)]
        with counterfactual_mode(model):
            dl = benefit(model, o, m, y, l, k)
        lg = model.gate_logits(o, l)[:, k]
        wt = dl.abs() / (dl.abs().mean() + 1e-6)
        gl = (wt * F.binary_cross_entropy_with_logits(lg, (dl > 0).float(), reduction="none")).mean()
        return task + LAMBDA * gl

    t0 = time.time()
    hist, info = pr.fit(model, step, Xg, Mg, Yg, lambda mdl: ev.val_metrics(mdl, Xv, Mv, Yv), seed, compiled=(fwd_c,))
    del Xg, Mg, Yg
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    calib = None
    if model.gate == "aiw":
        to_dev = lambda a: ev.channels_last(torch.from_numpy(np.ascontiguousarray(a)).to(pr.DEV).permute(0, 3, 1, 2).float())
        calib = calibrate_aiw(model, d["X"][tr], to_dev)
    gates = gate_statistics(model, Xv, Mv.float(), Yv) if model.tim and model.gate == "upa" else {}
    Xt, Mt, Yt = ev.to_device(d, te)
    test = ev.evaluate(model, Xt, Mt, Yt, classes, list(d["FILES"][te]), d["GROUP"][te], out)
    res = {"dataset": ds, "model": "BTI-Net", "ablation": ablation, "seed": seed, **info,
           "n": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
           "validation": ev.val_metrics(model, Xv, Mv, Yv), "gates": gates, "aiw_calibration": calib,
           "test": test, "complexity": ev.complexity(model), "minutes": round((time.time() - t0) / 60, 1)}
    if model.pres and model._empty >= 0:
        res["presence"] = {"alpha": float(model.pres_alpha), "beta": float(model.pres_beta),
                           "tau": float(F.softplus(model.pres_tau) + 0.05)}
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out, "weights.pt"))
    save_json(res, os.path.join(out, "results.json"))
    print("Test: IoU %.2f | Dice %.2f | HD95 %.1f | Acc %.2f | BAcc %.2f | AUC %.2f"
          % tuple(test[k] for k in ("IoU", "Dice", "HD95", "Acc", "Bal. Acc", "AUC")), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3] if len(a) > 3 else None)
