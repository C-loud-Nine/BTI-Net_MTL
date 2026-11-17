"""Training protocol shared by BTI-Net and every compared method: input size, augmentation, losses, optimisation and
model selection."""
import inspect
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import load_dataset

IMG = 224
BATCH = 24
EPOCHS, WARMUP, WARMUP_BATCHES = 100, 3, 100
LR, LR_MIN, COS_ALPHA = 3e-4, 1.5e-6, 0.03
PLATEAU_PATIENCE, PLATEAU_FACTOR, PLATEAU_COOLDOWN = 7, 0.55, 2
ES_PATIENCE, ES_DELTA = 22, 0.004
SEG_W, CLF_W = 0.82, 0.18
CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"
COMPILE = CUDA and os.environ.get("BTINET_COMPILE", "1") != "0"


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16) if CUDA else torch.autocast("cpu", enabled=False)


def seed_everything(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def load(ds):
    d = dict(load_dataset(ds))
    d["X"] = np.stack([cv2.resize(x, (IMG, IMG), interpolation=cv2.INTER_AREA) for x in d["X"]])
    d["M"] = np.stack([(cv2.resize(m, (IMG, IMG), interpolation=cv2.INTER_NEAREST) > 0) for m in d["M"]]).astype(np.uint8)
    return d


def augment(images, masks):
    import albumentations as A
    t = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT_101)],
                  **({"seed": int(np.random.randint(0, 2 ** 31 - 1))} if "seed" in inspect.signature(A.Compose.__init__).parameters else {}))
    ims, mks = [images], [masks]
    for _ in range(3):
        o = [t(image=a, mask=b) for a, b in zip(images, masks)]
        ims.append(np.stack([r["image"] for r in o])); mks.append(np.stack([r["mask"] for r in o]))
    return np.concatenate(ims), np.concatenate(mks)


def training_set(d, seed):
    """The training images and three augmented copies of each, shuffled once per seed, on the device."""
    tr = np.asarray(d["IDX"]["train"])
    Xa, Ma = augment(d["X"][tr], d["M"][tr])
    Ya = np.tile(np.asarray(d["Y"])[tr], 4)
    perm = np.random.RandomState(seed).permutation(len(Xa))
    Xg = torch.from_numpy(np.ascontiguousarray(Xa[perm])).to(DEV)
    Mg = torch.from_numpy(np.ascontiguousarray((Ma[perm] > 0).astype(np.uint8))).to(DEV)
    Yg = torch.from_numpy(Ya[perm].astype(np.int64)).to(DEV)
    return Xg, Mg, Yg


def _curvature(k=5, eps=1e-8):
    r = torch.arange(k, dtype=torch.float32) - k // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    f = (xx * xx - yy * yy) * torch.exp(-(xx * xx + yy * yy) / (2.0 * ((k / 4.0) ** 2 + eps)))
    return (f / (f.abs().sum() + eps)).view(1, 1, k, k)


CURV, SOBEL = _curvature(), torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)


def seg_loss_vec(t, p):
    """Per-image segmentation loss: focal Tversky with boundary and texture terms."""
    eps, dims = 1e-7, (1, 2, 3)
    t = t.float().clamp(0, 1)
    p = torch.nan_to_num(p.float(), 0.0).clamp(eps, 1 - eps)
    tp, fp, fn_ = (t * p).sum(dims), ((1 - t) * p).sum(dims), (t * (1 - p)).sum(dims)
    base = torch.clamp(1 - (tp + eps) / (tp + 0.5 * fp + 0.5 * fn_ + eps), min=eps) ** 0.75
    with torch.no_grad():
        k = CURV.to(p.device)
        b = (F.conv2d((t > 0.5).float(), k, padding=2) - F.conv2d((p > 0.5).float(), k, padding=2)).abs().mean(dims).clamp(0, 1)
    s = SOBEL.to(p.device)
    sd = lambda g: torch.sqrt(((g - g.mean(dims, keepdim=True)) ** 2).mean(dims).clamp(min=0) + 1e-8)
    tex = (sd(F.conv2d(t, s, padding=1)) - sd(F.conv2d(p, s, padding=1))).abs()
    boost = 1 + torch.exp(-10 * (t.sum(dims) / t[0].numel()).clamp(0, 1)).clamp(0, 2)
    return (base * boost + 0.25 * b + 0.15 * tex).clamp(0, 10)


def clf_loss_vec(y, prob):
    """Per-image focal cross-entropy."""
    eps = 1e-7
    prob = torch.nan_to_num(prob.float(), eps).clamp(eps, 1 - eps)
    prob = (prob / prob.sum(1, keepdim=True)).clamp(eps, 1 - eps)
    ce = -torch.log(prob.gather(1, y.view(-1, 1)).squeeze(1))
    return (torch.clamp(1 - torch.exp(-ce), min=eps) ** 2 * ce).clamp(0, 10)


def task_loss(o, m, y, tasks=("seg", "clf")):
    """Weighted segmentation and classification losses, each with the model's own auxiliary terms."""
    ls = seg_loss_vec(m, o["seg"]).mean() if "seg" in tasks else 0.0
    lc = clf_loss_vec(y, o["clf"]).mean() if "clf" in tasks else 0.0
    for a, w in o.get("aux", ()):
        if "seg" in a and "seg" in tasks:
            ls = ls + w * seg_loss_vec(m, a["seg"]).mean()
        if "clf" in a and "clf" in tasks:
            lc = lc + w * clf_loss_vec(y, a["clf"]).mean()
    return SEG_W * ls + CLF_W * lc


class CosinePlateau:
    """Cosine decay of the learning rate, multiplied by a factor that drops on a validation plateau."""

    def __init__(self, opt):
        self.opt, self.best, self.wait, self.cool, self.factor = opt, -np.inf, 0, 0, 1.0

    def begin(self, epoch):
        cos = 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
        for g in self.opt.param_groups:
            g["lr"] = max(LR * (COS_ALPHA + (1 - COS_ALPHA) * cos) * self.factor, LR_MIN)

    def end(self, score):
        if self.cool > 0:
            self.cool -= 1
            return
        if score > self.best:
            self.best, self.wait = score, 0
        else:
            self.wait += 1
            if self.wait >= PLATEAU_PATIENCE:
                self.factor = max(self.factor * PLATEAU_FACTOR, LR_MIN / LR)
                self.cool, self.wait = PLATEAU_COOLDOWN, 0


class Compiled:
    """torch.compile of a function, with a fall-back to eager mode if compilation fails."""

    def __init__(self, fn):
        self.eager = fn
        self.fn = torch.compile(fn, dynamic=False) if COMPILE else fn
        self.on = COMPILE

    def __call__(self, *a):
        return self.fn(*a)

    def disable(self, err):
        print("torch.compile not usable here (%s): eager mode" % str(err).strip().splitlines()[-1][:200], flush=True)
        self.fn, self.on = self.eager, False


def fit(model, step, Xg, Mg, Yg, val, seed, compiled=()):
    """Trains with step(x, m, y) -> loss and restores the weights of the epoch with the best val(model)["score"]."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR, betas=(0.91, 0.999), eps=1e-7, **({"fused": True} if CUDA else {}))
    sched = CosinePlateau(opt)
    N = len(Xg)
    nb_full = max(1, N // BATCH)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    checked = [not c.on for c in compiled]

    def one(x, m, y):
        opt.zero_grad(set_to_none=True)
        if not all(checked):
            try:
                loss = step(x, m, y)
                loss.backward()
                for i in range(len(compiled)):
                    checked[i] = True
                return loss
            except Exception as ex:
                opt.zero_grad(set_to_none=True)
                for i, c in enumerate(compiled):
                    if c.on:
                        c.disable(ex)
                    checked[i] = True
        loss = step(x, m, y)
        loss.backward()
        return loss

    def run_epoch(e, max_batches=None):
        model.train(); sched.begin(e)
        order = torch.randperm(N, generator=gen, device=DEV)
        nb = min(nb_full, max_batches) if max_batches else nb_full
        acc = torch.zeros((), device=DEV)
        for b in range(nb):
            idx = order[b * BATCH:(b + 1) * BATCH]
            loss = one(Xg[idx].permute(0, 3, 1, 2).float(), Mg[idx][:, None].float(), Yg[idx])
            torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
            opt.step()
            acc += loss.detach()
        return float(acc) / nb

    t0, hist = time.time(), []
    for e in range(WARMUP):
        t = time.time()
        l = run_epoch(e, WARMUP_BATCHES)
        print("Epoch %3d/%d %5.0fs | loss %.4f | warm-up" % (e + 1, EPOCHS, time.time() - t, l), flush=True)
    best, best_es, best_w, best_epoch, wait, last = -np.inf, -np.inf, None, -1, 0, WARMUP
    for e in range(WARMUP, EPOCHS):
        t = time.time()
        l = run_epoch(e)
        vm = val(model)
        s = vm["score"] if np.isfinite(vm["score"]) else -np.inf
        hist.append({"epoch": e + 1, "train loss": l, "lr": opt.param_groups[0]["lr"], **{"val " + k: v for k, v in vm.items()}})
        if s > best:
            best, best_epoch = s, e + 1
            best_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if s > best_es + ES_DELTA:
            best_es, wait = s, 0
        else:
            wait += 1
        sched.end(s)
        last = e + 1
        print("Epoch %3d/%d %5.0fs | loss %.4f | val score %.4f IoU %.1f BAcc %.1f | best %.4f @ %d (%d/%d)"
              % (e + 1, EPOCHS, time.time() - t, l, s, vm["IoU pooled"], vm["Bal. Acc"], best, best_epoch, wait, ES_PATIENCE), flush=True)
        if wait >= ES_PATIENCE:
            break
    model.load_state_dict(best_w)
    if CUDA:
        torch.cuda.empty_cache()
    return hist, {"best_epoch": best_epoch, "best_val_score": float(best), "epochs_run": last,
                  "train_minutes": round((time.time() - t0) / 60, 1)}
