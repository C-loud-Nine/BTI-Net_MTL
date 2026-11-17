"""Dataset readers and partitions.

Images are read once at 320 x 320 and cached in data/processed/; protocol.load resizes them to the input size.
BUSI and HAM10000 follow splits/*_partitions.csv. BRISC keeps its official test set; 15% of its training images
(class-stratified, random_state=42) form the validation set, written to splits/brisc_partitions.csv on first use.
"""
import csv
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .common import DATASETS, PROCESSED, RAW, SPLITS

READ_SIZE = 320
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def read_partition(name):
    with open(os.path.join(SPLITS, name.lower() + "_partitions.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _walk(root):
    out = []
    for d, sub, fs in os.walk(root, followlinks=True):
        sub[:] = [x for x in sub if x != "__MACOSX"]
        out += [os.path.join(d, f) for f in fs if f.lower().endswith(IMG_EXT) and not f.startswith("._")]
    return sorted(out)


def _subdirs(p):
    try:
        return {d.lower(): os.path.join(p, d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))}
    except OSError:
        return {}


def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError("cannot read image " + path)
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (READ_SIZE, READ_SIZE))


def read_mask(paths):
    m = None
    for p in paths:
        k = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if k is None:
            continue
        k = cv2.resize(k, (READ_SIZE, READ_SIZE), interpolation=cv2.INTER_NEAREST)
        m = k if m is None else np.maximum(m, k)
    if m is None:
        raise IOError("cannot read mask " + str(paths))
    return (m > 127).astype(np.uint8)


def _pload(fn, items):
    with ThreadPoolExecutor(max_workers=32) as ex:
        return list(ex.map(fn, items))


def _stack(out):
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def _build_busi(rows):
    root = None
    for d, _, _ in os.walk(os.path.join(RAW, "BUSI"), followlinks=True):
        if {"normal", "benign", "malignant"} <= set(_subdirs(d)):
            root = _subdirs(d)
            break
    if root is None:
        raise FileNotFoundError("BUSI not found under " + os.path.join(RAW, "BUSI"))

    def one(r):
        cls, base = r["file"].split("/", 1)
        p = os.path.join(root[cls], base)
        masks = sorted(glob.glob(glob.escape(p[:-4]) + "_mask*.png"))
        if not masks:
            raise FileNotFoundError("no mask for " + p)
        return read_rgb(p), read_mask(masks)
    return _stack(_pload(one, rows))


def _build_ham(rows):
    pat = re.compile(r"(ISIC_\d{7})(_segmentation)?\.(jpg|jpeg|png)$", re.I)
    imgs, masks = {}, {}
    root = os.path.join(RAW, "HAM10000")
    for f in _walk(root):
        m = pat.search(os.path.basename(f))
        if m:
            in_mask_dir = any(("mask" in p.lower() or "segmentation" in p.lower()) for p in os.path.relpath(f, root).split(os.sep)[:-1])
            (masks if (m.group(2) or in_mask_dir) else imgs).setdefault(m.group(1).upper(), f)
    ids = [r["file"][:-4].upper() for r in rows]
    miss = [i for i in ids if i not in imgs or i not in masks]
    if miss:
        raise FileNotFoundError(str(len(miss)) + " HAM10000 images or masks missing, e.g. " + str(miss[:3]))
    return _stack(_pload(lambda i: (read_rgb(imgs[i]), read_mask([masks[i]])), ids))


def _brisc_files():
    classes = DATASETS["BRISC"]["classes"]
    files = _walk(os.path.join(RAW, "BRISC"))
    rows, seen = [], set()
    for p in files:
        parts = [s.lower() for s in p.split(os.sep)]
        if "classification_task" in parts and parts[-2] in classes and parts[-3] in ("train", "test"):
            k = (parts[-3], parts[-2], os.path.basename(p))
            if k not in seen:
                seen.add(k)
                rows.append(k + (p,))
    masks = {}
    for p in files:
        parts = [s.lower() for s in p.split(os.sep)]
        if "segmentation_task" in parts and "masks" in parts:
            masks.setdefault(os.path.splitext(os.path.basename(p))[0].lower(), p)
    order = {c: i for i, c in enumerate(classes)}
    rows.sort(key=lambda r: (r[0], order[r[1]], r[2]))
    if (sum(r[0] == "train" for r in rows), sum(r[0] == "test" for r in rows)) != (5000, 1000):
        raise RuntimeError("BRISC should hold 5000 training and 1000 test images")
    return rows, masks, order


def _brisc_partition(rows, order):
    path = os.path.join(SPLITS, "brisc_partitions.csv")
    names = [r[0] + "/" + r[1] + "/" + r[2] for r in rows]
    if not os.path.isfile(path):
        from sklearn.model_selection import train_test_split
        y = np.array([order[r[1]] for r in rows])
        pool = np.where([r[0] == "train" for r in rows])[0]
        _, va = train_test_split(pool, test_size=0.15, stratify=y[pool], random_state=42)
        split = ["test" if r[0] == "test" else "train" for r in rows]
        for i in va:
            split[i] = "val"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "class", "split", "group"])
            for n, r, s in zip(names, rows, split):
                w.writerow([n, r[1], s, n])
    part = {r["file"]: r for r in read_partition("BRISC")}
    if set(part) != set(names):
        raise RuntimeError(path + " does not match the BRISC images")
    return [part[n] for n in names]


def _build_brisc():
    rows, masks, order = _brisc_files()
    part = _brisc_partition(rows, order)
    stems = [os.path.splitext(r[2])[0].lower() for r in rows]
    tumour = [r[1] != "no_tumor" for r in rows]
    miss = [s for s, t in zip(stems, tumour) if t and s not in masks]
    if miss:
        raise FileNotFoundError(str(len(miss)) + " BRISC tumour images without mask, e.g. " + str(miss[:3]))
    blank = np.zeros((READ_SIZE, READ_SIZE), np.uint8)
    X, M = _stack(_pload(lambda i: (read_rgb(rows[i][3]), read_mask([masks[stems[i]]]) if tumour[i] else blank), range(len(rows))))
    return X, M, part


def _build(name):
    if name == "BRISC":
        return _build_brisc()
    rows = read_partition(name)
    X, M = (_build_busi if name == "BUSI" else _build_ham)(rows)
    return X, M, rows


def load_dataset(name):
    cache = os.path.join(PROCESSED, "%s_%d.npz" % (name, READ_SIZE))
    if not os.path.isfile(cache):
        X, M, rows = _build(name)
        os.makedirs(PROCESSED, exist_ok=True)
        np.savez(cache + ".tmp.npz", X=X, M=M, **{k: np.array([r[k] for r in rows]) for k in ("file", "class", "split", "group")})
        os.replace(cache + ".tmp.npz", cache)
    z = np.load(cache, allow_pickle=False)
    classes = DATASETS[name]["classes"]
    d = {"X": z["X"], "M": z["M"], "FILES": z["file"].astype(str), "GROUP": z["group"].astype(str),
         "Y": np.array([classes.index(c) for c in z["class"].astype(str)])}
    split = z["split"].astype(str)
    d["IDX"] = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    return d
