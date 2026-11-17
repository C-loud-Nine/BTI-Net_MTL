"""Paths and dataset definitions."""
import json
import os

import numpy as np

ROOT = os.environ.get("BTINET_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "splits")

DATASETS = {
    "BUSI": {"classes": ["normal", "benign", "malignant"], "empty": "normal"},
    "BRISC": {"classes": ["glioma", "meningioma", "pituitary", "no_tumor"], "empty": "no_tumor"},
    "HAM10000": {"classes": ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"], "empty": None},
}


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(path + ".tmp", path)
