"""Memory-efficient tf.data pipeline for the BRISC dataset with synthesized zero-masks."""

import os
import glob
import logging
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Tuple, List, Dict, Any
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PATH & SENTINEL COMPILATION
# ============================================================================

def build_path_label_lists(
    clf_dir: str, 
    masks_dir: str
) -> Tuple[List[str], List[str], List[int]]:
    """
    Scan clf_dir/{class_name}/*.jpg for raw target images.
    If a mask file is absent, record the zero-mask sentinel string flag.
    
    Returns
    -------
    img_paths  : list[str]   absolute paths to images
    mask_paths : list[str]   absolute mask paths  OR  config.BRISC_ZERO_MASK_SENTINEL
    labels     : list[int]   integer class index
    """
    img_paths, mask_paths, labels = [], [], []

    for class_name, class_idx in config.BRISC_CLASS_TO_IDX.items():
        class_dir = os.path.join(clf_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Class folder not found, skipping: {class_dir}")
            continue

        for img_path in sorted(glob.glob(os.path.join(class_dir, "*.jpg"))):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(masks_dir, f"{basename}.png")

            # Handle non-existent masks gracefully via internal sentinel flag
            if not os.path.exists(mask_path):
                if class_name != "no_tumor":
                    logger.warning(f"Mask missing for tumor case: {basename}")
                mask_path = config.BRISC_ZERO_MASK_SENTINEL

            img_paths.append(img_path)
            mask_paths.append(mask_path)
            labels.append(class_idx)

    logger.info(
        f"Loaded {len(img_paths)} samples from {clf_dir} "
        f"| Zero-masks: {mask_paths.count(config.BRISC_ZERO_MASK_SENTINEL)}"
    )
    return img_paths, mask_paths, labels


# ============================================================================
# TENSORFLOW DATA PIPELINE ENGINES
# ============================================================================

def load_sample_tf(
    img_path: tf.Tensor, 
    mask_path: tf.Tensor, 
    label: tf.Tensor, 
    image_size: Tuple[int, int] = (config.INPUT_SIZE[0], config.INPUT_SIZE[1])
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """TF-compatible sample loader that parses or synthesizes zero masks on the fly."""
    def _read(ip, mp, lb):
        ip = ip.numpy().decode()
        mp = mp.numpy().decode()

        # ── Parse Image ──────────────────────────────────────────────────
        img = cv2.imread(ip)
        if img is None:
            logger.error(f"Could not read image: {ip}")
            img = np.zeros((*image_size, 3), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size).astype(np.float32) / 255.0

        # ── Parse Mask (Real vs Synthesized Sentinel) ───────────────────
        if mp == config.BRISC_ZERO_MASK_SENTINEL:
            msk = np.zeros((*image_size, 1), dtype=np.float32)
        else:
            msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if msk is None:
                logger.error(f"Could not read mask: {mp}")
                msk = np.zeros(image_size, dtype=np.float32)
            else:
                msk = cv2.resize(msk, image_size, interpolation=cv2.INTER_NEAREST)
                msk = (msk > 127).astype(np.float32)
            msk = msk[..., np.newaxis]

        return img, msk, np.int32(lb.numpy())

    img, msk, lbl = tf.py_function(
        _read,
        [img_path, mask_path, label],
        [tf.float32, tf.float32, tf.int32]
    )
    
    img.set_shape([image_size[0], image_size[1], 3])
    msk.set_shape([image_size[0], image_size[1], 1])
    lbl.set_shape([])

    return img, {"segmentation_output": msk, "classification_output": lbl}


def build_dataset(
    img_paths: List[str], 
    mask_paths: List[str], 
    labels: np.ndarray,
    batch_size: int = config.BATCH_SIZE, 
    shuffle: bool = False, 
    shuffle_buffer: int = 3000,
    image_size: Tuple[int, int] = (config.INPUT_SIZE[0], config.INPUT_SIZE[1])
) -> tf.data.Dataset:
    """Builds a memory-efficient stream generator via optimization mapping."""
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, labels))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda ip, mp, lb: load_sample_tf(ip, mp, lb, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================================
# MAIN DATASET FACTORY ORCHESTRATOR
# ============================================================================

def prepare_brics_datasets(
    train_clf_dir: str = None,
    train_masks_dir: str = None,
    test_clf_dir: str = None,
    test_masks_dir: str = None
) -> Dict[str, Any]:
    """
    Assembles partitioned training, validation, and testing lazy data streams.
    FIXED: Respects the official BRISC test split instead of randomly splitting.
    """
    logger.info("=" * 60)
    logger.info("BRISC MEMORY-EFFICIENT PIPELINE COMPILATION")
    logger.info("=" * 60)

    # Allow override of directories for flexibility
    train_clf_dir = train_clf_dir or config.BRISC_TRAIN_CLF_DIR
    train_masks_dir = train_masks_dir or config.BRISC_TRAIN_MASKS_DIR
    test_clf_dir = test_clf_dir or config.BRISC_TEST_CLF_DIR
    test_masks_dir = test_masks_dir or config.BRISC_TEST_MASKS_DIR

    logger.info(f"Using Batch Size: {config.BATCH_SIZE}")

    # 1. Extract Valid File Arrays from OFFICIAL TRAIN and TEST splits
    tr_img_paths, tr_mask_paths, tr_labels = build_path_label_lists(
        train_clf_dir, train_masks_dir
    )
    te_img_paths, te_mask_paths, te_labels = build_path_label_lists(
        test_clf_dir, test_masks_dir
    )
    
    if len(tr_img_paths) == 0:
        raise ValueError(f"No valid train samples found in {train_clf_dir}")
    if len(te_img_paths) == 0:
        raise ValueError(f"No valid test samples found in {test_clf_dir}")
    
    tr_labels = np.array(tr_labels, dtype=np.int32)
    te_labels = np.array(te_labels, dtype=np.int32)

    # 2. Carve validation split ONLY out of the official train pool
    # This exactly matches the original notebook logic
    idx = np.arange(len(tr_img_paths))
    tr_idx, vl_idx = train_test_split(
        idx, 
        test_size=0.15, 
        stratify=tr_labels, 
        random_state=config.RANDOM_STATE
    )

    def subset(paths, masks, labels, idxs):
        return ([paths[i] for i in idxs],
                [masks[i] for i in idxs],
                labels[idxs])

    X_train_img, X_train_msk, y_train = subset(tr_img_paths, tr_mask_paths, tr_labels, tr_idx)
    X_val_img, X_val_msk, y_val = subset(tr_img_paths, tr_mask_paths, tr_labels, vl_idx)
    
    # Official test set remains completely untouched
    X_test_img, X_test_msk, y_test = te_img_paths, te_mask_paths, te_labels

    logger.info(f"Split Configuration -> Train: {len(X_train_img)} | Val: {len(X_val_img)} | Test: {len(X_test_img)}")

    # 3. Initialize Data Pipelines Generators
    image_dim = (config.INPUT_SIZE[0], config.INPUT_SIZE[1])

    train_dataset = build_dataset(
        X_train_img, X_train_msk, y_train, 
        batch_size=config.BATCH_SIZE, shuffle=True, image_size=image_dim
    )
    val_dataset = build_dataset(
        X_val_img, X_val_msk, y_val, 
        batch_size=config.BATCH_SIZE, shuffle=False, image_size=image_dim
    )
    test_dataset = build_dataset(
        X_test_img, X_test_msk, y_test, 
        batch_size=config.BATCH_SIZE, shuffle=False, image_size=image_dim
    )

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }