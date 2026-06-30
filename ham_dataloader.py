"""Memory-efficient, lesion-aware tf.data pipeline for the HAM10000 dataset."""

import os
import logging
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Tuple, List, Dict, Any
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PATH & METADATA COMPILATION
# ============================================================================

def build_path_label_lists(
    csv_path: str, 
    images_dirs: List[str], 
    masks_dir: str, 
    categories: List[str]
) -> Tuple[List[str], List[str], List[int], List[str]]:
    """Compiles valid absolute file paths and tracking metadata from CSV file."""
    df = pd.read_csv(csv_path)
    img_paths, mask_paths, labels, lesion_ids = [], [], [], []
    cat_to_idx = config.HAM_CLASS_TO_IDX
    
    for _, row in df.iterrows():
        img_id = row["image_id"]
        dx = row["dx"].lower().strip()
        lesion_id = row["lesion_id"]
        
        if dx not in cat_to_idx:
            continue
            
        img_path = None
        
        # Scrape across split partition folders
        for d in images_dirs:
            for ext in [".jpg", ".png"]:
                p = os.path.join(d, f"{img_id}{ext}")
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path: 
                break
                
        if img_path is None: 
            continue
            
        mask_path = os.path.join(masks_dir, f"{img_id}_segmentation.png")
        if not os.path.exists(mask_path): 
            continue
            
        img_paths.append(img_path)
        mask_paths.append(mask_path)
        labels.append(cat_to_idx[dx])
        lesion_ids.append(lesion_id)
        
    return img_paths, mask_paths, labels, lesion_ids


# ============================================================================
# TENSORFLOW DATA PIPELINE ENGINES
# ============================================================================

def load_sample_tf(
    img_path: tf.Tensor, 
    mask_path: tf.Tensor, 
    label: tf.Tensor, 
    image_size: Tuple[int, int] = (config.INPUT_SIZE[0], config.INPUT_SIZE[1])
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Lazy I/O computational block utilizing explicit shape assignment."""
    def _read(ip, mp, lb):
        ip = ip.numpy().decode()
        mp = mp.numpy().decode()
        
        # Parse image
        img = cv2.imread(ip)
        if img is None:
            logger.error(f"Could not read image: {ip}")
            img = np.zeros((*image_size, 3), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size).astype(np.float32) / 255.0
        
        # Parse mask
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
    
    # Force programmatic shape assertions inside Graph execution contexts
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

def prepare_ham_datasets() -> Dict[str, Any]:
    """
    Assembles independent training, validation, and testing lazy data streams.
    Enforces a strict lesion-aware train/val/test split paradigm.
    """
    logger.info("=" * 60)
    logger.info("HAM10000 MEMORY-EFFICIENT PIPELINE COMPILATION")
    logger.info("=" * 60)
    
    logger.info(f"Using Batch Size: {config.BATCH_SIZE}")

    # 2. Extract Valid File Arrays
    img_paths, mask_paths, labels, lesion_ids = build_path_label_lists(
        config.HAM_CSV_PATH,
        images_dirs=[config.HAM_IMAGES_DIR_1, config.HAM_IMAGES_DIR_2],
        masks_dir=config.HAM_MASKS_DIR,
        categories=config.HAM_CATEGORIES
    )
    
    if len(img_paths) == 0:
        raise ValueError(f"No valid samples found. Check paths: {config.HAM_CSV_PATH}")
    
    labels = np.array(labels, dtype=np.int32)
    lesion_ids = np.array(lesion_ids, dtype=str)
    
    logger.info(f"Found total matches: {len(img_paths)} | Unique lesions: {len(np.unique(lesion_ids))}")

    # 3. Lesion-Aware Partitioning Splitting Engine
    unique_lesions = np.unique(lesion_ids)
    
    # FIXED: Isolate 30% pool for Val+Test combined (Matches original notebook logic)
    tr_lesions, tmp = train_test_split(
        unique_lesions, 
        test_size=config.VAL_SPLIT + config.TEST_SPLIT,  # 0.15 + 0.15 = 0.30
        random_state=config.RANDOM_STATE
    )
    
    # FIXED: Split the 30% pool exactly 50/50 to get 15% Val and 15% Test
    vl_lesions, te_lesions = train_test_split(
        tmp, 
        test_size=0.5, 
        random_state=config.RANDOM_STATE
    )

    tr_mask = np.isin(lesion_ids, tr_lesions)
    vl_mask = np.isin(lesion_ids, vl_lesions)
    te_mask = np.isin(lesion_ids, te_lesions)

    # Sanity integrity validations
    assert len(set(np.where(tr_mask)[0]) & set(np.where(te_mask)[0])) == 0, "Train/Test lesion overlap!"
    assert len(set(np.where(vl_mask)[0]) & set(np.where(te_mask)[0])) == 0, "Val/Test lesion overlap!"

    def subset_mask(mask):
        idxs = np.where(mask)[0]
        return ([img_paths[i] for i in idxs],
                [mask_paths[i] for i in idxs],
                labels[idxs])

    logger.info(f"Split Configuration -> Train: {tr_mask.sum()} images | Val: {vl_mask.sum()} images | Test: {te_mask.sum()} images")
    
    # Trace target allocations inside test sets
    logger.info("Test set distribution balance profile:")
    for ci, cat in enumerate(config.HAM_CATEGORIES):
        n = (labels[te_mask] == ci).sum()
        logger.info(f"  {cat}: {n}")

    # 4. Initialize Data Pipelines Generators
    image_dim = (config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    
    train_dataset = build_dataset(*subset_mask(tr_mask), batch_size=config.BATCH_SIZE, shuffle=True, image_size=image_dim)
    val_dataset = build_dataset(*subset_mask(vl_mask), batch_size=config.BATCH_SIZE, shuffle=False, image_size=image_dim)
    test_dataset = build_dataset(*subset_mask(te_mask), batch_size=config.BATCH_SIZE, shuffle=False, image_size=image_dim)

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }