"""Data loading and preprocessing for BUSI dataset."""

import os
import glob
import logging
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from typing import Tuple, Dict
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image_and_merge_masks(
    image_path: str,
    expected_shape: Tuple[int, int] = None,
) -> np.ndarray:
    """Merge multiple mask files using pixel-wise maximum."""
    if expected_shape is None:
        expected_shape = (config.INPUT_SIZE[0], config.INPUT_SIZE[1])
        
    mask_pattern = image_path.replace(".png", "_mask*.png")
    mask_files = sorted(glob.glob(mask_pattern))
    
    if not mask_files:
        raise FileNotFoundError(f"No masks found for {image_path}")
    
    merged_mask = None
    valid_count = 0
    
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Could not read mask: {mask_file}")
            continue
        mask = cv2.resize(mask, expected_shape, interpolation=cv2.INTER_NEAREST)
        merged_mask = mask if merged_mask is None else np.maximum(merged_mask, mask)
        valid_count += 1
    
    if merged_mask is None or valid_count == 0:
        raise ValueError(f"No valid masks for {image_path}")
    
    return merged_mask


def load_busi_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BUSI dataset with images, masks, and labels using global configuration profiles.
    """
    images, masks, class_labels = [], [], []
    target_size = (config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    
    for category in config.BUSI_CATEGORIES:
        category_folder = os.path.join(dataset_path, category)
        if not os.path.exists(category_folder):
            logger.warning(f"Category folder missing: {category_folder}")
            continue
            
        files = os.listdir(category_folder)
        
        for file in files:
            if not file.endswith(".png") or "_mask" in file:
                continue
            
            image_path = os.path.join(category_folder, file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, target_size)
                image = image.astype(np.float32) / 255.0
                
                # Load and merge masks
                mask = load_image_and_merge_masks(image_path, target_size)
                mask = (mask > 127).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)
                
                # Validate shapes matching setup
                if image.shape != (*target_size, 3) or mask.shape != (*target_size, 1):
                    logger.warning(f"Invalid shape in {file}: img={image.shape}, mask={mask.shape}")
                    continue
                
                images.append(image)
                masks.append(mask)
                # FIXED: Use BUSI_CLASS_TO_IDX for consistent label assignment
                class_labels.append(config.BUSI_CLASS_TO_IDX[category])
                
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue
    
    if len(images) == 0:
        raise ValueError(f"No valid samples found in {dataset_path}")
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32), np.array(class_labels, dtype=np.int32)


def augment_dataset(images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate augmented versions of training data based on config probabilities."""
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    
    transform = A.Compose([
        A.HorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        A.VerticalFlip(p=config.VERTICAL_FLIP_PROB),
        A.Rotate(limit=config.ROTATION_LIMIT, p=config.ROTATION_PROB),
    ])
    
    augmented_images = [images]
    augmented_masks = [masks]
    
    for _ in range(config.NUM_AUGMENTATIONS):
        aug_imgs, aug_msks = [], []
        for img, msk in zip(images, masks):
            augmented = transform(image=img, mask=msk)
            aug_imgs.append(augmented['image'])
            aug_msks.append(augmented['mask'])
        augmented_images.append(np.array(aug_imgs))
        augmented_masks.append(np.array(aug_msks))
    
    return np.concatenate(augmented_images), np.concatenate(augmented_masks)


def prepare_datasets(dataset_path: str = None) -> Dict[str, np.ndarray]:
    """
    Load, split, and augment the BUSI dataset systematically utilizing config boundaries.
    """
    dataset_path = dataset_path or config.BUSI_BASE_DIR
    
    logger.info("=" * 60)
    logger.info("BUSI DATASET PIPELINE COMPILATION")
    logger.info("=" * 60)
    logger.info(f"Loading BUSI dataset from: {dataset_path}")
    
    images, masks, labels = load_busi_dataset(dataset_path)
    
    logger.info(f"Loaded {len(images)} samples")
    logger.info(f"Class distribution: {dict(zip(config.BUSI_CATEGORIES, np.bincount(labels)))}")
    
    # First split: Isolate Test Split out of raw dataset chunk
    X_train_val, X_test, y_train_val, y_test, mask_train_val, mask_test = train_test_split(
        images, labels, masks,
        test_size=config.TEST_SPLIT,
        stratify=labels,
        random_state=config.RANDOM_STATE
    )
    
    # Second split: Extract Validation Split out of remainders
    val_relative_ratio = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(
        X_train_val, y_train_val, mask_train_val,
        test_size=val_relative_ratio,
        stratify=y_train_val,
        random_state=config.RANDOM_STATE
    )
    
    # Augment training data
    logger.info("Augmenting training data...")
    X_train, mask_train = augment_dataset(X_train, mask_train)
    # NUM_AUGMENTATIONS copies + 1 original
    y_train = np.tile(y_train, config.NUM_AUGMENTATIONS + 1)
    
    # Shuffle augmented training data
    np.random.seed(config.RANDOM_STATE)
    idx = np.random.permutation(len(X_train))
    X_train, mask_train, y_train = X_train[idx], mask_train[idx], y_train[idx]
    
    # Expand mask dimensions back to channel structure if needed
    if mask_train.ndim == 3: mask_train = np.expand_dims(mask_train, -1)
    if mask_val.ndim == 3: mask_val = np.expand_dims(mask_val, -1)
    if mask_test.ndim == 3: mask_test = np.expand_dims(mask_test, -1)
    
    logger.info(f"Split Configuration -> Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    logger.info(f"Train labels: {np.bincount(y_train)}")
    logger.info(f"Val labels: {np.bincount(y_val)}")
    logger.info(f"Test labels: {np.bincount(y_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'mask_train': mask_train,
        'X_val': X_val, 'y_val': y_val, 'mask_val': mask_val,
        'X_test': X_test, 'y_test': y_test, 'mask_test': mask_test,
    }