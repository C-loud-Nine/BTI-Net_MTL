"""Training script for multi-task model supporting BUSI, BRISC, and HAM10000 datasets."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback

import config
from model import enhanced_bti_model
from loss import (
    enhanced_lesion_focus_loss, 
    enhanced_multi_modal_focal_loss,
    make_seg_loss, 
    make_clf_loss
)

# Import all dataset loaders
from busi_dataloader import prepare_datasets as prepare_busi_datasets
from brisc_dataloader import prepare_brics_datasets
from ham_dataloader import prepare_ham_datasets


# ============================================================================
# CUSTOM CALLBACKS (UNCHANGED)
# ============================================================================
class CompositeMetric(Callback):
    """Combined metric: 70% IoU + 30% Accuracy"""
    def on_epoch_end(self, epoch, logs=None):
        seg_iou = logs.get("val_segmentation_output_mean_io_u", 0)
        clf_acc = logs.get("val_classification_output_accuracy", 0)
        val_combined = 0.7 * seg_iou + 0.3 * clf_acc
        logs["val_combined"] = val_combined
        print(f"\nEpoch {epoch+1}: Combined Metric = {val_combined:.4f}")


class AdaptiveLossWeights(Callback):
    """Dynamically adjust loss weights during training"""
    def __init__(self):
        super().__init__()
        self.seg_weight = config.SEG_WEIGHT_START
        self.clf_weight = config.CLF_WEIGHT_START
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 15:
            self.seg_weight = min(0.92, self.seg_weight + 0.007)
            self.clf_weight = max(0.08, self.clf_weight - 0.007)
        else:
            self.seg_weight = config.SEG_WEIGHT_FINAL
            self.clf_weight = config.CLF_WEIGHT_FINAL
        
        self.model.loss_weights = {
            "segmentation_output": self.seg_weight,
            "classification_output": self.clf_weight
        }
        print(f"Loss weights: Seg={self.seg_weight:.3f}, Clf={self.clf_weight:.3f}")


class CosineDecayScheduler(Callback):
    def __init__(
        self,
        initial_lr: float,
        total_epochs: int,
        alpha: float = 0.03,
        monitor: str = "val_combined",
        factor: float = 0.5,
        patience: int = 7,
        cooldown: int = 2,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.initial_lr    = initial_lr
        self.total_epochs  = total_epochs
        self.alpha         = alpha
        self.monitor       = monitor
        self.factor        = factor
        self.patience      = patience
        self.cooldown      = cooldown
        self.min_lr        = min_lr

        self.best               = -np.inf
        self.wait               = 0
        self.cooldown_counter   = 0
        self.lr_reduction_factor = 1.0   # accumulates all plateau multipliers

    def _cosine_lr(self, epoch):
        """Pure cosine value for this epoch, before plateau scaling."""
        progress     = epoch / float(self.total_epochs)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.initial_lr * (self.alpha + (1.0 - self.alpha) * cosine_decay)

    def on_epoch_begin(self, epoch, logs=None):
        # Cosine baseline × accumulated plateau reductions, clamped to min_lr
        new_lr = max(self._cosine_lr(epoch) * self.lr_reduction_factor, self.min_lr)
        self.model.optimizer.learning_rate.assign(new_lr)
        print(f"\nEpoch {epoch+1}: LR = {new_lr:.7f}  "
              f"(cosine × {self.lr_reduction_factor:.4f})")

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        # Count down cooldown — at end of epoch, not beginning
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return  # don't touch wait counter during cooldown

        if current > self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Store the reduction — on_epoch_begin of next epoch applies it
                self.lr_reduction_factor = max(
                    self.lr_reduction_factor * self.factor,
                    self.min_lr / self.initial_lr   # don't let factor push below min_lr
                )
                current_lr = float(self.model.optimizer.learning_rate)
                projected  = max(self._cosine_lr(epoch + 1) * self.lr_reduction_factor,
                                 self.min_lr)
                print(f"\nEpoch {epoch+1}: {self.monitor} plateaued "
                      f"({self.wait} epochs). "
                      f"Next LR → {projected:.7f}  "
                      f"[reduction_factor now {self.lr_reduction_factor:.4f}]")
                self.cooldown_counter = self.cooldown
                self.wait = 0


# ============================================================================
# DATASET CONFIGURATION REGISTRY
# ============================================================================
DATASET_REGISTRY = {
    "busi": {
        "name": "BUSI (Breast Ultrasound)",
        "num_clf_classes": config.BUSI_NUM_CLF_CLASSES,
        "categories": config.BUSI_CATEGORIES,
    },
    "brisc": {
        "name": "BRISC (Brain Tumor)",
        "num_clf_classes": config.BRISC_NUM_CLF_CLASSES,
        "categories": config.BRISC_CATEGORIES,
    },
    "ham": {
        "name": "HAM10000 (Skin Lesion)",
        "num_clf_classes": config.HAM_NUM_CLF_CLASSES,
        "categories": config.HAM_CATEGORIES,
    }
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def _busi_gen(images, masks, labels):
    """Generator to avoid Windows GPU EagerConst errors with from_tensor_slices."""
    for i in range(len(images)):
        yield (images[i], 
               {"segmentation_output": masks[i], 
                "classification_output": labels[i]})

def load_busi_data():
    """Load BUSI dataset (streams in-memory numpy arrays via generator)."""
    print(f"\n Loading BUSI dataset from: {config.BUSI_BASE_DIR}")
    data = prepare_busi_datasets(config.BUSI_BASE_DIR)
    
    # Strict output signature prevents Windows GPU memory copy crashes
    output_signature = (
        tf.TensorSpec(shape=config.INPUT_SIZE, dtype=tf.float32),
        {
            "segmentation_output": tf.TensorSpec(shape=(config.INPUT_SIZE[0], config.INPUT_SIZE[1], 1), dtype=tf.float32),
            "classification_output": tf.TensorSpec(shape=(), dtype=tf.int32)
        }
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _busi_gen(data['X_train'], data['mask_train'], data['y_train']),
        output_signature=output_signature
    ).shuffle(3000, reshuffle_each_iteration=True).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: _busi_gen(data['X_val'], data['mask_val'], data['y_val']),
        output_signature=output_signature
    ).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _busi_gen(data['X_test'], data['mask_test'], data['y_test']),
        output_signature=output_signature
    ).batch(config.BATCH_SIZE)
    
    return train_dataset, val_dataset, test_dataset


def load_brisc_data():
    """Load BRISC dataset (returns lazy tf.data pipelines)."""
    print(f"\n Loading BRISC dataset from: {config.BRISC_BASE_DIR}")
    datasets = prepare_brics_datasets() 
    
    return (
        datasets['train_dataset'],
        datasets['val_dataset'],
        datasets['test_dataset']
    )


def load_ham_data():
    """Load HAM10000 dataset (returns lazy tf.data pipelines)."""
    print(f"\n Loading HAM10000 dataset from: {config.HAM_BASE_DIR}")
    datasets = prepare_ham_datasets() 
    
    return (
        datasets['train_dataset'],
        datasets['val_dataset'],
        datasets['test_dataset']
    )


DATA_LOADERS = {
    "busi": load_busi_data,
    "brisc": load_brisc_data,
    "ham": load_ham_data,
}


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Task BTI-Net Training")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["busi", "brisc", "ham"],
        required=True,
        help="Dataset to train on: busi, brisc, or ham"
    )
    args = parser.parse_args()
    
    dataset_key = args.dataset
    dataset_cfg = DATASET_REGISTRY[dataset_key]
    
    print("=" * 80)
    print(f"Multi-Task BTI-Net Training — {dataset_cfg['name']}")
    print(f"Classes ({dataset_cfg['num_clf_classes']}): {dataset_cfg['categories']}")
    print("=" * 80)
    
    # Load data
    data_loader = DATA_LOADERS[dataset_key]
    train_dataset, val_dataset, test_dataset = data_loader()
    
    # Build model
    print("\nBuilding model...")
    model = enhanced_bti_model(
        input_size=config.INPUT_SIZE,
        num_seg_classes=config.NUM_SEG_CLASSES,
        num_clf_classes=dataset_cfg['num_clf_classes'],
        dropout_rate=config.DROPOUT_RATE
    )
        
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.INITIAL_LR,
            beta_1=config.ADAM_BETA_1,
            beta_2=config.ADAM_BETA_2,
            epsilon=config.ADAM_EPSILON,
            global_clipnorm=config.GLOBAL_CLIPNORM
        ),
        loss={
            "segmentation_output": enhanced_lesion_focus_loss,
            "classification_output": enhanced_multi_modal_focal_loss
        },
        metrics={
            "segmentation_output": [
                tf.keras.metrics.MeanIoU(num_classes=2, name="mean_io_u"),
                tf.keras.metrics.BinaryAccuracy(name="bin_acc")
            ],
            "classification_output": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        },
        loss_weights={
            "segmentation_output": config.SEG_WEIGHT_START,
            "classification_output": config.CLF_WEIGHT_START
        }
    )
    
    model.summary()
    
    # Callbacks
    composite_metric = CompositeMetric()
    adaptive_weights = AdaptiveLossWeights()
    lr_scheduler = CosineDecayScheduler(
        initial_lr=config.INITIAL_LR,
        total_epochs=config.EPOCHS,
        alpha=config.COSINE_ALPHA,
        monitor=config.MONITOR,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        cooldown=config.REDUCE_LR_COOLDOWN,
        min_lr=config.MIN_LR
    )
    early_stop = EarlyStopping(
        monitor=config.MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode=config.MODE,
        restore_best_weights=True,
        verbose=1
    )
    
    # ========================================================================
    # STAGE 1: MAIN TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: MAIN TRAINING")
    print("=" * 80)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=[
            composite_metric,
            early_stop,
            lr_scheduler,
            adaptive_weights
        ],
        verbose=1
    )
    
    
    # ========================================================================
    # STAGE 2: SECONDARY TRAINING PHASE (UPA FINE-TUNING)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: SECONDARY TRAINING PHASE (UPA FINE-TUNING)")
    print("=" * 80)
    
    model_ft = model
    
    # Freeze all layers individually — do NOT touch model.trainable
    for layer in model_ft.layers:
        layer.trainable = False

    # Unfreeze UPA layers using config constants
    upa_layers = []
    
    for name in config.UPA_LAYER_NAMES:
        try:
            upa = model_ft.get_layer(name)
            upa.trainable = True
            upa_layers.append(upa)
        except ValueError:
            print(f"  Warning: Layer '{name}' not found. Skipping UPA unfreezing.")

    if not upa_layers:
        print("  Error: No UPA layers found! Skipping Stage 2.")
    else:
        # Verify trainable parameters before compiling
        trainable_count = sum(
            tf.size(v).numpy() for v in model_ft.trainable_variables
        )
        print(f"  Trainable parameters: {trainable_count:,}")

        model_ft.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.STAGE2_INITIAL_LR,
                global_clipnorm=config.GLOBAL_CLIPNORM,
            ),
            loss={
                'segmentation_output':   make_seg_loss(upa_layers, lambda_gate=config.STAGE2_LAMBDA_GATE),
                'classification_output': make_clf_loss(upa_layers, lambda_gate=config.STAGE2_LAMBDA_GATE),
            },
            metrics={
                'segmentation_output': [
                    tf.keras.metrics.MeanIoU(num_classes=2, name='mean_io_u'),
                ],
                'classification_output': [
                    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                ],
            },
        )

        stage2_callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=config.STAGE2_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, 
                verbose=1
            ),
        ]

        history2 = model_ft.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.STAGE2_EPOCHS,
            callbacks=stage2_callbacks,
            verbose=1,
        )
        
        # Save Final Fine-tuned model
        model_ft_path = os.path.join(config.CHECKPOINT_DIR, f'final_model_ft_{dataset_key}.keras')
        model_ft.save(model_ft_path)
        print(f"\n  Final fine-tuned model saved to {model_ft_path}")
    
    # ========================================================================
    # TEST EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)
    
    # Evaluate the fine-tuned model (falls back to base model if UPA skipped)
    eval_model = model_ft if upa_layers else model
    
    test_results = eval_model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Results ({dataset_cfg['name']}):")
    for name, value in zip(eval_model.metrics_names, test_results):
        print(f"  {name}: {value:.4f}")
    
    # Save results
    results_path = os.path.join(config.RESULTS_DIR, f'test_results_{dataset_key}.txt')
    with open(results_path, 'w') as f:
        f.write(f"Dataset: {dataset_cfg['name']}\n")
        f.write(f"Classes: {dataset_cfg['categories']}\n")
        f.write("-" * 40 + "\n")
        for name, value in zip(eval_model.metrics_names, test_results):
            f.write(f"{name}: {value:.6f}\n")
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()