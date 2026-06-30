import os

# ============================================================================
# SHARED MODEL & TRAINING CONFIGURATION
# ============================================================================
INPUT_SIZE = (224, 224, 3)
DECODER_CHANNELS = [384, 192, 96, 48]
NUM_SEG_CLASSES = 1
DROPOUT_RATE = 0.3

BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
EPOCHS = 50

# Optimizer (Adam)
INITIAL_LR = 3.0e-4
MIN_LR = 1.5e-6
COSINE_ALPHA = 0.03
ADAM_BETA_1 = 0.91
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-07
GLOBAL_CLIPNORM = 1.0

# Data Split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# Loss Weights
SEG_WEIGHT_START = 0.82
CLF_WEIGHT_START = 0.18
SEG_WEIGHT_FINAL = 0.80
CLF_WEIGHT_FINAL = 0.20

# Callbacks
EARLY_STOPPING_PATIENCE = 22
EARLY_STOPPING_MIN_DELTA = 0.004
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.55
REDUCE_LR_COOLDOWN = 2
MONITOR = "val_combined"
MODE = "max"

# ============================================================================
# STAGE 2: SECONDARY TRAINING (UGA FINE-TUNING) CONSTANTS
# ============================================================================
STAGE2_EPOCHS = 15
STAGE2_INITIAL_LR = 3.0e-4
STAGE2_LAMBDA_GATE = 1.0
STAGE2_EARLY_STOPPING_PATIENCE = 5
UPA_LAYER_NAMES = [
    'upa_d1', 
    'upa_d2',
    'upa_d3', 
    'upa_d4'
]

# Augmentation
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_LIMIT = 15
ROTATION_PROB = 0.7
NUM_AUGMENTATIONS = 3  # Generates 3 augmented copies + 1 original = 4x total

# Global Outputs
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# BUSI DATASET CONSTANTS
# ============================================================================
BUSI_BASE_DIR = './data/busi'
BUSI_NUM_CLF_CLASSES = 3
# FIXED: Order must match original notebook: benign=0, malignant=1, normal=2
BUSI_CATEGORIES = ['benign', 'malignant', 'normal']
BUSI_CLASS_TO_IDX = {c: i for i, c in enumerate(BUSI_CATEGORIES)}


# ============================================================================
# BRISC DATASET CONSTANTS
# ============================================================================
BRISC_BASE_DIR = './data/brisc'
BRISC_NUM_CLF_CLASSES = 4
# Order: glioma=0, meningioma=1, pituitary=2, no_tumor=3
BRISC_CATEGORIES = ["glioma", "meningioma", "pituitary", "no_tumor"]
BRISC_CLASS_TO_IDX = {c: i for i, c in enumerate(BRISC_CATEGORIES)}
BRISC_NO_TUMOR_IDX = BRISC_CLASS_TO_IDX["no_tumor"]
# Sentinel string used to signal "create zero mask at load time"
BRISC_ZERO_MASK_SENTINEL = "__ZERO__"

BRISC_TRAIN_CLF_DIR = os.path.join(BRISC_BASE_DIR, "classification_task", "train")
BRISC_TRAIN_MASKS_DIR = os.path.join(BRISC_BASE_DIR, "segmentation_task", "train", "masks")
BRISC_TEST_CLF_DIR = os.path.join(BRISC_BASE_DIR, "classification_task", "test")
BRISC_TEST_MASKS_DIR = os.path.join(BRISC_BASE_DIR, "segmentation_task", "test", "masks")


# ============================================================================
# HAM10000 DATASET CONSTANTS
# ============================================================================
HAM_BASE_DIR = './data/ham'
HAM_NUM_CLF_CLASSES = 7
# FIXED: Order must match original notebook: mel=0, nv=1, bcc=2, akiec=3, bkl=4, df=5, vasc=6
HAM_CATEGORIES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
HAM_CLASS_TO_IDX = {c: i for i, c in enumerate(HAM_CATEGORIES)}

HAM_IMAGES_DIR_1 = os.path.join(HAM_BASE_DIR, "HAM10000_images_part_1")
HAM_IMAGES_DIR_2 = os.path.join(HAM_BASE_DIR, "HAM10000_images_part_2")
HAM_MASKS_DIR = os.path.join(HAM_BASE_DIR, "HAM10000_segmentations_lesion_tschandl")
HAM_CSV_PATH = os.path.join(HAM_BASE_DIR, "HAM10000_metadata.csv")