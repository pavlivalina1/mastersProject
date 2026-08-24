from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PLANTVILLAGE_RAW_DIR = DATA_RAW_DIR / "plantvillage" / "raw" / "color"
PLANTDOC_RAW_DIR = DATA_RAW_DIR / "plantdoc"

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_TRAIN = DATA_PROCESSED_DIR / "train.csv"
MANIFEST_VAL = DATA_PROCESSED_DIR / "val.csv"
MANIFEST_TEST = DATA_PROCESSED_DIR / "test.csv"
MANIFEST_OOD = DATA_PROCESSED_DIR / "ood_plantdoc.csv"
MANIFEST_OOD_TRAIN = DATA_PROCESSED_DIR / "ood_plantdoc_train.csv"
MANIFEST_OOD_VAL = DATA_PROCESSED_DIR / "ood_plantdoc_val.csv"
MANIFEST_OOD_TEST = DATA_PROCESSED_DIR / "ood_plantdoc_test.csv"
MANIFEST_JOINT_TRAIN = DATA_PROCESSED_DIR / "joint_train.csv"
MANIFEST_JOINT_VAL = DATA_PROCESSED_DIR / "joint_val.csv"
CLASS_LIST_PATH = DATA_PROCESSED_DIR / "classes.json"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
XAI_DIR = EXPERIMENTS_DIR / "xai"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42
IMAGE_SIZE = 128
IMAGES_PER_CLASS = 350
SPLIT_RATIOS = (0.70, 0.15, 0.15)

BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MODEL_NAMES = [
    "custom_cnn",
    "resnet18",
    "mobilenet_v2",
    "efficientnet_b0",
    "shufflenet_v2",
]

CV_FOLDS = 3

ROBUSTNESS_LEVELS = [0, 1, 2, 3]

DOMAIN_ADAPT_VAL_RATIO = 0.15
DOMAIN_ADAPT_MAX_EPOCHS = 20
DOMAIN_ADAPT_PATIENCE = 4
DOMAIN_ADAPT_LEARNING_RATE = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

for _dir in (DATA_PROCESSED_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR, XAI_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
