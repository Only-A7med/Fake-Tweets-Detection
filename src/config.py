"""Central configuration: paths, dataset identifiers, and paper hyperparameters.

The augmentation/split constants come from the paper (PLAN.md §4) and must not
be changed if the goal is to reproduce its results.
"""

from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = Path.home() / ".cache" / "tweetguard"

# TruthSeeker 2023 (UNB CIC), mirrored on Kaggle. Public — kagglehub needs no API key.
KAGGLE_DATASET = "sudishbasnet/truthseekertwitterdataset2023"
DATASET_CSV_NAME = "Truth_Seeker_Model_Dataset.csv"
TEXT_COLUMN = "tweet"
LABEL_COLUMN = "BinaryNumTarget"  # 1 = real, 0 = fake

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"

# Paper hyperparameters (replicate exactly).
AUGMENT_TARGET_ROWS = 200_000
WORD_SWAP_PROB = 0.2
TEST_SIZE = 0.2

AUGMENTED_CSV = PROCESSED_DIR / "augmented.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
