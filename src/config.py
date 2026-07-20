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

# Official Stanford NLP mirror on HF (CDN-backed; nlp.stanford.edu/data/glove.6B.zip
# serves the same archive but is heavily throttled).
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"

# Paper hyperparameters (replicate exactly).
AUGMENT_TARGET_ROWS = 200_000
WORD_SWAP_PROB = 0.2
TEST_SIZE = 0.2

# Paper protocol: augment first, then split (matches the paper; augmented
# variants of one source tweet can land on both sides of the split).
AUGMENTED_CSV = PROCESSED_DIR / "augmented.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Leakage-safe protocol: split the original tweets first, then augment only
# the training side up to the paper's training-set size. The test set stays
# un-augmented, so no source tweet (or variant of it) crosses the split.
TRAIN_TARGET_ROWS = int(AUGMENT_TARGET_ROWS * (1 - TEST_SIZE))  # 160,000
TRAIN_SAFE_CSV = PROCESSED_DIR / "train_safe.csv"
TEST_SAFE_CSV = PROCESSED_DIR / "test_safe.csv"

# Train/test CSVs per evaluation protocol.
PROTOCOL_SPLITS = {
    "paper": (TRAIN_CSV, TEST_CSV),
    "safe": (TRAIN_SAFE_CSV, TEST_SAFE_CSV),
}

# Baseline hyperparameters (paper: TF-IDF 10k features, uni+bigrams; LogReg
# capped at 1,000 iterations).
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 1000

# BiLSTM hyperparameters. Embedding dim, epochs, loss, and optimizer come from
# the paper; the sizes the paper does not specify are marked and were chosen
# from the EDA (sequence length covers >99% of tweets) or common defaults.
EMBEDDING_DIM = 100
EPOCHS = 10
LSTM_UNITS = 64  # not specified in the paper
DENSE_UNITS = 64  # not specified in the paper
MAX_VOCAB = 50_000  # not specified in the paper
MAX_SEQUENCE_LEN = 64  # not specified in the paper; >99% of tweets fit
BATCH_SIZE = 128  # not specified in the paper

# Evaluation outputs.
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
