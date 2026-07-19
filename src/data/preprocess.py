"""Clean and augment the TruthSeeker dataset, then produce train/test CSVs.

Cleaning replicates the paper exactly (PLAN.md §4): keep the tweet text
(``tweet``) and binary label (``BinaryNumTarget``, 1 = real, 0 = fake), fill
missing text with empty strings, drop entries whose tweet is empty, then apply
word-swap augmentation — each word is replaced, with probability 0.2, by a
random word drawn from the corpus vocabulary ("words from the same or other
tweets").

Two split protocols are supported:

- **paper** (default): augment to 200,000 rows first, then split 80/20 —
  exactly what the paper describes. Because variants of one source tweet can
  land on both sides of the split, this leaks training information into the
  test set; it is kept as the default because this repo is a reproduction.
- **leakage-safe** (``--leakage-safe``): split the original tweets 80/20
  first, then augment only the training side up to the paper's training-set
  size (160,000 rows). The test set contains only original, unseen tweets.

Usage:
    python -m src.data.preprocess                  # paper protocol
    python -m src.data.preprocess --leakage-safe   # leakage-safe protocol
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd

from src import config
from src.data import download, split

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Normalise one tweet: missing values become "", whitespace is collapsed.

    The paper applies minimal cleaning (missing values to empty strings plus
    text-format standardisation). Lowercasing and tokenisation are left to the
    vectorisers in M2, which handle them the same way the paper's models did.

    Args:
        text: Raw cell value; may be a string, None/NaN, or any scalar.

    Returns:
        Cleaned tweet text (possibly empty).
    """
    if isinstance(text, str):
        return _WHITESPACE.sub(" ", text).strip()
    if text is None or pd.isna(text):
        return ""
    return _WHITESPACE.sub(" ", str(text)).strip()


def load_and_clean(csv_path: Path | str) -> pd.DataFrame:
    """Load the raw TruthSeeker CSV and return a clean two-column frame.

    Rows without a label and rows whose tweet is empty after cleaning are
    dropped (the paper's "removal of empty entries").

    Args:
        csv_path: Path to ``Truth_Seeker_Model_Dataset.csv``.

    Returns:
        DataFrame with columns ``tweet`` (str) and ``label`` (int, 1 = real).
    """
    raw = pd.read_csv(csv_path, usecols=[config.TEXT_COLUMN, config.LABEL_COLUMN])
    raw = raw.dropna(subset=[config.LABEL_COLUMN])
    frame = pd.DataFrame(
        {
            "tweet": raw[config.TEXT_COLUMN].map(clean_text),
            "label": raw[config.LABEL_COLUMN].astype(float).astype(int),
        }
    )
    return frame[frame["tweet"] != ""].reset_index(drop=True)


def build_vocabulary(tweets: pd.Series | list[str]) -> list[str]:
    """Collect the unique words used across all tweets (the swap pool).

    The result is sorted so that seeded augmentation is reproducible across
    runs regardless of set iteration order.

    Args:
        tweets: Cleaned tweet texts.

    Returns:
        Sorted list of unique whitespace-delimited words.
    """
    vocabulary: set[str] = set()
    for tweet in tweets:
        vocabulary.update(tweet.split())
    return sorted(vocabulary)


def word_swap(
    tweet: str,
    vocabulary: list[str],
    rng: random.Random,
    swap_prob: float = config.WORD_SWAP_PROB,
) -> str:
    """Replace each word, with probability ``swap_prob``, by a random vocabulary word.

    Args:
        tweet: Cleaned tweet text.
        vocabulary: Swap pool built from the whole corpus.
        rng: Seeded random generator (drives both the swap decision and the
            replacement choice).
        swap_prob: Per-word replacement probability (paper: 0.2).

    Returns:
        The augmented tweet.
    """
    words = [rng.choice(vocabulary) if rng.random() < swap_prob else word for word in tweet.split()]
    return " ".join(words)


def augment(
    frame: pd.DataFrame,
    target_rows: int = config.AUGMENT_TARGET_ROWS,
    swap_prob: float = config.WORD_SWAP_PROB,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Grow the dataset to ``target_rows`` with word-swapped copies (paper method).

    Original rows are kept untouched and come first; the shortfall is filled by
    sampling tweets with replacement and word-swapping each copy. Labels are
    carried over from the sampled originals.

    Args:
        frame: Clean dataset with ``tweet`` and ``label`` columns.
        target_rows: Final size of the augmented dataset (paper: 200,000).
        swap_prob: Per-word replacement probability (paper: 0.2).
        seed: Seed for sampling and word swapping.

    Returns:
        DataFrame with exactly ``target_rows`` rows, or an unchanged copy if
        the input already has at least that many.
    """
    if len(frame) >= target_rows:
        return frame.copy()

    rng = random.Random(seed)
    vocabulary = build_vocabulary(frame["tweet"])
    source_rows = frame.sample(n=target_rows - len(frame), replace=True, random_state=seed)
    augmented = pd.DataFrame(
        {
            "tweet": [word_swap(t, vocabulary, rng, swap_prob) for t in source_rows["tweet"]],
            "label": source_rows["label"].to_numpy(),
        }
    )
    return pd.concat([frame, augmented], ignore_index=True)


def split_first_then_augment(
    frame: pd.DataFrame,
    train_target_rows: int = config.TRAIN_TARGET_ROWS,
    test_size: float = config.TEST_SIZE,
    swap_prob: float = config.WORD_SWAP_PROB,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce a leakage-safe train/test pair: split originals, augment train only.

    Exact duplicate tweet strings are collapsed first (the source data carries
    a few hundred, all with consistent labels) — otherwise copies of one tweet
    could land on both sides of a row-level split. The 80/20 split then happens
    on unique original tweets, so every test row is an unmodified source tweet
    absent from training. Augmentation grows only the training side, sampling
    replacement words exclusively from the training vocabulary — nothing
    derived from a test tweet can enter training.

    Args:
        frame: Clean dataset with ``tweet`` and ``label`` columns.
        train_target_rows: Final training-set size (default: 160,000, the
            paper protocol's training-set size, for comparability).
        test_size: Fraction of unique original rows held out for testing.
        swap_prob: Per-word replacement probability (paper: 0.2).
        seed: Seed for the split, sampling, and word swapping.

    Returns:
        ``(train, test)`` dataframes; ``train`` is augmented, ``test`` is not.
    """
    unique = frame.drop_duplicates(subset="tweet").reset_index(drop=True)
    train, test = split.split_dataframe(unique, test_size=test_size, seed=seed)
    train = augment(train, target_rows=train_target_rows, swap_prob=swap_prob, seed=seed)
    return train, test


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline: download, clean, augment, and write train/test CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leakage-safe",
        action="store_true",
        help="split original tweets first, then augment only the training set",
    )
    args = parser.parse_args(argv)

    csv_path = download.download_dataset()
    print(f"Loading {csv_path} ...")
    frame = load_and_clean(csv_path)
    real, fake = int(frame["label"].sum()), int((frame["label"] == 0).sum())
    print(f"Clean rows: {len(frame):,} (real={real:,}, fake={fake:,})")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.leakage_safe:
        train, test = split_first_then_augment(frame)
        train.to_csv(config.TRAIN_SAFE_CSV, index=False)
        test.to_csv(config.TEST_SAFE_CSV, index=False)
        print(f"train (augmented): {len(train):,} rows -> {config.TRAIN_SAFE_CSV}")
        print(f"test (originals only): {len(test):,} rows -> {config.TEST_SAFE_CSV}")
        return 0

    frame = augment(frame)
    frame.to_csv(config.AUGMENTED_CSV, index=False)
    print(f"Augmented dataset: {len(frame):,} rows -> {config.AUGMENTED_CSV}")
    return split.main()


if __name__ == "__main__":
    sys.exit(main())
