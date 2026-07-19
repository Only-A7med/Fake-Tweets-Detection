"""Clean and augment the TruthSeeker dataset, then trigger the train/test split.

Replicates the paper's preprocessing exactly (PLAN.md §4):

1. Keep the tweet text (``tweet``) and binary label (``BinaryNumTarget``,
   1 = real, 0 = fake).
2. Fill missing text with empty strings, then drop entries whose tweet is
   empty (the paper's "removal of empty entries").
3. Word-swap augmentation: each word is replaced, with probability 0.2, by a
   random word drawn from the corpus vocabulary ("words from the same or
   other tweets"), growing the dataset to 200,000 rows.
4. Hand the augmented dataset to ``src.data.split`` for the 80/20 split.

Usage:
    python -m src.data.preprocess
"""

from __future__ import annotations

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


def main() -> int:
    """Run the full pipeline: download, clean, augment, save, and split."""
    csv_path = download.download_dataset()
    print(f"Loading {csv_path} ...")
    frame = load_and_clean(csv_path)
    real, fake = int(frame["label"].sum()), int((frame["label"] == 0).sum())
    print(f"Clean rows: {len(frame):,} (real={real:,}, fake={fake:,})")

    frame = augment(frame)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.AUGMENTED_CSV, index=False)
    print(f"Augmented dataset: {len(frame):,} rows -> {config.AUGMENTED_CSV}")
    return split.main()


if __name__ == "__main__":
    sys.exit(main())
