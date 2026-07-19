"""Split the augmented dataset 80/20 into train and test CSVs (seed 42).

The split happens after augmentation, matching the paper (its 40,000-tweet
test set is 20% of the 200,000 augmented rows).

Usage:
    python -m src.data.split   # expects data/processed/augmented.csv
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def split_dataframe(
    frame: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a dataframe into train and test parts.

    Args:
        frame: Dataset to split.
        test_size: Fraction of rows assigned to the test set (paper: 0.2).
        seed: Random seed for the shuffle.

    Returns:
        ``(train, test)`` dataframes with fresh integer indexes.
    """
    train, test = train_test_split(frame, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> int:
    """Split ``augmented.csv`` and write ``train.csv`` / ``test.csv``."""
    if not config.AUGMENTED_CSV.exists():
        print(
            f"{config.AUGMENTED_CSV} not found - run `python -m src.data.preprocess` first.",
            file=sys.stderr,
        )
        return 1

    # keep_default_na=False so tweets like "null" or "NA" stay strings.
    frame = pd.read_csv(config.AUGMENTED_CSV, keep_default_na=False, dtype={"tweet": str})
    frame["label"] = frame["label"].astype(int)
    train, test = split_dataframe(frame)
    train.to_csv(config.TRAIN_CSV, index=False)
    test.to_csv(config.TEST_CSV, index=False)
    print(f"train: {len(train):,} rows -> {config.TRAIN_CSV}")
    print(f"test:  {len(test):,} rows -> {config.TEST_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
