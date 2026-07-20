"""TF-IDF + Logistic Regression baseline (paper hyperparameters).

The paper's baseline: a 10,000-feature TF-IDF vectoriser over unigrams and
bigrams feeding a logistic regression capped at 1,000 iterations.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config


def build_baseline() -> Pipeline:
    """Create the untrained TF-IDF + LogisticRegression pipeline (paper spec)."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.TFIDF_MAX_FEATURES,
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                ),
            ),
            (
                "logreg",
                LogisticRegression(max_iter=config.LOGREG_MAX_ITER, random_state=config.SEED),
            ),
        ]
    )


def artifact_path(protocol: str) -> Path:
    """Location of the trained baseline artifact for a protocol ("paper"/"safe")."""
    return config.MODELS_DIR / protocol / "baseline.joblib"


def save_baseline(pipeline: Pipeline, protocol: str) -> Path:
    """Persist a trained pipeline; returns the artifact path."""
    path = artifact_path(protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_baseline(protocol: str) -> Pipeline:
    """Load the trained baseline for a protocol.

    Raises:
        FileNotFoundError: If the artifact does not exist yet (train it with
            ``python -m src.training.train_baseline``).
    """
    path = artifact_path(protocol)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.training.train_baseline "
            f"--protocol {protocol}` first."
        )
    return joblib.load(path)
