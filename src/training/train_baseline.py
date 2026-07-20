"""Train the TF-IDF + LogisticRegression baseline.

Each run is tracked in MLflow (local ``mlruns/`` store): hyperparameters,
training time, held-out test accuracy, and the saved artifact.

Usage:
    python -m src.training.train_baseline [--protocol paper|safe|all]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

from src import config
from src.models import baseline

MLFLOW_EXPERIMENT = "tweetguard"


def load_split(csv_path: Path) -> pd.DataFrame:
    """Load a processed train/test CSV, keeping tweets like "null" as strings.

    Raises:
        FileNotFoundError: If the CSV has not been generated yet.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found - generate it with `python -m src.data.preprocess`"
            " (add --leakage-safe for the safe protocol)."
        )
    frame = pd.read_csv(csv_path, keep_default_na=False, dtype={"tweet": str})
    frame["label"] = frame["label"].astype(int)
    return frame


def train_protocol(protocol: str) -> Path:
    """Fit the baseline on a protocol's training set and save the artifact."""
    train_csv, test_csv = config.PROTOCOL_SPLITS[protocol]
    frame = load_split(train_csv)
    pipeline = baseline.build_baseline()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"baseline-{protocol}"):
        mlflow.log_params(
            {
                "model": "baseline",
                "protocol": protocol,
                "tfidf_max_features": config.TFIDF_MAX_FEATURES,
                "tfidf_ngram_range": str(config.TFIDF_NGRAM_RANGE),
                "logreg_max_iter": config.LOGREG_MAX_ITER,
                "train_rows": len(frame),
                "seed": config.SEED,
            }
        )

        started = time.perf_counter()
        pipeline.fit(frame["tweet"], frame["label"])
        elapsed = time.perf_counter() - started

        test = load_split(test_csv)
        test_accuracy = accuracy_score(test["label"], pipeline.predict(test["tweet"]))

        path = baseline.save_baseline(pipeline, protocol)
        mlflow.log_metrics({"train_seconds": elapsed, "test_accuracy": test_accuracy})
        mlflow.log_artifact(str(path))

    print(
        f"[{protocol}] baseline trained on {len(frame):,} rows in {elapsed:.1f}s "
        f"(test acc {test_accuracy:.4f}) -> {path}"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=["paper", "safe", "all"], default="all", help="split to train on"
    )
    args = parser.parse_args(argv)

    protocols = ["paper", "safe"] if args.protocol == "all" else [args.protocol]
    for protocol in protocols:
        train_protocol(protocol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
