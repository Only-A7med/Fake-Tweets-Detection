"""Train the BiLSTM + GloVe model (paper: 10 epochs, Adam, binary cross-entropy).

Downloads GloVe on first use. Saves one ``.keras`` artifact per protocol (the
text vectoriser is embedded, so the model consumes raw strings) plus the
training history for the training-curve figures. Every run is tracked in
MLflow (local ``mlruns/`` store), including per-epoch loss/accuracy.

``--lstm-units`` / ``--dense-units`` allow experiment variations; those runs
are tracked in MLflow but only paper-spec runs overwrite the served artifact
(pass ``--no-save`` for throwaway experiments).

Usage:
    python -m src.training.train_bilstm [--protocol paper|safe|all] [--epochs N]
        [--lstm-units N] [--dense-units N] [--no-save]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import keras
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src import config
from src.data import download
from src.models import bilstm, predict
from src.training.train_baseline import MLFLOW_EXPERIMENT, load_split


def train_protocol(
    protocol: str,
    epochs: int = config.EPOCHS,
    lstm_units: int = config.LSTM_UNITS,
    dense_units: int = config.DENSE_UNITS,
    save: bool = True,
) -> None:
    """Train the BiLSTM for one protocol's training set and log the run to MLflow.

    Args:
        protocol: "paper" or "safe" (selects the train/test CSV pair).
        epochs: Training epochs (paper: 10).
        lstm_units: LSTM units per direction (default from config).
        dense_units: Units of the penultimate Dense layer.
        save: Persist the ``.keras`` artifact and history (disable for
            experiment variations that must not overwrite the served model).
    """
    keras.utils.set_random_seed(config.SEED)

    train_csv, test_csv = config.PROTOCOL_SPLITS[protocol]
    frame = load_split(train_csv)
    texts = frame["tweet"].to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype="float32")

    glove = bilstm.load_glove(download.download_glove())
    print(f"[{protocol}] building vocabulary from {len(texts):,} training tweets ...")
    vectorizer = bilstm.build_vectorizer(texts)
    matrix = bilstm.build_embedding_matrix(vectorizer.get_vocabulary(), glove)
    covered = int((np.abs(matrix).sum(axis=1) > 0).sum())
    print(f"[{protocol}] vocabulary {len(matrix):,} tokens, {covered:,} covered by GloVe")

    model = bilstm.build_model(vectorizer, matrix, lstm_units=lstm_units, dense_units=dense_units)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"bilstm-{protocol}-lstm{lstm_units}"):
        mlflow.log_params(
            {
                "model": "bilstm",
                "protocol": protocol,
                "embedding_dim": config.EMBEDDING_DIM,
                "lstm_units": lstm_units,
                "dense_units": dense_units,
                "max_vocab": config.MAX_VOCAB,
                "max_sequence_len": config.MAX_SEQUENCE_LEN,
                "batch_size": config.BATCH_SIZE,
                "epochs": epochs,
                "train_rows": len(frame),
                "glove_coverage": covered,
                "seed": config.SEED,
            }
        )

        started = time.perf_counter()
        history = model.fit(
            tf.constant(texts.reshape(-1, 1)),
            labels,
            batch_size=config.BATCH_SIZE,
            epochs=epochs,
            verbose=2,
        )
        elapsed = time.perf_counter() - started

        for epoch, (loss, accuracy) in enumerate(
            zip(history.history["loss"], history.history["accuracy"], strict=True)
        ):
            mlflow.log_metrics({"train_loss": loss, "train_accuracy": accuracy}, step=epoch)

        test = load_split(test_csv)
        probabilities = predict.predict_proba(model, "bilstm", test["tweet"].tolist())
        test_accuracy = accuracy_score(test["label"], (probabilities >= 0.5).astype(int))
        mlflow.log_metrics({"train_seconds": elapsed, "test_accuracy": test_accuracy})

        if save:
            path = bilstm.artifact_path(protocol)
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path)
            with open(bilstm.history_path(protocol), "w", encoding="utf-8") as handle:
                json.dump(history.history, handle, indent=2)
            print(
                f"[{protocol}] BiLSTM trained for {epochs} epochs in {elapsed / 60:.1f} min "
                f"(test acc {test_accuracy:.4f}) -> {path}"
            )
        else:
            print(
                f"[{protocol}] experiment lstm={lstm_units} dense={dense_units}: "
                f"test acc {test_accuracy:.4f} (not saved)"
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=["paper", "safe", "all"], default="all", help="split to train on"
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="training epochs")
    parser.add_argument(
        "--lstm-units", type=int, default=config.LSTM_UNITS, help="LSTM units per direction"
    )
    parser.add_argument(
        "--dense-units", type=int, default=config.DENSE_UNITS, help="penultimate Dense units"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="track in MLflow only; keep the served artifact"
    )
    args = parser.parse_args(argv)

    protocols = ["paper", "safe"] if args.protocol == "all" else [args.protocol]
    for protocol in protocols:
        train_protocol(
            protocol,
            epochs=args.epochs,
            lstm_units=args.lstm_units,
            dense_units=args.dense_units,
            save=not args.no_save,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
