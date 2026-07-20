"""Evaluate trained models: metrics table, confusion matrices, training curves.

For each protocol ("paper" and "safe") this evaluates every trained model on
that protocol's test set and produces:

* a metrics table (accuracy / precision / recall / F1) printed to stdout in
  GitHub-Markdown form, ready to paste into the README;
* ``reports/metrics.json`` with the raw numbers;
* ``reports/figures/confusion_<model>_<protocol>.png`` per model/protocol;
* ``reports/figures/training_curves_<protocol>.png`` when a BiLSTM training
  history exists.

Usage:
    python -m src.training.evaluate [--protocol paper|safe|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config
from src.models import bilstm, predict
from src.training.train_baseline import load_split

CLASS_NAMES = ["fake", "real"]  # index = label value (0 = fake, 1 = real)


def evaluate_model(model_name: predict.ModelName, protocol: str) -> dict:
    """Compute test-set metrics for one trained model under one protocol.

    Args:
        model_name: "baseline" or "bilstm".
        protocol: "paper" or "safe".

    Returns:
        Dict with accuracy/precision/recall/f1 (floats), n_test, and the
        confusion matrix as a nested list ``[[tn, fp], [fn, tp]]``.
    """
    _, test_csv = config.PROTOCOL_SPLITS[protocol]
    frame = load_split(test_csv)
    model = predict.load_model(model_name, protocol)
    probabilities = predict.predict_proba(model, model_name, frame["tweet"].tolist())
    predictions = (probabilities >= 0.5).astype(int)
    labels = frame["label"].to_numpy()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "n_test": int(len(labels)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def plot_confusion(matrix: list[list[int]], model_name: str, protocol: str) -> Path:
    """Save a confusion-matrix heatmap PNG; returns the figure path."""
    counts = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(4.2, 3.6))
    image = axis.imshow(counts, cmap="Blues")
    for (row, col), value in np.ndenumerate(counts):
        color = "white" if value > counts.max() / 2 else "black"
        axis.text(col, row, f"{value:,}", ha="center", va="center", color=color)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"{model_name} — {protocol} protocol")
    figure.colorbar(image, fraction=0.046)
    figure.tight_layout()

    path = config.FIGURES_DIR / f"confusion_{model_name}_{protocol}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_training_curves(protocol: str) -> Path | None:
    """Save loss/accuracy-per-epoch curves for the BiLSTM, if history exists."""
    history_file = bilstm.history_path(protocol)
    if not history_file.exists():
        return None
    history = json.loads(history_file.read_text(encoding="utf-8"))

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = range(1, len(history["loss"]) + 1)
    loss_axis.plot(epochs, history["loss"], marker="o")
    loss_axis.set_title("Training loss")
    loss_axis.set_xlabel("Epoch")
    accuracy_axis.plot(epochs, history["accuracy"], marker="o")
    accuracy_axis.set_title("Training accuracy")
    accuracy_axis.set_xlabel("Epoch")
    figure.suptitle(f"BiLSTM training — {protocol} protocol")
    figure.tight_layout()

    path = config.FIGURES_DIR / f"training_curves_{protocol}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def markdown_table(results: dict[str, dict[str, dict]]) -> str:
    """Render nested ``{protocol: {model: metrics}}`` results as a Markdown table."""
    lines = [
        "| Protocol | Model | Accuracy | Precision | Recall | F1 | Test n |",
        "|---|---|---|---|---|---|---|",
    ]
    for protocol, models in results.items():
        for model_name, metrics in models.items():
            lines.append(
                f"| {protocol} | {model_name} "
                f"| {metrics['accuracy']:.4f} | {metrics['precision']:.4f} "
                f"| {metrics['recall']:.4f} | {metrics['f1']:.4f} "
                f"| {metrics['n_test']:,} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Evaluates whichever model artifacts exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=["paper", "safe", "all"], default="all", help="which test set to use"
    )
    args = parser.parse_args(argv)
    protocols = ["paper", "safe"] if args.protocol == "all" else [args.protocol]

    results: dict[str, dict[str, dict]] = {}
    for protocol in protocols:
        results[protocol] = {}
        for model_name in ("baseline", "bilstm"):
            try:
                metrics = evaluate_model(model_name, protocol)
            except FileNotFoundError as error:
                print(f"[skip] {error}")
                continue
            results[protocol][model_name] = metrics
            figure_path = plot_confusion(metrics["confusion_matrix"], model_name, protocol)
            print(f"[{protocol}] {model_name}: acc={metrics['accuracy']:.4f} -> {figure_path}")
        curves = plot_training_curves(protocol)
        if curves:
            print(f"[{protocol}] training curves -> {curves}")

    if not any(results.values()):
        print("No trained models found; train them first (see src/training/).")
        return 1

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_json = config.REPORTS_DIR / "metrics.json"
    existing = json.loads(metrics_json.read_text(encoding="utf-8")) if metrics_json.exists() else {}
    for protocol, models in results.items():
        existing.setdefault(protocol, {}).update(models)
    metrics_json.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to {metrics_json}\n")
    print(markdown_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
