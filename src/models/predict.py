"""Unified prediction interface over both trained models.

Used by the evaluation script (batch) and the Flask app (single tweet). All
probabilities are P(real); labels follow the dataset convention 1 = real,
0 = fake.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import tensorflow as tf

from src.models import baseline, bilstm

ModelName = Literal["baseline", "bilstm"]


def load_model(model_name: ModelName, protocol: str):
    """Load a trained model artifact ("baseline" or "bilstm") for a protocol."""
    if model_name == "baseline":
        return baseline.load_baseline(protocol)
    if model_name == "bilstm":
        return bilstm.load_bilstm(protocol)
    raise ValueError(f"unknown model {model_name!r}; expected 'baseline' or 'bilstm'")


def predict_proba(model, model_name: ModelName, texts: list[str] | np.ndarray) -> np.ndarray:
    """Return P(real) for each text under the given loaded model.

    Args:
        model: Object returned by :func:`load_model`.
        model_name: Which kind of model ``model`` is.
        texts: Tweet texts.

    Returns:
        Float array of shape ``(len(texts),)`` with probabilities of class 1.
    """
    if model_name == "baseline":
        real_column = list(model.classes_).index(1)
        return model.predict_proba(texts)[:, real_column]
    if model_name == "bilstm":
        # Keras 3 only accepts strings as tf.string tensors, not numpy arrays.
        batch = tf.constant(np.asarray(texts, dtype=str).reshape(-1, 1))
        return model.predict(batch, verbose=0).ravel()
    raise ValueError(f"unknown model {model_name!r}; expected 'baseline' or 'bilstm'")


def decide(probability: float) -> tuple[str, float]:
    """Turn P(real) into a label and a confidence in that label.

    Returns:
        ``(label, confidence)`` where label is ``"real"`` or ``"fake"`` and
        confidence is the probability of the chosen label.
    """
    if probability >= 0.5:
        return "real", float(probability)
    return "fake", float(1.0 - probability)


def predict_one(text: str, model_name: ModelName, protocol: str) -> dict:
    """Classify a single tweet with a stored model.

    Args:
        text: Tweet text.
        model_name: "baseline" or "bilstm".
        protocol: Which training protocol's artifact to use ("paper"/"safe").

    Returns:
        Dict with ``model``, ``protocol``, ``label`` and ``confidence`` keys.
    """
    model = load_model(model_name, protocol)
    probability = float(predict_proba(model, model_name, [text])[0])
    label, confidence = decide(probability)
    return {
        "model": model_name,
        "protocol": protocol,
        "label": label,
        "confidence": round(confidence, 4),
    }
