"""BiLSTM + GloVe architecture (paper spec) built on Keras 3.

Architecture from the paper: GloVe 100-d embeddings -> Bidirectional LSTM ->
global max pooling -> Dense ReLU -> Dense sigmoid, trained with Adam on
binary cross-entropy. Text vectorisation lives inside the model, so the
single ``.keras`` artifact accepts raw tweet strings (the tokenizer the
paper stores separately is embedded in the artifact here).
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from src import config


def artifact_path(protocol: str) -> Path:
    """Location of the trained BiLSTM artifact for a protocol ("paper"/"safe")."""
    return config.MODELS_DIR / protocol / "bilstm.keras"


def history_path(protocol: str) -> Path:
    """Location of the training-history JSON for a protocol."""
    return config.MODELS_DIR / protocol / "bilstm_history.json"


def load_glove(path: Path | str) -> dict[str, np.ndarray]:
    """Parse a GloVe text file into a word -> vector mapping.

    Args:
        path: Path to ``glove.6B.100d.txt``.

    Returns:
        Mapping of ~400k lowercase words to float32 vectors of length 100.
    """
    vectors: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            word, *values = line.rstrip().split(" ")
            vectors[word] = np.asarray(values, dtype="float32")
    return vectors


def build_vectorizer(texts: np.ndarray | list[str]) -> layers.TextVectorization:
    """Create and adapt the word-level text vectoriser.

    Uses Keras defaults (lowercase + strip punctuation, whitespace split),
    matching what the paper's Keras tokenizer did.

    Args:
        texts: Training tweets; the vocabulary is learned from these only.

    Returns:
        An adapted ``TextVectorization`` layer.
    """
    vectorizer = layers.TextVectorization(
        max_tokens=config.MAX_VOCAB,
        output_sequence_length=config.MAX_SEQUENCE_LEN,
    )
    # Keras 3 only accepts strings as tf.string tensors, not numpy arrays.
    vectorizer.adapt(tf.constant(np.asarray(texts, dtype=str)))
    return vectorizer


def build_embedding_matrix(vocabulary: list[str], glove: dict[str, np.ndarray]) -> np.ndarray:
    """Build the embedding matrix aligned with the vectoriser vocabulary.

    Args:
        vocabulary: ``vectorizer.get_vocabulary()`` (index 0 is padding,
            index 1 the OOV token).
        glove: Word -> vector mapping from :func:`load_glove`.

    Returns:
        Matrix of shape ``(len(vocabulary), EMBEDDING_DIM)``; words missing
        from GloVe (and the padding/OOV rows) stay zero vectors.
    """
    matrix = np.zeros((len(vocabulary), config.EMBEDDING_DIM), dtype="float32")
    for index, word in enumerate(vocabulary):
        vector = glove.get(word)
        if vector is not None:
            matrix[index] = vector
    return matrix


def build_model(
    vectorizer: layers.TextVectorization,
    embedding_matrix: np.ndarray,
    lstm_units: int = config.LSTM_UNITS,
    dense_units: int = config.DENSE_UNITS,
) -> keras.Model:
    """Assemble and compile the paper's BiLSTM over raw-string inputs.

    Args:
        vectorizer: Adapted text vectoriser (becomes the model's first layer).
        embedding_matrix: GloVe weights aligned to the vectoriser vocabulary;
            frozen during training (standard practice for pretrained
            embeddings; the paper does not specify).
        lstm_units: LSTM units per direction (overridable for experiments).
        dense_units: Units of the penultimate Dense layer.

    Returns:
        A compiled Keras model mapping ``(batch, 1)`` strings to P(real).
    """
    inputs = keras.Input(shape=(1,), dtype="string")
    sequences = vectorizer(inputs)
    embedded = layers.Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=config.EMBEDDING_DIM,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )(sequences)
    hidden = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(embedded)
    pooled = layers.GlobalMaxPooling1D()(hidden)
    dense = layers.Dense(dense_units, activation="relu")(pooled)
    outputs = layers.Dense(1, activation="sigmoid")(dense)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_bilstm(protocol: str) -> keras.Model:
    """Load the trained BiLSTM for a protocol.

    Raises:
        FileNotFoundError: If the artifact does not exist yet (train it with
            ``python -m src.training.train_bilstm``).
    """
    path = artifact_path(protocol)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.training.train_bilstm "
            f"--protocol {protocol}` first."
        )
    return keras.models.load_model(path)
