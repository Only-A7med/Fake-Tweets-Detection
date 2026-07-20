"""Tests for the M2 model code: baseline pipeline, BiLSTM building blocks, predict API."""

import numpy as np
import pytest

from src import config
from src.models import baseline, bilstm, predict

# --- baseline ---


def test_build_baseline_uses_paper_hyperparameters():
    pipeline = baseline.build_baseline()
    tfidf = pipeline.named_steps["tfidf"]
    logreg = pipeline.named_steps["logreg"]
    assert tfidf.max_features == config.TFIDF_MAX_FEATURES == 10_000
    assert tfidf.ngram_range == config.TFIDF_NGRAM_RANGE == (1, 2)
    assert logreg.max_iter == config.LOGREG_MAX_ITER == 1000


def test_baseline_round_trip_learns_a_separable_toy_problem():
    texts = ["genuine verified report"] * 20 + ["fabricated hoax rumour"] * 20
    labels = [1] * 20 + [0] * 20
    pipeline = baseline.build_baseline()
    pipeline.fit(texts, labels)
    assert pipeline.predict(["genuine verified report"])[0] == 1
    assert pipeline.predict(["fabricated hoax rumour"])[0] == 0


def test_load_baseline_missing_artifact_raises_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_baseline"):
        baseline.load_baseline("paper")


# --- bilstm building blocks ---


@pytest.fixture(scope="module")
def tiny_glove():
    return {
        "real": np.full(config.EMBEDDING_DIM, 0.5, dtype="float32"),
        "fake": np.full(config.EMBEDDING_DIM, -0.5, dtype="float32"),
    }


def test_build_vectorizer_lowercases_and_learns_vocabulary():
    vectorizer = bilstm.build_vectorizer(["Real News", "fake news"])
    vocabulary = vectorizer.get_vocabulary()
    assert vocabulary[0] == ""  # padding
    assert vocabulary[1] == "[UNK]"
    assert {"real", "fake", "news"} <= set(vocabulary)


def test_build_embedding_matrix_aligns_known_words(tiny_glove):
    vocabulary = ["", "[UNK]", "real", "unknownword"]
    matrix = bilstm.build_embedding_matrix(vocabulary, tiny_glove)
    assert matrix.shape == (4, config.EMBEDDING_DIM)
    np.testing.assert_array_equal(matrix[2], tiny_glove["real"])
    assert not matrix[0].any() and not matrix[1].any() and not matrix[3].any()


def test_build_model_outputs_probabilities_from_raw_strings(tiny_glove):
    vectorizer = bilstm.build_vectorizer(["real news", "fake rumour"])
    matrix = bilstm.build_embedding_matrix(vectorizer.get_vocabulary(), tiny_glove)
    model = bilstm.build_model(vectorizer, matrix)
    probabilities = predict.predict_proba(model, "bilstm", ["real news", "fake rumour"])
    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_load_bilstm_missing_artifact_raises_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_bilstm"):
        bilstm.load_bilstm("safe")


# --- predict interface ---


def test_decide_maps_probability_to_label_and_confidence():
    assert predict.decide(0.9) == ("real", 0.9)
    label, confidence = predict.decide(0.1)
    assert label == "fake"
    assert confidence == pytest.approx(0.9)


def test_decide_boundary_favours_real():
    assert predict.decide(0.5)[0] == "real"


def test_load_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        predict.load_model("transformer", "paper")


def test_predict_proba_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        predict.predict_proba(object(), "transformer", ["x"])


def test_predict_proba_baseline_returns_p_real():
    texts = ["good true story"] * 10 + ["bad false story"] * 10
    pipeline = baseline.build_baseline()
    pipeline.fit(texts, [1] * 10 + [0] * 10)
    probabilities = predict.predict_proba(pipeline, "baseline", ["good true story"])
    assert probabilities[0] > 0.5
