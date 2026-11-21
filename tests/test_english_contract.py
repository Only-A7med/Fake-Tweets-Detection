"""Fast half of the English guard: preprocessing output and the ``/predict`` contract.

Neither test needs trained weights, so both run everywhere and on every pull
request. They cover what the existing suite does not: a frozen input-to-output
table for the cleaning step, and the exact shape, types and ranges of the
``/predict`` response.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app import app as app_module
from src.data import preprocess
from src.models import predict
from tests import golden_support as support

# Frozen cleaning results for the same inputs the golden predictions use.
# clean_text collapses whitespace and strips; it changes nothing else — no
# lowercasing, no URL or emoji handling. Freezing that keeps a later Arabic
# normaliser from quietly altering English behaviour.
EXPECTED_CLEAN = {
    "true_rates": "The Federal Reserve raised interest rates by 25 basis points on Wednesday.",
    "edge_empty": "",
    "edge_whitespace": "",
    "edge_url_only": "https://example.com/news/article/12345",
    "edge_mention_hashtag": "@newsdesk check this out #BreakingNews #Update",
    "edge_emoji_only": "🔥🔥🔥 😂 🙈",
    "edge_all_caps": "THIS IS COMPLETELY TRUE AND YOU MUST SHARE IT WITH EVERYONE RIGHT NOW",
    "edge_punctuation_only": "!!! ??? ... ,,, ---",
    "edge_single_word": "Breaking",
    "edge_digits": "42",
}


@pytest.mark.parametrize("input_key", sorted(EXPECTED_CLEAN))
def test_clean_text_output_is_frozen(input_key):
    assert preprocess.clean_text(support.INPUTS[input_key]) == EXPECTED_CLEAN[input_key]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("line one\nline two", "line one line two"),
        ("tabs\tand\tspaces", "tabs and spaces"),
        ("  leading and trailing  ", "leading and trailing"),
        ("multiple     spaces", "multiple spaces"),
        ("\r\nmixed\r\nnewlines\r\n", "mixed newlines"),
    ],
)
def test_clean_text_whitespace_handling_is_frozen(raw, expected):
    assert preprocess.clean_text(raw) == expected


class StubModel:
    """Baseline-shaped stub returning a fixed P(real) for every input."""

    classes_ = np.array([0, 1])

    def __init__(self, p_real: float):
        self.p_real = p_real

    def predict_proba(self, texts):
        return np.array([[1 - self.p_real, self.p_real]] * len(list(texts)))


@pytest.fixture
def client(monkeypatch):
    """A test client whose two models are stubs, so no weights are needed."""
    monkeypatch.setattr(
        predict,
        "load_model",
        lambda name, protocol: StubModel(0.8125 if name == "baseline" else 0.25),
    )
    monkeypatch.setattr(
        predict,
        "predict_proba",
        lambda model, name, texts: model.predict_proba(texts)[:, 1],
    )
    flask_app = app_module.create_app(protocol="safe")
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_predict_response_has_exactly_the_documented_shape(client):
    payload = client.post("/predict", json={"text": "  A tweet to classify  "}).get_json()

    assert sorted(payload) == ["predictions", "protocol", "text"]
    assert payload["text"] == "A tweet to classify"
    assert payload["protocol"] == "safe"
    assert isinstance(payload["predictions"], list)
    assert len(payload["predictions"]) == 2

    for prediction in payload["predictions"]:
        assert sorted(prediction) == ["confidence", "label", "model", "p_real"]
        assert prediction["model"] in ("baseline", "bilstm")
        assert prediction["label"] in ("real", "fake")
        assert isinstance(prediction["confidence"], float)
        assert isinstance(prediction["p_real"], float)
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert 0.0 <= prediction["p_real"] <= 1.0
        assert prediction["confidence"] >= 0.5


def test_predict_confidence_agrees_with_label_and_p_real(client):
    predictions = client.post("/predict", json={"text": "A tweet"}).get_json()["predictions"]
    by_model = {p["model"]: p for p in predictions}

    assert by_model["baseline"]["label"] == "real"
    assert by_model["baseline"]["p_real"] == pytest.approx(0.8125)
    assert by_model["baseline"]["confidence"] == pytest.approx(0.8125)

    assert by_model["bilstm"]["label"] == "fake"
    assert by_model["bilstm"]["p_real"] == pytest.approx(0.25)
    assert by_model["bilstm"]["confidence"] == pytest.approx(0.75)


def test_predict_rounds_probabilities_to_four_decimals(client):
    prediction = client.post("/predict", json={"text": "A tweet"}).get_json()["predictions"][0]
    assert prediction["p_real"] == round(prediction["p_real"], 4)
    assert prediction["confidence"] == round(prediction["confidence"], 4)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({}, "no text key"),
        ({"tweet": "wrong key"}, "wrong key"),
        ({"text": None}, "null text"),
        ({"text": 7}, "non-string text"),
        ({"text": ""}, "empty text"),
        ({"text": "    "}, "whitespace-only text"),
    ],
)
def test_predict_error_response_shape(client, body, reason):
    response = client.post("/predict", json=body)
    assert response.status_code == 400, reason
    payload = response.get_json()
    assert sorted(payload) == ["error"], reason
    assert isinstance(payload["error"], str) and payload["error"]
