"""Flask API tests using stub models (no trained artifacts required)."""

import numpy as np
import pytest

from src.app import app as app_module
from src.models import predict


class StubModel:
    """Baseline-shaped stub: predict_proba over [P(fake), P(real)] columns."""

    classes_ = np.array([0, 1])

    def __init__(self, p_real: float):
        self.p_real = p_real

    def predict_proba(self, texts):
        return np.array([[1 - self.p_real, self.p_real]] * len(list(texts)))


@pytest.fixture
def client(monkeypatch):
    def fake_load(name, protocol):
        return StubModel(0.9 if name == "baseline" else 0.2)

    monkeypatch.setattr(predict, "load_model", fake_load)
    # Route every stub through the baseline predict_proba path.
    monkeypatch.setattr(
        predict,
        "predict_proba",
        lambda model, name, texts: model.predict_proba(texts)[:, 1],
    )
    flask_app = app_module.create_app(protocol="safe")
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def modelless_client(monkeypatch):
    def raise_missing(name, protocol):
        raise FileNotFoundError("no artifact")

    monkeypatch.setattr(predict, "load_model", raise_missing)
    flask_app = app_module.create_app(protocol="safe")
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_index_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"TweetGuard" in response.data


def test_health_lists_models_and_protocol(client):
    payload = client.get("/health").get_json()
    assert payload == {"status": "ok", "protocol": "safe", "models": ["baseline", "bilstm"]}


def test_predict_returns_both_model_verdicts(client):
    response = client.post("/predict", json={"text": "Some tweet"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["protocol"] == "safe"
    by_model = {p["model"]: p for p in payload["predictions"]}
    assert by_model["baseline"]["label"] == "real"
    assert by_model["baseline"]["confidence"] == pytest.approx(0.9)
    assert by_model["bilstm"]["label"] == "fake"
    assert by_model["bilstm"]["confidence"] == pytest.approx(0.8)


def test_predict_strips_and_echoes_text(client):
    payload = client.post("/predict", json={"text": "  padded  "}).get_json()
    assert payload["text"] == "padded"


@pytest.mark.parametrize(
    "body",
    [None, {}, {"text": 42}, {"wrong_key": "x"}],
)
def test_predict_rejects_malformed_payloads(client, body):
    response = client.post("/predict", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_rejects_non_json_body(client):
    response = client.post("/predict", data="just text", content_type="text/plain")
    assert response.status_code == 400


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 400


def test_predict_rejects_overlong_text(client):
    response = client.post("/predict", json={"text": "x" * (app_module.MAX_TWEET_CHARS + 1)})
    assert response.status_code == 400
    assert "280" in response.get_json()["error"]


def test_predict_accepts_exactly_280_chars(client):
    response = client.post("/predict", json={"text": "x" * app_module.MAX_TWEET_CHARS})
    assert response.status_code == 200


def test_predict_without_models_returns_503(modelless_client):
    response = modelless_client.post("/predict", json={"text": "hello"})
    assert response.status_code == 503


def test_health_without_models_still_ok(modelless_client):
    payload = modelless_client.get("/health").get_json()
    assert payload["status"] == "ok"
    assert payload["models"] == []
