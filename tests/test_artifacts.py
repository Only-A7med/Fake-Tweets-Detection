"""Tests for artifact save/load round trips, GloVe parsing, and startup download."""

import numpy as np
import pytest

from src import config
from src.app import download_artifacts
from src.models import baseline, bilstm, predict

# --- baseline save/load round trip ---


def test_baseline_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    pipeline = baseline.build_baseline()
    pipeline.fit(["real thing", "fake thing"], [1, 0])
    saved_path = baseline.save_baseline(pipeline, "paper")
    loaded = baseline.load_baseline("paper")
    assert loaded.predict(["real thing"])[0] == pipeline.predict(["real thing"])[0]
    assert saved_path == baseline.artifact_path("paper")


# --- glove parsing and bilstm save/load ---


def test_load_glove_parses_word_vectors(tmp_path):
    glove_file = tmp_path / "glove.txt"
    glove_file.write_text("hello 0.1 0.2 0.3\nworld -0.1 0.0 0.5\n", encoding="utf-8")
    vectors = bilstm.load_glove(glove_file)
    assert set(vectors) == {"hello", "world"}
    np.testing.assert_allclose(vectors["hello"], [0.1, 0.2, 0.3])


def test_bilstm_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    vectorizer = bilstm.build_vectorizer(["real news", "fake rumour"])
    matrix = np.zeros((len(vectorizer.get_vocabulary()), config.EMBEDDING_DIM), dtype="float32")
    model = bilstm.build_model(vectorizer, matrix, lstm_units=4, dense_units=4)
    path = bilstm.artifact_path("safe")
    path.parent.mkdir(parents=True)
    model.save(path)

    loaded = bilstm.load_bilstm("safe")
    texts = ["real news"]
    np.testing.assert_allclose(
        predict.predict_proba(loaded, "bilstm", texts),
        predict.predict_proba(model, "bilstm", texts),
        atol=1e-6,
    )


# --- predict_one over stored artifacts ---


def test_predict_one_returns_contract_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    pipeline = baseline.build_baseline()
    pipeline.fit(["real thing"] * 5 + ["fake thing"] * 5, [1] * 5 + [0] * 5)
    baseline.save_baseline(pipeline, "safe")

    result = predict.predict_one("real thing", "baseline", "safe")
    assert result["model"] == "baseline"
    assert result["protocol"] == "safe"
    assert result["label"] in {"real", "fake"}
    assert 0.5 <= result["confidence"] <= 1


# --- startup artifact download ---


@pytest.fixture
def empty_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.delenv("TWEETGUARD_MODEL_REPO", raising=False)
    return tmp_path / "models"


def test_artifacts_present_false_when_missing(empty_models_dir):
    assert not download_artifacts.artifacts_present("safe")


def test_download_skips_when_artifacts_exist(empty_models_dir, monkeypatch, capsys):
    for artifact in (baseline.artifact_path("safe"), bilstm.artifact_path("safe")):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"stub")
    monkeypatch.setattr(
        download_artifacts,
        "snapshot_download",
        lambda **kwargs: pytest.fail("should not download"),
    )
    download_artifacts.download_artifacts("safe")
    assert "skipping download" in capsys.readouterr().out


def test_download_without_repo_env_var_exits_with_hint(empty_models_dir):
    with pytest.raises(SystemExit, match="TWEETGUARD_MODEL_REPO"):
        download_artifacts.download_artifacts("safe")


def test_download_calls_snapshot_with_protocol_pattern(empty_models_dir, monkeypatch):
    calls = {}
    monkeypatch.setenv("TWEETGUARD_MODEL_REPO", "someone/tweetguard-models")
    monkeypatch.setattr(
        download_artifacts, "snapshot_download", lambda **kwargs: calls.update(kwargs)
    )
    download_artifacts.download_artifacts("safe")
    assert calls["repo_id"] == "someone/tweetguard-models"
    assert calls["allow_patterns"] == ["safe/*"]
