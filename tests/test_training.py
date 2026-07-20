"""Tests for the training entry points and evaluation flow on tiny fixtures.

Real network and GloVe downloads are stubbed out; MLflow tracking is pointed
at a per-test temporary store so runs do not pollute the project's mlruns/.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import download
from src.models import baseline, bilstm
from src.training import evaluate, train_baseline, train_bilstm

TINY_ROWS = 24


@pytest.fixture
def tiny_env(tmp_path, monkeypatch):
    """Tiny train/test CSVs + isolated models/reports/mlflow dirs for one protocol."""
    processed = tmp_path / "processed"
    processed.mkdir()
    train_csv, test_csv = processed / "train.csv", processed / "test.csv"
    rng = np.random.default_rng(config.SEED)
    for path, rows in ((train_csv, TINY_ROWS), (test_csv, 8)):
        labels = rng.integers(0, 2, size=rows)
        tweets = [
            f"{'verified genuine report' if y else 'fabricated hoax rumour'} number {i}"
            for i, y in enumerate(labels)
        ]
        pd.DataFrame({"tweet": tweets, "label": labels}).to_csv(path, index=False)

    monkeypatch.setitem(config.PROTOCOL_SPLITS, "paper", (train_csv, test_csv))
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    return tmp_path


# --- train_baseline ---


def test_load_split_missing_csv_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        train_baseline.load_split(tmp_path / "nope.csv")


def test_load_split_keeps_null_like_tweets_as_strings(tmp_path):
    csv = tmp_path / "train.csv"
    pd.DataFrame({"tweet": ["null", "None"], "label": [1, 0]}).to_csv(csv, index=False)
    frame = train_baseline.load_split(csv)
    assert frame["tweet"].tolist() == ["null", "None"]


def test_train_baseline_protocol_saves_artifact(tiny_env):
    path = train_baseline.train_protocol("paper")
    assert path.exists()
    assert baseline.load_baseline("paper") is not None


def test_train_baseline_main_runs_selected_protocol(tiny_env):
    assert train_baseline.main(["--protocol", "paper"]) == 0


# --- train_bilstm (tiny GloVe, 1 epoch) ---


@pytest.fixture
def stub_glove(monkeypatch):
    vectors = {
        word: np.full(config.EMBEDDING_DIM, value, dtype="float32")
        for word, value in [("verified", 0.3), ("genuine", 0.2), ("hoax", -0.3)]
    }
    monkeypatch.setattr(download, "download_glove", lambda: "unused-path")
    monkeypatch.setattr(bilstm, "load_glove", lambda path: vectors)
    return vectors


def test_train_bilstm_protocol_saves_artifact_and_history(tiny_env, stub_glove):
    train_bilstm.train_protocol("paper", epochs=1)
    assert bilstm.artifact_path("paper").exists()
    history = json.loads(bilstm.history_path("paper").read_text(encoding="utf-8"))
    assert len(history["loss"]) == 1


def test_train_bilstm_no_save_leaves_no_artifact(tiny_env, stub_glove):
    train_bilstm.train_protocol("paper", epochs=1, lstm_units=4, dense_units=4, save=False)
    assert not bilstm.artifact_path("paper").exists()


def test_train_bilstm_main_parses_variation_flags(tiny_env, stub_glove):
    argv = ["--protocol", "paper", "--epochs", "1", "--lstm-units", "4", "--no-save"]
    assert train_bilstm.main(argv) == 0


# --- evaluate ---


def test_evaluate_model_computes_metrics_on_test_split(tiny_env):
    train_baseline.train_protocol("paper")
    metrics = evaluate.evaluate_model("baseline", "paper")
    assert metrics["n_test"] == 8
    assert 0 <= metrics["accuracy"] <= 1
    assert np.asarray(metrics["confusion_matrix"]).sum() == 8


def test_evaluate_main_writes_metrics_json_and_figures(tiny_env):
    train_baseline.train_protocol("paper")
    assert evaluate.main(["--protocol", "paper"]) == 0
    saved = json.loads((config.REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert "baseline" in saved["paper"]
    assert (config.FIGURES_DIR / "confusion_baseline_paper.png").exists()


def test_evaluate_main_without_artifacts_returns_error(tiny_env):
    assert evaluate.main(["--protocol", "paper"]) == 1
