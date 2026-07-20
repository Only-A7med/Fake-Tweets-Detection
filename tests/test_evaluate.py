"""Tests for the evaluation script's table rendering and figure output."""

import json

from src import config
from src.training import evaluate

SAMPLE_METRICS = {
    "accuracy": 0.9812,
    "precision": 0.98,
    "recall": 0.9825,
    "f1": 0.9812,
    "n_test": 40_000,
    "confusion_matrix": [[19_600, 400], [350, 19_650]],
}


def test_markdown_table_renders_one_row_per_model():
    table = evaluate.markdown_table({"paper": {"baseline": SAMPLE_METRICS}})
    lines = table.splitlines()
    assert lines[0].startswith("| Protocol | Model ")
    assert "| paper | baseline | 0.9812 | 0.9800 | 0.9825 | 0.9812 | 40,000 |" in lines


def test_plot_confusion_writes_png(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    path = evaluate.plot_confusion(SAMPLE_METRICS["confusion_matrix"], "baseline", "paper")
    assert path.exists() and path.suffix == ".png" and path.stat().st_size > 0


def test_plot_training_curves_reads_history_json(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    (models_dir / "paper").mkdir(parents=True)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    history = {"loss": [0.4, 0.2, 0.1], "accuracy": [0.8, 0.9, 0.95]}
    (models_dir / "paper" / "bilstm_history.json").write_text(json.dumps(history))
    path = evaluate.plot_training_curves("paper")
    assert path is not None and path.exists()


def test_plot_training_curves_returns_none_without_history(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert evaluate.plot_training_curves("paper") is None
