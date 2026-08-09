---
title: TweetGuard
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TweetGuard — Fake Tweet Detection

[![CI](https://github.com/Only-A7med/Fake-Tweets-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Only-A7med/Fake-Tweets-Detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-HF%20Spaces-blue)](https://huggingface.co/spaces/Only-A7med/tweetguard)

Companion code and live demo for the IEEE paper:

> A. Tariq, S. I. Khan, M. Imran, A. Mustafa, "Fake Tweets Detection using Bi-LSTM with GloVe
> Embedding," *Proc. ICCIKE 2025*, IEEE. DOI:
> [10.1109/ICCIKE67021.2025.11318261](https://doi.org/10.1109/ICCIKE67021.2025.11318261)

Two models trained on the [TruthSeeker 2023](https://www.unb.ca/cic/datasets/truthseeker-2023.html)
dataset classify tweets as **real** or **fake**: a TF-IDF + Logistic Regression baseline and the
paper's primary BiLSTM + GloVe model, served side by side from a small Flask app.

## Architecture

```mermaid
flowchart LR
    U[User] -->|paste tweet| UI[Flask UI]
    UI -->|POST /predict| API[/predict endpoint/]
    API --> P[Preprocessing]
    P --> A["Model A\nTF-IDF + LogReg"]
    P --> B["Model B\nBiLSTM + GloVe 100d"]
    A --> J["JSON: label, confidence"]
    B --> J
    J --> UI
```

## Results

**Paper vs. this repo.** The paper reports **98.1%** (LogReg) and **98.2%** (BiLSTM) accuracy on a
40,000-tweet test split. The numbers below are this repo's retrained results — reported honestly,
never substituted with the paper's.

This repo evaluates under **two protocols**:

- **paper** — augment to 200k rows first, then split 80/20, exactly as the paper describes.
  Word-swapped variants of one source tweet can land on both sides of the split, so some training
  information leaks into the test set.
- **leakage-safe** — split original tweets 80/20 first, then augment only the training side.
  Every test row is an unmodified tweet never seen (even as a variant) in training. These are the
  honest generalisation numbers, and what the live demo serves.

| Protocol | Model | Accuracy | Precision | Recall | F1 | Test n |
|---|---|---|---|---|---|---|
| paper | TF-IDF + LogReg | 97.2% | 0.9714 | 0.9746 | 0.9730 | 40,000 |
| paper | BiLSTM + GloVe | 96.8% | 0.9751 | 0.9632 | 0.9691 | 40,000 |
| leakage-safe | TF-IDF + LogReg | 97.7% | 0.9762 | 0.9799 | 0.9780 | 26,804 |
| leakage-safe | BiLSTM + GloVe | 95.8% | 0.9287 | 0.9951 | 0.9608 | 26,804 |

Confusion matrices and training curves are generated into `reports/figures/` by
`python -m src.training.evaluate`.

<p>
  <img src="reports/figures/confusion_bilstm_safe.png" alt="BiLSTM confusion matrix (leakage-safe)" width="360">
  <img src="reports/figures/training_curves_safe.png" alt="BiLSTM training curves (leakage-safe)" width="420">
</p>

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Data (TruthSeeker 2023 via Kaggle — public, no API key needed)
python -m src.data.download --glove
python -m src.data.preprocess                  # paper protocol -> train.csv/test.csv
python -m src.data.preprocess --leakage-safe   # safe protocol  -> train_safe.csv/test_safe.csv

# 2. Train (baseline: ~1 min; BiLSTM: ~15 min/protocol on CPU)
python -m src.training.train_baseline
python -m src.training.train_bilstm

# 3. Evaluate -> metrics table + reports/figures/
python -m src.training.evaluate

# 4. Serve the app -> http://localhost:7860
python -m src.app.app
```

### API

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Federal Reserve raised interest rates on Wednesday."}'
```

```json
{
  "predictions": [
    {"model": "baseline", "label": "real", "confidence": 0.9859, "p_real": 0.9859},
    {"model": "bilstm",   "label": "real", "confidence": 1.0,    "p_real": 1.0}
  ],
  "protocol": "safe",
  "text": "The Federal Reserve raised interest rates on Wednesday."
}
```

## Experiment tracking

Training runs are tracked with MLflow (local `sqlite:///mlflow.db` store): hyperparameters,
per-epoch curves, test accuracy, and artifacts. Inspect them with `mlflow ui`. LSTM width was
varied to sanity-check the paper's architecture choices (see `--lstm-units` /
`--dense-units` / `--no-save` on `train_bilstm`).

## Deployment (Hugging Face Spaces, Docker runtime)

The image never contains the model weights; they are pulled at container start from a HF model
repo:

```bash
# one-time: upload trained artifacts (needs `hf auth login` with a write token)
python scripts/upload_artifacts.py --repo Only-A7med/tweetguard-models
```

Create a Docker-runtime Space, push this repo to it, and set the Space variable
`TWEETGUARD_MODEL_REPO=Only-A7med/tweetguard-models`. The YAML front-matter at the top of this
README configures the Space (port 7860). Local container:

```bash
docker build -t tweetguard .
docker run -p 7860:7860 -e TWEETGUARD_MODEL_REPO=Only-A7med/tweetguard-models tweetguard
```

## Repository layout

```
src/
├── data/        # download (kagglehub + GloVe), preprocessing + augmentation, split
├── models/      # baseline pipeline, BiLSTM architecture, unified predict API
├── training/    # train_baseline, train_bilstm, evaluate (all MLflow-tracked)
└── app/         # Flask app: UI, /predict, /health, startup artifact download
scripts/         # upload_artifacts.py (one-time HF Hub upload)
notebooks/       # EDA on TruthSeeker 2023
tests/           # 75 tests, 87% coverage on src/
```

## Reproducibility notes

- Seeds fixed at 42 (numpy, tensorflow, random, sklearn) everywhere.
- Dataset and GloVe embeddings are never committed; download scripts cache them outside the repo.
- Hyperparameters not specified in the paper (LSTM units, vocab size, sequence length, batch
  size) are marked as such in `src/config.py`.
- The paper's augment-then-split protocol is reproduced as published; the leakage-safe protocol
  exists because the honest generalisation number matters for a deployed demo.

## License

[MIT](LICENSE) — the paper itself is © IEEE.
