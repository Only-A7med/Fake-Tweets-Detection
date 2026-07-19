# CLAUDE.md — TweetGuard

## What this is
Deployment + reimplementation of my published IEEE paper: "Fake Tweets Detection using Bi-LSTM with GloVe Embedding" (ICCIKE 2025, DOI: 10.1109/ICCIKE67021.2025.11318261). I'm a co-author. Full plan in PLAN.md — always follow it; model hyperparameters in PLAN.md §4 are from the paper and must be replicated exactly.

## Purpose
Portfolio project for my CV. Code quality matters as much as functionality — a hiring manager will read this repo. Prefer clear, simple, well-documented code over clever code.

## Conventions
- Python 3.11, PEP8, type hints on public functions, docstrings (Google style).
- Run modules as packages: `python -m src.training.train_bilstm`.
- Lint/format: `ruff check . && ruff format .` before every commit.
- Tests: `pytest -q`. Write tests alongside new code, not after.
- Commits: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`).
- Seeds fixed at 42 (numpy, tensorflow, random, sklearn) for reproducibility.

## Hard rules
- NEVER commit: dataset files, `glove.6B.100d.txt`, anything >50MB, secrets, `.env`.
- `.gitignore` these from the first commit.
- Report real evaluation numbers — never copy paper numbers as if they were this repo's output. README distinguishes "paper results" vs "this repo's results".
- Model artifacts: if >100MB, host on HF Hub and download at startup (see PLAN.md §8).

## Key facts (avoid re-deriving)
- Dataset: TruthSeeker 2023 (Kaggle). Text column `tweet`, label `BinaryNumTarget` (1=real, 0=fake).
- Augmentation: word-swap p=0.2, expand to 200k rows (paper method).
- Baseline: TF-IDF (10k feats, 1-2 grams) + LogReg(max_iter=1000) → paper: 98.1%.
- BiLSTM: GloVe 100d → BiLSTM → GlobalMaxPool → Dense ReLU → sigmoid, 10 epochs → paper: 98.2%.
- Flask app must listen on port 7860, host 0.0.0.0 (HF Spaces Docker requirement).

## Environment notes
- Training the BiLSTM locally may be slow; offer Colab option (notebook in `notebooks/`) rather than blocking.
- pip installs: plain `pip install -r requirements.txt` inside venv/Docker.
