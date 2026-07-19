# Project Plan — TweetGuard: Fake Tweet Detection Web App

**Companion code and live demo for the published paper:**
A. Tariq, S. I. Khan, M. Imran, A. Mustafa, "Fake Tweets Detection using Bi-LSTM with GloVe Embedding," *ICCIKE 2025*, IEEE Xplore. DOI: 10.1109/ICCIKE67021.2025.11318261

**Author of this repo:** Ahmad Mustafa (co-author of the paper) — built during BSc Computer Science (AI concentration).

---

## 1. Goal & CV Positioning

Turn the published research into a deployed, production-quality product:

> *"Reimplemented and deployed the Bi-LSTM misinformation detection model from my IEEE paper as a live web app with full CI/CD, tests, and Docker — 98% accuracy on the TruthSeeker 2023 dataset."*

Deliverables:
1. Public GitHub repo with clean history, README, tests, CI badge.
2. Live demo on Hugging Face Spaces (free tier, Docker runtime).
3. Reproducible training pipeline (anyone can retrain from scratch).

## 2. Architecture

```
User → Flask UI (paste tweet) → /predict API
        → Preprocessing (same cleaning as paper)
        → Model A: TF-IDF + Logistic Regression (baseline, fast)
        → Model B: Bi-LSTM + GloVe 100d (primary)
        → JSON: {label: real/fake, confidence, model_used}
```

Both models served; UI shows both predictions side by side (nice demo touch: "baseline vs deep learning").

## 3. Repo Structure

```
tweetguard/
├── README.md                  # badges, demo GIF, paper link, results table
├── PLAN.md                    # this file
├── CLAUDE.md                  # Claude Code project context
├── requirements.txt           # pinned versions
├── Dockerfile                 # runs Flask app; used by HF Spaces too
├── .github/workflows/ci.yml   # lint + tests on push (Phase 2)
├── src/
│   ├── data/
│   │   ├── download.py        # fetch TruthSeeker 2023 from Kaggle (kagglehub)
│   │   ├── preprocess.py      # cleaning, missing→"", word-swap augmentation (p=0.2)
│   │   └── split.py           # 80/20 train/test split
│   ├── models/
│   │   ├── baseline.py        # TF-IDF (10k features, uni+bigrams) + LogReg (max_iter=1000)
│   │   ├── bilstm.py          # Embedding(GloVe 100d) → BiLSTM → GlobalMaxPool → Dense(ReLU) → Dense(1, sigmoid)
│   │   └── predict.py         # unified predict interface for both models
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_bilstm.py    # 10 epochs, saves .keras + tokenizer
│   │   └── evaluate.py        # accuracy, precision, recall, F1, confusion matrix plots
│   └── app/
│       ├── app.py             # Flask: / (UI) and /predict (POST JSON)
│       ├── templates/index.html
│       └── static/style.css
├── models/                    # trained artifacts (gitignored if >50MB; else committed)
├── notebooks/
│   └── exploration.ipynb      # EDA on TruthSeeker (class balance, tweet lengths)
└── tests/
    ├── test_preprocess.py
    ├── test_predict.py
    └── test_api.py
```

## 4. Model Spec (from the paper — replicate exactly)

| Item | Value |
|---|---|
| Dataset | TruthSeeker 2023 (Kaggle). Key columns: `tweet`, `BinaryNumTarget` (1=real, 0=fake) |
| Augmentation | Word-swap, 20% probability, expand to 200,000 rows |
| Split | 80/20 train/test |
| Baseline | TF-IDF: 10,000 features, unigrams+bigrams; LogisticRegression max_iter=1000 |
| Deep model | GloVe 100d embeddings → BiLSTM → MaxPooling → Dense(ReLU) → Dense(1, sigmoid) |
| Training | 10 epochs, binary cross-entropy, Adam |
| Paper results | LogReg 98.1% acc; BiLSTM 98.2% acc (test n=40,000) |
| Metrics | Accuracy, precision, recall, F1, confusion matrix, training curves |

Notes:
- GloVe file (`glove.6B.100d.txt`, ~350MB) must NOT be committed — `src/data/download.py` fetches it; cache in `~/.cache/tweetguard/`.
- Dataset must NOT be committed — download script + Kaggle API instructions in README.
- Set random seeds everywhere for reproducibility.
- If retrained results differ slightly from paper (they will), report your actual numbers in README honestly, alongside the paper's.

## 5. Phase 1 — Working App (Milestones)

**M1 — Data pipeline** (acceptance: `python -m src.data.download && python -m src.data.preprocess` produces clean train/test CSVs)
1. Download script (kagglehub), preprocessing, augmentation, split.
2. EDA notebook with 3–4 plots.

**M2 — Models** (acceptance: `evaluate.py` prints metrics table ≥95% accuracy both models, saves confusion matrix PNG)
1. Train baseline, train BiLSTM, save artifacts + tokenizer.
2. Evaluation script generating README-ready results table and figures.

**M3 — Flask app** (acceptance: `python -m src.app.app` serves UI locally; POST /predict returns JSON <1s)
1. Simple, clean single-page UI: textarea → both model verdicts + confidence bars.
2. Input validation, error handling, 280-char limit.

**M4 — Deploy** (acceptance: public HF Spaces URL works from a phone)
1. Dockerfile (python:3.11-slim, port 7860 for HF Spaces).
2. Create HF Space (Docker runtime, free CPU tier), push, verify.
3. If model artifacts too large for git: upload to HF Hub, download at container start.

## 6. Phase 2 — MLOps Hardening

**M5 — Tests & quality** (acceptance: `pytest` green, `ruff check` clean)
- Unit tests: preprocessing edge cases (empty, emoji, URLs, Arabic text), predict contract, API tests via Flask test client. Target ~80% coverage on `src/`.
- `ruff` for lint+format, `pre-commit` hook.

**M6 — CI/CD** (acceptance: green badge in README)
- GitHub Actions: on push/PR → ruff + pytest on Python 3.11.
- Optional: auto-sync main branch → HF Spaces via Actions.

**M7 — Experiment tracking** (acceptance: MLflow screenshots in README)
- MLflow local tracking in training scripts (params, metrics, artifacts).
- Log 2–3 experiment variations (embedding dim, LSTM units) to show real usage.

**M8 — Polish**
- README: demo GIF, architecture diagram (mermaid), results table, "Paper vs. this repo" section, badges (CI, license, HF Spaces).
- MIT license. Clean commit history with conventional commits.

## 7. Tech Stack (pin these)

Python 3.11 · TensorFlow/Keras 2.16+ · scikit-learn · pandas · Flask 3 · gunicorn · kagglehub · pytest · ruff · MLflow · Docker

## 8. Risks & Fallbacks

| Risk | Fallback |
|---|---|
| TruthSeeker dataset unavailable on Kaggle | Use UNB CIC original source, or Constraint@AAAI COVID fake-news dataset (document the swap) |
| BiLSTM too slow on HF free CPU | Serve baseline by default, BiLSTM on demand; or quantize/reduce LSTM units |
| Model artifacts >100MB (GitHub limit) | Store on HF Hub, download at startup |
| Training too slow locally | Train on free Google Colab GPU, commit artifacts |

## 9. Free-Tier Requirements (verified July 2026 — $0 total)

Free accounts needed before starting (all free, no card required):
- **GitHub** — repo hosting, Actions CI (free for public repos).
- **Kaggle** — free account + API token (`kaggle.json`) to download TruthSeeker 2023. Dataset confirmed available on Kaggle and at UNB CIC.
- **Hugging Face** — free account + write token for Spaces deploy. CPU Basic tier (2 vCPU, 16GB) is free for Docker Spaces; free allowance covers demo usage.

Known free-tier behaviors (not problems, just expectations):
- HF Space sleeps after ~48h inactivity → first visitor waits ~30-60s for cold start. Fine for a CV demo.
- GloVe embeddings download free from Stanford NLP servers.
- If local training is too slow: Google Colab free GPU tier.

## 10. Suggested Claude Code Session Plan

One milestone per session; commit at each acceptance criterion.

1. "Read PLAN.md and CLAUDE.md. Scaffold the repo structure with empty modules, requirements.txt, .gitignore. Then implement M1."
2. "Implement M2 per PLAN.md section 4. Verify metrics against the paper's table."
3. "Implement M3. Test the API with curl before finishing."
4. "Implement M4. Walk me through creating the HF Space, then verify the live URL."
5. "Implement M5 and M6."
6. "Implement M7 and M8. Review the whole repo as a hiring manager would and fix anything embarrassing."
