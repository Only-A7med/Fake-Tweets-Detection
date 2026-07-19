# TweetGuard — Fake Tweet Detection

Companion code and live demo for the IEEE paper:

> A. Tariq, S. I. Khan, M. Imran, A. Mustafa, "Fake Tweets Detection using Bi-LSTM with GloVe
> Embedding," *Proc. ICCIKE 2025*, IEEE. DOI:
> [10.1109/ICCIKE67021.2025.11318261](https://doi.org/10.1109/ICCIKE67021.2025.11318261)

Two models trained on the [TruthSeeker 2023](https://www.unb.ca/cic/datasets/truthseeker-2023.html)
dataset classify tweets as **real** or **fake**: a TF-IDF + Logistic Regression baseline and the
paper's primary BiLSTM + GloVe model, served side by side from a small Flask app.

**Status: work in progress.** The data pipeline (M1) is done; models, web app, and deployment are
next — see [PLAN.md](PLAN.md) for the milestone plan.

## Quickstart — data pipeline (M1)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

python -m src.data.download       # fetch TruthSeeker 2023 from Kaggle (public dataset)
python -m src.data.preprocess     # clean, augment to 200k rows, 80/20 split
```

Outputs are written to `data/processed/` (gitignored): `augmented.csv`, `train.csv`, `test.csv`.
Add `--glove` to the download command to also fetch the GloVe 100-d embeddings (needed for
training the BiLSTM in M2).

## Results

Training and evaluation land in M2. The paper reports **98.1%** (LogReg baseline) and **98.2%**
(BiLSTM) accuracy on a 40,000-tweet test split; this repo will report its own retrained numbers
alongside the paper's — never as a substitute for them.

## Repository layout

```
src/
├── data/        # download, preprocessing + augmentation, train/test split
├── models/      # model definitions (M2)
├── training/    # training + evaluation entry points (M2)
└── app/         # Flask app (M3)
notebooks/       # EDA on TruthSeeker 2023
tests/           # pytest suite
```
