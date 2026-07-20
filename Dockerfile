# TweetGuard — Flask app image, also used as-is by Hugging Face Spaces
# (Docker runtime). HF Spaces requires the app to listen on port 7860.
FROM python:3.11-slim

# HF Spaces runs containers as a non-root user; give it a writable home for
# model downloads and the HF cache.
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

USER appuser
ENV PYTHONUNBUFFERED=1 \
    TWEETGUARD_PROTOCOL=safe \
    HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 7860

# Fetch trained artifacts from the HF Hub (TWEETGUARD_MODEL_REPO), then serve.
# Single worker: the free CPU tier has 2 vCPUs and the models are loaded per
# worker; threads handle concurrent demo traffic fine.
CMD ["sh", "-c", "python -m src.app.download_artifacts && gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120 'src.app.app:create_app()'"]
