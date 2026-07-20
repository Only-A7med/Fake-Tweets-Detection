"""Flask app serving the TweetGuard UI and prediction API.

Endpoints:
    GET  /         Single-page UI (textarea -> verdicts from both models).
    POST /predict  JSON ``{"text": "..."}`` -> both models' verdicts.
    GET  /health   Liveness probe listing which models are loaded.

Models are loaded once at startup from ``models/<protocol>/`` where protocol
comes from the ``TWEETGUARD_PROTOCOL`` env var (default ``safe`` — its test
metrics are leakage-free, so they are the honest numbers to quote for a live
demo). Listens on 0.0.0.0:7860, the port Hugging Face Spaces expects.

Usage:
    python -m src.app.app
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from src.models import predict

MAX_TWEET_CHARS = 280
MODEL_NAMES: tuple[predict.ModelName, ...] = ("baseline", "bilstm")


def create_app(protocol: str | None = None) -> Flask:
    """Build the Flask app and load whichever model artifacts exist.

    Args:
        protocol: Artifact set to serve ("paper" or "safe"). Defaults to the
            ``TWEETGUARD_PROTOCOL`` env var, then "safe".

    Returns:
        A configured Flask application; missing artifacts are reported by
        ``/health`` and yield HTTP 503 on ``/predict`` rather than a crash.
    """
    protocol = protocol or os.environ.get("TWEETGUARD_PROTOCOL", "safe")
    app = Flask(__name__)
    app.config["PROTOCOL"] = protocol

    models: dict[str, object] = {}
    for name in MODEL_NAMES:
        try:
            models[name] = predict.load_model(name, protocol)
        except FileNotFoundError as error:
            app.logger.warning("model unavailable: %s", error)
    app.config["MODELS"] = models

    @app.get("/")
    def index() -> str:
        return render_template("index.html", max_chars=MAX_TWEET_CHARS, protocol=protocol)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "protocol": protocol, "models": sorted(models)})

    @app.post("/predict")
    def predict_route():
        payload = request.get_json(silent=True)
        if payload is None or not isinstance(payload.get("text"), str):
            return jsonify({"error": 'send JSON like {"text": "..."}'}), 400
        text = payload["text"].strip()
        if not text:
            return jsonify({"error": "text is empty"}), 400
        if len(text) > MAX_TWEET_CHARS:
            return (
                jsonify({"error": f"text exceeds {MAX_TWEET_CHARS} characters ({len(text)})"}),
                400,
            )
        if not models:
            return jsonify({"error": "no trained models available"}), 503

        results = []
        for name, model in models.items():
            probability = float(predict.predict_proba(model, name, [text])[0])
            label, confidence = predict.decide(probability)
            results.append(
                {
                    "model": name,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "p_real": round(probability, 4),
                }
            )
        return jsonify({"text": text, "protocol": protocol, "predictions": results})

    return app


if __name__ == "__main__":
    # Dev server; production (Docker/HF Spaces) runs `gunicorn "src.app.app:create_app()"`.
    create_app().run(host="0.0.0.0", port=7860)
