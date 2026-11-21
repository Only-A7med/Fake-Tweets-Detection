"""Regenerate ``tests/data/english_golden.json`` from the pinned weights.

Run this only when the pinned revision in ``tests/golden_support.py`` changes —
regenerating it against different weights is exactly the drift the guard exists
to catch, so a regeneration must be a deliberate, reviewed act.

Usage:
    python -m tests.generate_english_golden
"""

from __future__ import annotations

import json
import sys

from src import config
from src.models import predict
from tests import golden_support as support


def build() -> dict:
    """Predict every fixed input with every model and protocol.

    Uses exactly the calls ``/predict`` makes — ``predict.load_model``,
    ``predict.predict_proba``, ``predict.decide`` — so the recorded values
    describe the served behaviour, not a parallel implementation.
    """
    config.MODELS_DIR = support.fetch_models_dir()

    records: dict[str, dict] = {}
    for protocol in support.PROTOCOLS:
        for model_name in support.MODEL_NAMES:
            model = predict.load_model(model_name, protocol)
            for key, text in support.INPUTS.items():
                probability = float(predict.predict_proba(model, model_name, [text])[0])
                label, confidence = predict.decide(probability)
                records.setdefault(key, {})[f"{protocol}/{model_name}"] = {
                    "label": label,
                    "confidence": confidence,
                    "p_real": probability,
                }

    return {
        "model_repo": support.MODEL_REPO,
        "revision": support.MODEL_REVISION,
        "note": (
            "Golden predictions for the English pipeline. Labels must match exactly; "
            "probabilities to 1e-6. Regenerate only when the pinned revision changes."
        ),
        "inputs": support.INPUTS,
        "predictions": records,
    }


def main() -> int:
    """Write the golden file, creating ``tests/data/`` if needed."""
    payload = build()
    support.DATA_DIR.mkdir(parents=True, exist_ok=True)
    support.GOLDEN_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {support.GOLDEN_JSON} ({len(payload['predictions'])} inputs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
