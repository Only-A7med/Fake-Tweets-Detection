"""Fetch trained model artifacts from the Hugging Face Hub at container start.

The Docker image ships without model weights (they are gitignored and too
large to bake into a portfolio repo). At startup this pulls the ``models/``
tree from the model repo named by ``TWEETGUARD_MODEL_REPO`` into the local
``models/`` directory. If the artifacts are already present (local dev, warm
container) it does nothing.

Usage:
    python -m src.app.download_artifacts
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

from src import config
from src.models import baseline, bilstm


def artifacts_present(protocol: str) -> bool:
    """True if both trained artifacts for a protocol exist locally."""
    return baseline.artifact_path(protocol).exists() and bilstm.artifact_path(protocol).exists()


def download_artifacts(protocol: str | None = None) -> None:
    """Download the ``models/`` tree from the Hub unless it is already local.

    Args:
        protocol: Protocol whose artifacts are required (defaults to the
            ``TWEETGUARD_PROTOCOL`` env var, then "safe").
    """
    protocol = protocol or os.environ.get("TWEETGUARD_PROTOCOL", "safe")
    if artifacts_present(protocol):
        print(f"Artifacts for protocol '{protocol}' already present; skipping download.")
        return

    repo_id = os.environ.get("TWEETGUARD_MODEL_REPO")
    if not repo_id:
        raise SystemExit(
            "TWEETGUARD_MODEL_REPO is not set and no local artifacts were found. "
            "Set it to the HF model repo holding the trained artifacts "
            "(see scripts/upload_artifacts.py), or train locally first."
        )
    print(f"Downloading artifacts from {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{protocol}/*"],
        local_dir=config.MODELS_DIR,
    )
    print(f"Artifacts ready under {config.MODELS_DIR}")


if __name__ == "__main__":
    sys.exit(download_artifacts())
