"""Upload trained model artifacts to a Hugging Face model repo (one-time deploy step).

Run after training both models locally. Requires a write token: either run
``hf auth login`` first or set the ``HF_TOKEN`` env var.

Usage:
    python scripts/upload_artifacts.py --repo <user>/fake-tweets-detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXPECTED = [
    "paper/baseline.joblib",
    "paper/bilstm.keras",
    "safe/baseline.joblib",
    "safe/bilstm.keras",
]


def main(argv: list[str] | None = None) -> int:
    """Create the model repo if needed and upload the local ``models/`` tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", required=True, help="target repo id, e.g. user/fake-tweets-detection"
    )
    args = parser.parse_args(argv)

    missing = [name for name in EXPECTED if not (MODELS_DIR / name).exists()]
    if missing:
        print(f"Missing artifacts (train first): {', '.join(missing)}")
        return 1

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=MODELS_DIR,
        commit_message="Upload TweetGuard trained artifacts",
    )
    print(f"Uploaded {MODELS_DIR} -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
