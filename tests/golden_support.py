"""Shared fixtures for the English regression guard.

The guard pins the English pipeline to one published set of weights so that any
later change to shared code shows up as a failing test rather than a silent
drift in behaviour. Three pieces live here:

* ``INPUTS`` — the fixed tweet-like strings the golden predictions cover. They
  are written for this test, not taken from the dataset, so nothing about the
  training data is committed.
* the weight loader — ``snapshot_download`` pinned to one revision of the
  public model repository, anonymous (``token=False``) and cached outside the
  repository.
* ``models_dir_or_skip`` — hands a test the directory the artifacts live in,
  skipping when they cannot be fetched, unless ``REQUIRE_GOLDEN=1`` turns that
  skip into a failure (continuous integration sets it).

``src/app/download_artifacts.py`` is deliberately untouched: the app downloads
whatever the deployment points at, while these tests must always read one
fixed revision.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_REPO = "Only-A7med/fake-tweets-detection"
MODEL_REVISION = "5624bbdaafe313bfc14c6d00d50a6ef274c03357"
PROTOCOLS = ("paper", "safe")
MODEL_NAMES = ("baseline", "bilstm")

# Probabilities are compared to this tolerance; labels must match exactly.
PROBABILITY_TOLERANCE = 1e-6

ARTIFACT_FILES = (
    "paper/baseline.joblib",
    "paper/bilstm.keras",
    "safe/baseline.joblib",
    "safe/bilstm.keras",
)

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN_JSON = DATA_DIR / "english_golden.json"
METRICS_EXPECTED_JSON = DATA_DIR / "english_metrics_expected.json"

# Twenty-two fixed inputs: true-style claims, false-style claims, then edge
# cases. Keys are stable identifiers so a reordering cannot silently remap
# expected values.
INPUTS: dict[str, str] = {
    "true_rates": "The Federal Reserve raised interest rates by 25 basis points on Wednesday.",
    "true_inflation": "Inflation fell to 3.1% in October, according to official statistics.",
    "true_guidance": (
        "The World Health Organization published new guidance on seasonal influenza vaccines."
    ),
    "true_council": "The local council approved a budget for road repairs starting next month.",
    "true_study": (
        "A study of 40,000 adults found that regular exercise lowers the risk of heart disease."
    ),
    "true_election": (
        "Officials said turnout in the regional election was the highest in twelve years."
    ),
    "fake_bleach": (
        "BREAKING: scientists discover that drinking bleach cures every known disease overnight."
    ),
    "fake_moon": "Government finally admits the moon landing was filmed in a basement in Nevada.",
    "fake_5g": "Doctors confirm 5G towers implant microchips through ordinary tap water.",
    "fake_lemon": "One lemon a day replaces every vaccine, an anonymous insider has revealed.",
    "fake_double": (
        "A celebrity was secretly replaced by a body double in 1997, the family confirms."
    ),
    "fake_banks": "All banks will seize every savings account at midnight, leaked memo shows.",
    "edge_empty": "",
    "edge_whitespace": "   ",
    "edge_url_only": "https://example.com/news/article/12345",
    "edge_mention_hashtag": "@newsdesk check this out #BreakingNews #Update",
    "edge_emoji_only": "🔥🔥🔥 😂 🙈",
    "edge_all_caps": "THIS IS COMPLETELY TRUE AND YOU MUST SHARE IT WITH EVERYONE RIGHT NOW",
    "edge_punctuation_only": "!!! ??? ... ,,, ---",
    "edge_single_word": "Breaking",
    "edge_digits": "42",
    "edge_very_long": (
        "A report circulating online claims that every major news organisation agreed in secret "
        "to publish the same headline on the same morning, and that the agreement was signed at "
        "a hotel whose name nobody can remember, by people whose names nobody will confirm, in a "
        "country that the report does not identify, during a month that the report does not give, "
        "for reasons the report says will become obvious to anyone who reads between the lines of "
        "any front page printed since then."
    ),
}


def require_golden() -> bool:
    """True when a missing-weights skip must be reported as a failure instead."""
    return os.environ.get("REQUIRE_GOLDEN") == "1"


def cache_root() -> Path:
    """Directory the pinned artifacts are cached in — always outside the repository."""
    override = os.environ.get("TWEETGUARD_GOLDEN_CACHE")
    root = Path(override) if override else Path.home() / ".cache" / "tweetguard-golden"
    return root / MODEL_REVISION


def artifacts_cached(root: Path) -> bool:
    """True when every pinned artifact is already present under ``root``."""
    return all((root / name).exists() for name in ARTIFACT_FILES)


def fetch_models_dir() -> Path:
    """Return the local directory holding the pinned artifacts, downloading once.

    The download is anonymous and pinned by commit hash, so the test reads the
    exact weights the results were measured with, and needs no credentials.

    Raises:
        Exception: Whatever ``snapshot_download`` raises when the Hub cannot be
            reached or the revision is gone.
    """
    root = cache_root()
    if artifacts_cached(root):
        return root

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        repo_type="model",
        allow_patterns=["paper/*", "safe/*"],
        local_dir=root,
        token=False,
    )
    return root


def models_dir_or_skip() -> Path:
    """Fetch the pinned artifacts, or skip the calling test when that is impossible.

    With ``REQUIRE_GOLDEN=1`` the skip becomes a failure, so continuous
    integration cannot go green on a guard that never ran.
    """
    try:
        return fetch_models_dir()
    except Exception as error:  # noqa: BLE001 - any Hub/network failure means "no weights"
        message = (
            f"pinned English artifacts unavailable ({MODEL_REPO}@{MODEL_REVISION[:12]}): "
            f"{type(error).__name__}: {error}"
        )
        if require_golden():
            pytest.fail(f"REQUIRE_GOLDEN=1 but {message}")
        pytest.skip(message)
