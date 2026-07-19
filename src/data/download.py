"""Download the TruthSeeker 2023 dataset and (optionally) GloVe embeddings.

The dataset comes from Kaggle via ``kagglehub`` (public dataset, no API key
required); GloVe comes from the Stanford NLP server. Both are cached outside
the repository so no large file can end up in git.

Usage:
    python -m src.data.download            # dataset only (enough for M1)
    python -m src.data.download --glove    # dataset + glove.6B.100d.txt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import kagglehub

from src import config


def download_dataset() -> Path:
    """Fetch (or reuse the cached copy of) the TruthSeeker 2023 dataset.

    Returns:
        Path to ``Truth_Seeker_Model_Dataset.csv`` inside the kagglehub cache.

    Raises:
        FileNotFoundError: If the download succeeded but the expected CSV is
            missing from the dataset files.
    """
    dataset_dir = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    return find_dataset_csv(dataset_dir)


def find_dataset_csv(dataset_dir: Path) -> Path:
    """Locate the model-ready CSV inside a downloaded dataset directory.

    Args:
        dataset_dir: Directory returned by ``kagglehub.dataset_download``.

    Returns:
        Path to ``Truth_Seeker_Model_Dataset.csv``.

    Raises:
        FileNotFoundError: If the expected CSV is not present.
    """
    matches = sorted(dataset_dir.rglob(config.DATASET_CSV_NAME))
    if not matches:
        raise FileNotFoundError(
            f"{config.DATASET_CSV_NAME} not found in Kaggle dataset "
            f"'{config.KAGGLE_DATASET}' (downloaded to {dataset_dir})."
        )
    return matches[0]


def download_glove() -> Path:
    """Fetch ``glove.6B.zip`` and extract the 100-d vectors into the local cache.

    The ~822 MB archive is downloaded once to ``~/.cache/tweetguard/`` and
    deleted after extraction; only ``glove.6B.100d.txt`` (~350 MB) is kept.

    Returns:
        Path to the extracted ``glove.6B.100d.txt``.
    """
    target = config.CACHE_DIR / config.GLOVE_FILE
    if target.exists():
        print(f"GloVe already cached: {target}")
        return target

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.CACHE_DIR / "glove.6B.zip"
    if not archive.exists():
        print(f"Downloading {config.GLOVE_URL} (~822 MB, one-off)...")
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(config.GLOVE_URL, partial, _report_progress)
        print()
        partial.rename(archive)

    with zipfile.ZipFile(archive) as bundle:
        bundle.extract(config.GLOVE_FILE, config.CACHE_DIR)
    archive.unlink()
    return target


def _report_progress(blocks_read: int, block_size: int, total_bytes: int) -> None:
    """Print download progress roughly every 50 MB (urlretrieve report hook)."""
    downloaded = blocks_read * block_size
    if total_bytes > 0 and downloaded % (50 * 1024 * 1024) < block_size:
        print(f"  {downloaded / 1e6:,.0f} / {total_bytes / 1e6:,.0f} MB", flush=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: download the dataset, plus GloVe when ``--glove`` is set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glove", action="store_true", help="also download the GloVe 100-d embeddings"
    )
    args = parser.parse_args(argv)

    csv_path = download_dataset()
    print(f"Dataset CSV ready: {csv_path}")
    if args.glove:
        print(f"GloVe embeddings ready: {download_glove()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
