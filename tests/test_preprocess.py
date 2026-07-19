"""Tests for the M1 data pipeline: cleaning, augmentation, and splitting."""

import random

import pandas as pd
import pytest

from src import config
from src.data.download import find_dataset_csv
from src.data.preprocess import augment, build_vocabulary, clean_text, load_and_clean, word_swap
from src.data.split import split_dataframe

# --- clean_text ---


def test_clean_text_collapses_whitespace():
    assert clean_text("  Breaking:\t news \n today  ") == "Breaking: news today"


def test_clean_text_missing_values_become_empty():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_whitespace_only_becomes_empty():
    assert clean_text("   ") == ""


def test_clean_text_non_string_scalars_are_stringified():
    assert clean_text(42) == "42"


def test_clean_text_preserves_emoji_urls_and_non_latin_text():
    text = "منشور مزيف 🚨 https://example.com/x?y=1"
    assert clean_text(text) == text


# --- load_and_clean ---


def _write_raw_csv(path, rows):
    frame = pd.DataFrame(rows, columns=["author", config.TEXT_COLUMN, config.LABEL_COLUMN])
    frame.to_csv(path, index=False)
    return path


def test_load_and_clean_selects_and_cleans_columns(tmp_path):
    csv = _write_raw_csv(
        tmp_path / "raw.csv",
        [["a", "  real   news ", 1.0], ["b", "fake news", 0.0]],
    )
    frame = load_and_clean(csv)
    assert list(frame.columns) == ["tweet", "label"]
    assert frame["tweet"].tolist() == ["real news", "fake news"]
    assert frame["label"].tolist() == [1, 0]
    assert pd.api.types.is_integer_dtype(frame["label"])


def test_load_and_clean_drops_empty_and_unlabelled_rows(tmp_path):
    csv = _write_raw_csv(
        tmp_path / "raw.csv",
        [
            ["a", "kept", 1.0],
            ["b", None, 0.0],  # missing tweet -> dropped
            ["c", "   ", 1.0],  # whitespace-only tweet -> dropped
            ["d", "no label", None],  # missing label -> dropped
        ],
    )
    frame = load_and_clean(csv)
    assert frame["tweet"].tolist() == ["kept"]
    assert frame.index.tolist() == [0]


# --- vocabulary and word swap ---


def test_build_vocabulary_is_unique_and_sorted():
    assert build_vocabulary(["b a", "a c"]) == ["a", "b", "c"]


def test_word_swap_zero_probability_is_identity():
    rng = random.Random(config.SEED)
    assert word_swap("nothing changes here", ["x"], rng, swap_prob=0.0) == "nothing changes here"


def test_word_swap_full_probability_only_uses_vocabulary_words():
    rng = random.Random(config.SEED)
    swapped = word_swap("one two three", ["x", "y"], rng, swap_prob=1.0)
    assert len(swapped.split()) == 3
    assert set(swapped.split()) <= {"x", "y"}


def test_word_swap_is_deterministic_for_a_given_seed():
    args = ("w1 w2 w3 w4 w5", ["p", "q"])
    first = word_swap(*args, random.Random(7), swap_prob=0.5)
    second = word_swap(*args, random.Random(7), swap_prob=0.5)
    assert first == second


# --- augment ---


@pytest.fixture
def small_frame():
    return pd.DataFrame(
        {
            "tweet": [f"tweet number {i} about the news" for i in range(10)],
            "label": [i % 2 for i in range(10)],
        }
    )


def test_augment_reaches_exact_target(small_frame):
    assert len(augment(small_frame, target_rows=37)) == 37


def test_augment_keeps_originals_untouched_first(small_frame):
    grown = augment(small_frame, target_rows=30)
    pd.testing.assert_frame_equal(grown.iloc[:10].reset_index(drop=True), small_frame)


def test_augment_preserves_binary_labels(small_frame):
    grown = augment(small_frame, target_rows=50)
    assert set(grown["label"].unique()) <= {0, 1}


def test_augment_is_deterministic(small_frame):
    first = augment(small_frame, target_rows=40, seed=config.SEED)
    second = augment(small_frame, target_rows=40, seed=config.SEED)
    pd.testing.assert_frame_equal(first, second)


def test_augment_is_a_noop_when_already_large_enough(small_frame):
    pd.testing.assert_frame_equal(augment(small_frame, target_rows=5), small_frame)


def test_augmented_rows_only_use_corpus_vocabulary(small_frame):
    grown = augment(small_frame, target_rows=60, swap_prob=1.0)
    vocabulary = set(build_vocabulary(small_frame["tweet"]))
    for tweet in grown["tweet"]:
        assert set(tweet.split()) <= vocabulary


# --- split ---


def _hundred_rows():
    return pd.DataFrame(
        {"tweet": [f"t{i}" for i in range(100)], "label": [i % 2 for i in range(100)]}
    )


def test_split_dataframe_produces_80_20():
    train, test = split_dataframe(_hundred_rows(), test_size=config.TEST_SIZE, seed=config.SEED)
    assert (len(train), len(test)) == (80, 20)


def test_split_dataframe_partitions_all_rows_without_overlap():
    train, test = split_dataframe(_hundred_rows())
    train_tweets, test_tweets = set(train["tweet"]), set(test["tweet"])
    assert train_tweets.isdisjoint(test_tweets)
    assert train_tweets | test_tweets == set(_hundred_rows()["tweet"])


def test_split_dataframe_is_deterministic():
    _, first = split_dataframe(_hundred_rows(), seed=config.SEED)
    _, second = split_dataframe(_hundred_rows(), seed=config.SEED)
    assert first["tweet"].tolist() == second["tweet"].tolist()


# --- download helpers ---


def test_find_dataset_csv_locates_nested_file(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    target = nested / config.DATASET_CSV_NAME
    target.write_text("stub", encoding="utf-8")
    assert find_dataset_csv(tmp_path) == target


def test_find_dataset_csv_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_dataset_csv(tmp_path)
