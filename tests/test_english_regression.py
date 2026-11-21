"""English regression guard: the published results and predictions are frozen.

Two halves, deliberately separate:

* the **results guard** needs no weights. It asserts that every one of the 24
  metric fields in ``reports/metrics.json`` still holds the value it had when
  the English pipeline was frozen, and that the provenance file agrees with the
  pinned revision. Any edit to a published number fails here.
* the **golden-output guard** loads the pinned weights and re-predicts a fixed
  set of inputs through the same calls ``/predict`` makes. Labels must match
  exactly, probabilities to 1e-6.

Every later pull request must leave this file passing untouched.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.models import predict
from tests import golden_support as support

EXPECTED_METRICS = json.loads(support.METRICS_EXPECTED_JSON.read_text(encoding="utf-8"))
GOLDEN = json.loads(support.GOLDEN_JSON.read_text(encoding="utf-8"))
MODEL_KEYS = [
    f"{protocol}/{name}" for protocol in support.PROTOCOLS for name in support.MODEL_NAMES
]


def published_metrics() -> dict[str, object]:
    """Flatten ``reports/metrics.json`` to ``protocol/model/field`` keys."""
    raw = json.loads((config.REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    return {
        f"{protocol}/{model}/{field}": value
        for protocol in raw
        for model in raw[protocol]
        for field, value in raw[protocol][model].items()
    }


# --- results guard (no weights needed) ---------------------------------------


def test_frozen_expectation_covers_twenty_four_fields():
    assert EXPECTED_METRICS["field_count"] == 24
    assert len(EXPECTED_METRICS["fields"]) == 24


@pytest.mark.parametrize("field", sorted(EXPECTED_METRICS["fields"]))
def test_published_metric_is_unchanged(field):
    published = published_metrics()
    assert field in published, f"{field} disappeared from reports/metrics.json"
    assert published[field] == EXPECTED_METRICS["fields"][field]


def test_published_metrics_have_no_extra_or_missing_fields():
    assert set(published_metrics()) == set(EXPECTED_METRICS["fields"])


def test_provenance_records_the_pinned_revision_and_every_run():
    provenance = json.loads(
        (config.REPORTS_DIR / "metrics" / "english_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["artifacts"]["revision"] == support.MODEL_REVISION
    assert provenance["artifacts"]["model_repo"] == support.MODEL_REPO
    assert sorted(provenance["mlflow"]["runs"]) == sorted(MODEL_KEYS)
    assert all(len(run) == 32 for run in provenance["mlflow"]["runs"].values())
    assert provenance["seed"] == config.SEED


def test_golden_file_pins_the_same_revision():
    assert GOLDEN["revision"] == support.MODEL_REVISION
    assert GOLDEN["model_repo"] == support.MODEL_REPO
    assert GOLDEN["inputs"] == support.INPUTS
    assert sorted(GOLDEN["predictions"]) == sorted(support.INPUTS)
    for per_input in GOLDEN["predictions"].values():
        assert sorted(per_input) == sorted(MODEL_KEYS)


# --- golden-output guard (needs the pinned weights) --------------------------


@pytest.fixture(scope="module")
def pinned_models_dir():
    """The directory holding the pinned artifacts (skips, or fails under REQUIRE_GOLDEN=1)."""
    return support.models_dir_or_skip()


@pytest.fixture
def pinned_config(pinned_models_dir, monkeypatch):
    """Point the loaders at the pinned artifacts instead of the local models tree."""
    monkeypatch.setattr(config, "MODELS_DIR", pinned_models_dir)
    return pinned_models_dir


@pytest.mark.parametrize("protocol", support.PROTOCOLS)
@pytest.mark.parametrize("model_name", support.MODEL_NAMES)
def test_predictions_match_the_golden_values(pinned_config, protocol, model_name):
    """Re-predict every fixed input and compare with the recorded values.

    One test per model and protocol, so a failure names the model and reports
    its largest probability difference rather than stopping at the first input.
    """
    key = f"{protocol}/{model_name}"
    model = predict.load_model(model_name, protocol)

    texts = list(support.INPUTS)
    probabilities = predict.predict_proba(model, model_name, [support.INPUTS[k] for k in texts])

    label_mismatches = []
    worst_input, worst_difference = None, 0.0
    for input_key, probability in zip(texts, probabilities, strict=True):
        expected = GOLDEN["predictions"][input_key][key]
        probability = float(probability)
        label, confidence = predict.decide(probability)

        if label != expected["label"]:
            label_mismatches.append((input_key, expected["label"], label))
        for measured, recorded in (
            (probability, expected["p_real"]),
            (confidence, expected["confidence"]),
        ):
            difference = abs(measured - recorded)
            if difference > worst_difference:
                worst_input, worst_difference = input_key, difference

    # Printed on success too (visible with -s), so the drift between the machine
    # the golden values were generated on and the machine running the guard is a
    # recorded number rather than a pass/fail bit.
    print(f"golden-drift {key}: max probability difference {worst_difference:.3e}")

    assert not label_mismatches, f"{key}: labels changed: {label_mismatches}"
    assert worst_difference <= support.PROBABILITY_TOLERANCE, (
        f"{key}: largest probability difference {worst_difference:.3e} exceeds "
        f"{support.PROBABILITY_TOLERANCE:.0e} (worst input: {worst_input!r})"
    )


def test_predict_one_agrees_with_the_golden_values(pinned_config):
    """``predict_one`` — the single-tweet entry point — stays consistent too."""
    text = support.INPUTS["true_rates"]
    result = predict.predict_one(text, "baseline", "safe")
    expected = GOLDEN["predictions"]["true_rates"]["safe/baseline"]
    assert result["label"] == expected["label"]
    assert result["confidence"] == pytest.approx(round(expected["confidence"], 4), abs=1e-9)


def test_continuous_integration_pins_the_same_revision_and_requires_the_guard():
    """The workflow must cache the pinned revision and refuse to skip the guard."""
    workflow = (config.PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert support.MODEL_REVISION in workflow
    assert 'REQUIRE_GOLDEN: "1"' in workflow
