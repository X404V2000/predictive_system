"""
Tests for the probabilistic action classifier.

Two properties matter beyond raw accuracy:

  * the probabilities must be CALIBRATED. The first version claimed 98.3% mean
    confidence while being right 85.1% of the time, with 86% of predictions at
    99%+ certainty, so every mistake was a confident one. The covariance ridge is
    set to keep that gap small.
  * accuracy must be measured by holding out whole SUBJECTS. Class and subject
    are confounded here (most classes appear in only two of five subjects), so a
    random clip split would leak identity and report a flattering number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from actions import (
    ACTION_RULES,
    GaussianActionModel,
    coarse_action,
    load_labelled_features,
)
from features import FEATURE_NAMES, ClipFeatures

_HAS_LABELS = (
    Path(__file__).resolve().parent.parent
    / "training_data"
    / "cmu_mocap"
    / "action_labels.tsv"
).exists()
requires_data = pytest.mark.skipif(not _HAS_LABELS, reason="labelled data not present")


# --- label mapping (no data needed) ---------------------------------------


def test_coarse_action_maps_catalogue_descriptions() -> None:
    cases = {
        "walk": "locomote",
        "slow walk": "locomote",
        "run": "locomote",
        "jog": "locomote",
        "sit on high stool, stand up": "seated",
        "climb ladder": "climb",
        "forward jump": "jump",
        "direct traffic, wave, point": "gesture",
        "wash windows": "manipulate",
        "sweep floor": "manipulate",
        "boxing": "exercise",
        "laugh": "exercise",
    }
    for description, expected in cases.items():
        assert coarse_action(description) == expected, description


def test_coarse_action_is_total() -> None:
    """An unrecognised description must not raise; it maps to 'other'."""
    assert coarse_action("totally unknown activity") == "other"


def test_action_rules_cover_distinct_classes() -> None:
    names = [name for name, _ in ACTION_RULES]
    assert len(names) == len(set(names)), "action classes must be unique"


# --- model behaviour (synthetic) ------------------------------------------


def _synthetic(seed: int, n_per_class: int = 12) -> list[ClipFeatures]:
    """Builds two well-separated synthetic classes for unit testing."""
    rng = np.random.default_rng(seed)
    samples: list[ClipFeatures] = []
    for index, (label, offset) in enumerate((("walk", -3.0), ("sit on stool", 3.0))):
        for i in range(n_per_class):
            values = rng.normal(0.0, 0.4, size=len(FEATURE_NAMES))
            values[0] += offset
            samples.append(
                ClipFeatures(
                    clip_id=f"c{index}_{i}",
                    subject=f"S{index}",
                    label=label,
                    values=values,
                )
            )
    return samples


def test_probabilities_form_a_distribution() -> None:
    model = GaussianActionModel.fit(_synthetic(0))
    probabilities = model.predict_proba(np.zeros(len(FEATURE_NAMES)))

    assert probabilities.shape == (len(model.classes),)
    assert np.all(probabilities >= 0.0)
    assert probabilities.sum() == pytest.approx(1.0)


def test_separated_classes_are_classified_correctly() -> None:
    model = GaussianActionModel.fit(_synthetic(0))
    left = np.zeros(len(FEATURE_NAMES))
    left[0] = -3.0
    right = np.zeros(len(FEATURE_NAMES))
    right[0] = 3.0

    assert model.predict(left)[0] == "locomote"
    assert model.predict(right)[0] == "seated"


def test_temperature_softens_confidence() -> None:
    """Raising the temperature must reduce peak confidence, never increase it."""
    samples = _synthetic(1)
    model = GaussianActionModel.fit(samples)
    point = np.zeros(len(FEATURE_NAMES))
    point[0] = -3.0

    sharp = float(model.predict_proba(point).max())
    model.temperature = 20.0
    soft = float(model.predict_proba(point).max())

    assert soft < sharp


def test_fit_temperature_returns_a_positive_value() -> None:
    samples = _synthetic(2)
    model = GaussianActionModel.fit(samples)
    temperature = model.fit_temperature(samples, grid=np.array([1.0, 5.0, 20.0]))
    assert temperature > 0.0
    assert model.temperature == temperature


def test_fit_rejects_a_single_class() -> None:
    only_walk = [s for s in _synthetic(3) if coarse_action(s.label) == "locomote"]
    with pytest.raises(ValueError):
        GaussianActionModel.fit(only_walk)


def test_fit_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        GaussianActionModel.fit([])


def test_unfitted_model_refuses_inference() -> None:
    with pytest.raises(ValueError):
        GaussianActionModel().predict_proba(np.zeros(len(FEATURE_NAMES)))


def test_covariance_is_positive_definite() -> None:
    """A non-invertible covariance would make the likelihood meaningless."""
    model = GaussianActionModel.fit(_synthetic(4))
    eigenvalues = np.linalg.eigvalsh(model.covariance)
    assert eigenvalues.min() > 0.0


# --- real-data behaviour ---------------------------------------------------


@requires_data
def test_labelled_features_load_with_subjects() -> None:
    samples, subjects = load_labelled_features()
    assert len(samples) == len(subjects) > 50
    assert all(s.values.shape == (len(FEATURE_NAMES),) for s in samples)
    assert all(np.all(np.isfinite(s.values)) for s in samples)


@requires_data
def test_leave_one_subject_out_beats_majority_baseline() -> None:
    """
    The honest accuracy figure: the classifier must generalise to an unseen
    person, and beat always predicting the most common class by a wide margin.
    """
    samples, subjects = load_labelled_features()
    unique = sorted(set(subjects))

    truths, predictions = [], []
    for held_out in unique:
        train = [s for s, sub in zip(samples, subjects) if sub != held_out]
        test = [s for s, sub in zip(samples, subjects) if sub == held_out]
        train_classes = {coarse_action(s.label) for s in train}
        test = [s for s in test if coarse_action(s.label) in train_classes]
        if not test:
            continue

        model = GaussianActionModel.fit(train)
        for s in test:
            truths.append(coarse_action(s.label))
            predictions.append(model.predict(s.values)[0])

    accuracy = float(np.mean(np.array(truths) == np.array(predictions)))
    majority = max(truths.count(t) for t in set(truths)) / len(truths)

    assert accuracy > 0.65, f"leave-one-subject-out accuracy {accuracy:.1%} too low"
    assert accuracy > majority + 0.3


@requires_data
def test_confidences_are_calibrated_on_held_out_subjects() -> None:
    """
    Mean confidence must not exceed observed accuracy by a wide margin.

    An overconfident classifier is worse than a less accurate honest one, because
    downstream consumers act on the stated probability.
    """
    samples, subjects = load_labelled_features()
    unique = sorted(set(subjects))

    confidences, outcomes = [], []
    for held_out in unique:
        train = [s for s, sub in zip(samples, subjects) if sub != held_out]
        test = [s for s, sub in zip(samples, subjects) if sub == held_out]
        train_classes = {coarse_action(s.label) for s in train}
        test = [s for s in test if coarse_action(s.label) in train_classes]
        if not test:
            continue

        model = GaussianActionModel.fit(train)
        for s in test:
            predicted, confidence = model.predict(s.values)
            confidences.append(confidence)
            outcomes.append(int(predicted == coarse_action(s.label)))

    gap = float(np.mean(confidences)) - float(np.mean(outcomes))
    assert gap < 0.10, (
        f"classifier is overconfident: claims {np.mean(confidences):.3f} but is "
        f"right {np.mean(outcomes):.3f} (gap {gap:+.3f})"
    )
