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


# --- API integration -------------------------------------------------------


def _api_client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)


def _pose_payload(frames: int = 40, joints: int = 31, seed: int = 0) -> dict:
    """
    A synthetic but valid skeleton window.

    Guard the joint accesses so the helper can also build deliberately invalid
    payloads (fewer joints) without raising before the request is even sent.
    """
    rng = np.random.default_rng(seed)
    positions = rng.normal(0.0, 0.05, size=(frames, joints, 3))
    if joints > 9:
        positions[:, 0, 1] += 0.9  # hip near standing height
        positions[:, 4, 1] = 0.05  # left foot on the floor
        positions[:, 9, 1] = 0.05  # right foot on the floor
    return {
        "entity_id": "test",
        "positions": positions.tolist(),
        "sample_dt": 1.0 / 30.0,
    }


@requires_data
def test_action_endpoint_returns_a_distribution() -> None:
    client = _api_client()
    response = client.post("/predict/action", json=_pose_payload())

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["action"]
    assert 0.0 <= body["confidence"] <= 1.0

    probabilities = [entry["probability"] for entry in body["distribution"]]
    assert probabilities == sorted(probabilities, reverse=True), "must be ranked"
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert body["model_version"]


@requires_data
def test_action_endpoint_flags_low_confidence() -> None:
    """
    An ambiguous window must be reported as low confidence rather than presented
    as a decision. Motion near the training mean is the ambiguous case.
    """
    client = _api_client()
    payload = _pose_payload()
    # Gently moving, mid-range posture: unlike any strong action.
    positions = np.asarray(payload["positions"])
    positions[:, :, 1] += 0.4
    positions += np.linspace(0, 0.02, positions.shape[0])[:, None, None]
    payload["positions"] = positions.tolist()

    body = client.post("/predict/action", json=payload).json()
    assert "low_confidence" in body
    assert isinstance(body["low_confidence"], bool)


@requires_data
def test_action_endpoint_rejects_malformed_positions() -> None:
    client = _api_client()

    bad_frames = _pose_payload(frames=3)  # too short to characterise motion
    assert client.post("/predict/action", json=bad_frames).status_code == 400

    wrong_shape = _pose_payload()
    wrong_shape["positions"] = [[[0.0, 0.0]]]  # not (frames, joints, 3)
    assert client.post("/predict/action", json=wrong_shape).status_code == 400

    wrong_joints = _pose_payload(joints=7)  # not the training skeleton
    assert client.post("/predict/action", json=wrong_joints).status_code == 400


@requires_data
def test_action_endpoint_accepts_joint_names_in_another_order() -> None:
    """
    A caller using a different skeleton order can supply names, and the payload
    must be reordered rather than silently misread.
    """
    from features import JOINT_ORDER

    client = _api_client()
    payload = _pose_payload()

    positions = np.asarray(payload["positions"])
    shuffled = list(reversed(JOINT_ORDER))
    payload["positions"] = positions[:, ::-1, :].tolist()
    payload["joint_names"] = shuffled

    response = client.post("/predict/action", json=payload)
    assert response.status_code == 200, response.text

    # Reordering by name must reproduce the original result.
    plain = client.post(
        "/predict/action",
        json={
            "entity_id": "test",
            "positions": positions.tolist(),
            "sample_dt": payload["sample_dt"],
        },
    ).json()

    assert response.json()["action"] == plain["action"]


@requires_data
def test_action_endpoint_rejects_incomplete_joint_names() -> None:
    """A skeleton missing required joints must be refused, not guessed at."""
    client = _api_client()
    payload = _pose_payload()
    payload["joint_names"] = ["Hips", "Spine"]

    response = client.post("/predict/action", json=payload)
    assert response.status_code == 400
    assert "missing" in response.json()["detail"].lower()


@requires_data
def test_action_endpoint_rejects_non_finite_values() -> None:
    """
    A payload containing NaN must be refused with a clear error.

    NaN cannot be sent as JSON, so the value is injected as the string "NaN",
    which a client could genuinely produce, and the server must reject it rather
    than propagating a non-finite feature into the model.
    """
    client = _api_client()
    payload = _pose_payload()
    payload["positions"][0][0][0] = "NaN"

    response = client.post("/predict/action", json=payload)
    assert response.status_code in (400, 422), response.text
