"""
Tests for action sequence segmentation and transition modelling.

The transition model answers "given the current action, what tends to follow?".
Two honest constraints shape these tests:

  * Clip labels describe sequences ("sit on high stool, stand up") but carry NO
    timings, so segment boundaries must be inferred from motion. A misclassified
    window becomes a spurious transition, which is why the model reports its own
    accuracy rather than presenting itself as precise.
  * High-stool sitting raises the hip only ~0.06 m below standing, so it is
    nearly indistinguishable from standing using posture features. Clips on a
    stepstool (hip ~0.6 m) are clearly separable; high-stool clips are not. This
    is a property of the data, not a defect, and it should stay visible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from actions import GaussianActionModel, load_labelled_features
from features import extract_features
from mocap import DEFAULT_DATA_ROOT, load_clip
from transitions import (
    END_ACTION,
    START_ACTION,
    MIN_TRANSITION_CONFIDENCE,
    SegmentedClip,
    TransitionModel,
    _subclip,
)

_HAS_DATA = DEFAULT_DATA_ROOT.exists() and any(DEFAULT_DATA_ROOT.glob("*/*.bvh"))
_HAS_LABELS = (
    Path(__file__).resolve().parent.parent / "training_data" / "cmu_mocap" / "action_labels.tsv"
).exists()
requires_data = pytest.mark.skipif(
    not (_HAS_DATA and _HAS_LABELS), reason="labelled motion capture not present"
)

# Segmentation re-reads motion capture for every 2 s window, so the two tests
# that need real sequences take a few minutes. They stay in the default run
# because they encode the measured limitation of the transition model, but they
# are marked so the cost is visible and can be deselected with `-m "not slow"`.
slow = pytest.mark.slow

STEPSTOOL_CLIPS = ["013/13_04", "013/13_05", "014/14_30", "014/14_31"]


# --- segmentation mechanics ------------------------------------------------


def _synthetic_sequences() -> list[SegmentedClip]:
    return [
        SegmentedClip("S1/1", "sit, stand", ("seated", "seated", "locomote"), (0.9, 0.8, 0.7)),
        SegmentedClip("S2/1", "sit, stand", ("seated", "locomote", "locomote"), (0.8, 0.9, 0.9)),
        SegmentedClip("S3/1", "walk, jog", ("locomote", "locomote"), (0.9, 0.9)),
    ]


def test_sequence_includes_start_and_end_sentinels() -> None:
    clip = SegmentedClip("S/1", "walk", ("locomote", "exercise"), (0.9, 0.8))
    assert clip.sequence == (START_ACTION, "locomote", "exercise", END_ACTION)


def test_transition_matrix_rows_are_distributions() -> None:
    model = TransitionModel.fit(_synthetic_sequences())
    assert model.probabilities is not None

    assert np.allclose(model.probabilities.sum(axis=1), 1.0)
    assert np.all(model.probabilities >= 0.0)


def test_end_state_is_absorbing() -> None:
    """Nothing follows the end of a clip."""
    model = TransitionModel.fit(_synthetic_sequences())
    row = model.next_action_distribution(END_ACTION)
    assert row[END_ACTION] == pytest.approx(1.0)


def test_self_transitions_are_learned() -> None:
    """
    An action that repeats in most sequences should predict itself as the most
    likely successor.

    The probability is not asserted to be near 1: smoothing spreads mass across
    every state, and sequences genuinely end, so the value is moderate.
    """
    model = TransitionModel.fit(_synthetic_sequences())
    predicted, probability = model.most_likely_next("locomote")

    assert predicted == "locomote"
    distribution = model.next_action_distribution("locomote")
    assert probability == max(
        p for action, p in distribution.items() if action != END_ACTION
    )


def test_seated_is_followed_by_locomotion_in_observed_data() -> None:
    model = TransitionModel.fit(_synthetic_sequences())
    distribution = model.next_action_distribution("seated")
    assert "locomote" in distribution
    assert distribution["locomote"] > distribution.get("seated", 0.0)


def test_unobserved_transition_stays_possible() -> None:
    """Smoothing must keep an unseen pair non-zero rather than impossible."""
    sequences = [
        SegmentedClip("S1/1", "a", ("locomote",), (0.9,)),
        SegmentedClip("S2/1", "a", ("locomote",), (0.9,)),
    ]
    model = TransitionModel.fit(sequences)
    # "exercise" never appears, so it is not a state at all; but every listed
    # state must retain non-zero mass everywhere it could lead.
    distribution = model.next_action_distribution(START_ACTION)
    assert all(p > 0.0 for p in distribution.values())


def test_unknown_current_action_falls_back_to_prior() -> None:
    model = TransitionModel.fit(_synthetic_sequences())
    fallback = model.next_action_distribution("never_seen_action")
    assert abs(sum(fallback.values()) - 1.0) < 1e-9


def test_fit_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        TransitionModel.fit([])


def test_unfitted_model_refuses_inference() -> None:
    with pytest.raises(ValueError):
        TransitionModel().next_action_distribution(START_ACTION)


# --- real-data segmentation ------------------------------------------------


@requires_data
def test_subclip_aligns_joints_with_root_trajectory() -> None:
    """
    A sub-clip must read joint data for the SAME frame window as its root
    trajectory, otherwise forward kinematics and the root disagree in length.
    """
    from features import load_joint_positions

    full = load_clip(
        next(iter(DEFAULT_DATA_ROOT.glob("013/13_04.bvh")))
    )
    sub = _subclip(full, 240, 480)

    assert sub.frame_count == 240
    assert sub.frame_range == (240, 480)

    joints, skeleton = load_joint_positions(sub)
    assert joints.shape[0] == sub.frame_count

    # Root Y from the stored trajectory must equal the hip height from FK.
    hip_from_fk = joints[:, skeleton.index_of("Hips"), 1]
    assert np.allclose(hip_from_fk, sub.positions[:, 1], atol=1e-9)


@requires_data
@slow
def test_stepstool_clips_are_classified_as_seated() -> None:
    """
    A stepstool lowers the hip to ~0.6 m, well below standing, so these clips
    must be separable. This is the case the features are actually able to model.
    """
    samples, _ = load_labelled_features()
    model = GaussianActionModel.fit(samples)

    for clip_id in STEPSTOOL_CLIPS:
        subject, name = clip_id.split("/")
        path = DEFAULT_DATA_ROOT / subject / f"{name}.bvh"
        if not path.exists():
            continue
        features = extract_features(load_clip(path), label="sit on stepstool")
        predicted, confidence = model.predict(features.values)
        assert predicted == "seated", (
            f"{clip_id}: expected seated, got {predicted} ({confidence:.2f})"
        )


@requires_data
@slow
def test_high_stool_sitting_is_nearly_indistinguishable_from_standing() -> None:
    """
    Documents a real limitation of the feature set rather than asserting a
    capability that does not exist.

    On a high stool the hip sits only ~0.06 m below standing height, so posture
    alone cannot separate the two. This test fails loudly if the data ever
    changes such that the distinction becomes available, prompting a rethink.
    """
    def hip_mean(clip_id: str) -> float:
        subject, name = clip_id.split("/")
        path = DEFAULT_DATA_ROOT / subject / f"{name}.bvh"
        return float(
            extract_features(load_clip(path), label="sit").as_dict()["hip_height_mean"]
        )

    high_stool = hip_mean("013/13_01") if (DEFAULT_DATA_ROOT / "013/13_01.bvh").exists() else None
    standing = hip_mean("007/07_01")

    if high_stool is None:
        pytest.skip("high-stool clip not present")

    # The gap between high-stool sitting and standing is small.
    assert abs(standing - high_stool) < 0.15, (
        "high-stool and standing hip heights are now separable; revisit the "
        "seated action class"
    )


@requires_data
@slow
def test_transition_model_has_not_yet_demonstrated_lift() -> None:
    """
    Records the CURRENT measured state of the transition model: it does not beat
    a constant predictor.

    This is a guard, not an endorsement. If a future change gives the model real
    lift, this test will fail, and that failure is the signal to update the
    module docstring and start presenting the capability. Until then, code and
    documentation should not claim "predicts the next movement".
    """
    from collections import Counter

    from actions import GaussianActionModel, load_labelled_features
    from transitions import collect_sequences

    samples, _ = load_labelled_features()
    model = GaussianActionModel.fit(samples)
    sequences = collect_sequences(model)

    pairs = [
        (a, b)
        for s in sequences
        for a, b in zip(s.actions, s.actions[1:])
        if a != "uncertain" and b != "uncertain"
    ]
    assert pairs, "no adjacent action pairs available"

    by_current: dict[str, list[str]] = {}
    for current, following in pairs:
        by_current.setdefault(current, []).append(following)

    transition = TransitionModel.fit(sequences)

    model_hits = 0.0
    baseline_hits = 0.0
    considered = 0
    for current, followings in by_current.items():
        if len(followings) < 3:
            continue
        predicted, _ = transition.most_likely_next(current)
        model_hits += sum(1 for f in followings if f == predicted) / len(followings)
        majority = Counter(followings).most_common(1)[0]
        baseline_hits += majority[1] / len(followings)
        considered += 1

    assert considered >= 3, "not enough states to compare"
    model_balanced = model_hits / considered
    baseline_balanced = baseline_hits / considered

    # The model currently matches rather than beats the baseline. If lift appears,
    # this assertion fails deliberately: revisit the docs and this guard.
    assert model_balanced <= baseline_balanced + 0.10, (
        f"transition model now shows real lift "
        f"({model_balanced:.1%} vs {baseline_balanced:.1%}); update the "
        f"module docstring and stop reporting it as not-yet-useful"
    )
