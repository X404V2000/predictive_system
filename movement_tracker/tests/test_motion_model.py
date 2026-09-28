"""
Tests for automatic motion-model selection.

The tracker must choose a process-noise profile appropriate to the observed
motion. Three failure modes motivated these tests, each found empirically:

  1. Selecting from the raw per-frame velocity created a feedback loop that
     latched into the noisiest model and never returned.
  2. Frame-based smoothing and evidence windows behaved completely differently
     at 10 Hz and 120 Hz.
  3. Judging motion by the filter's own velocity trapped genuinely walking
     targets in the STATIONARY model, because that model decays velocity.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from filter import MotionModel
from schema import TelemetryInput
from tracker import EntityTracker


def _stream(
    tracker: EntityTracker,
    velocity: np.ndarray,
    sample_dt: float = 0.05,
    steps: int = 400,
    noise_std: float = 0.02,
    seed: int = 0,
    entity: str = "target",
) -> np.ndarray:
    """Streams a constant-velocity target and returns the final state."""
    rng = np.random.default_rng(seed)
    position = np.zeros(3)
    timestamp = 1000.0

    for _ in range(steps):
        position = position + velocity * sample_dt
        timestamp += sample_dt
        observed = position + rng.normal(0.0, noise_std, size=3)
        tracker.process_telemetry(
            TelemetryInput(
                entity_id=entity,
                x=float(observed[0]),
                y=float(observed[1]),
                z=float(observed[2]),
                timestamp=timestamp,
            ),
            dt=0.5,
        )
    return tracker.filters[entity].position


def test_moving_target_is_not_classified_stationary() -> None:
    """
    A walking target must never end up in the STATIONARY model.

    The stationary model decays velocity toward zero, so misclassifying real
    motion as stationary silently suppresses the prediction.
    """
    tracker = EntityTracker(filter_enabled=True)
    _stream(tracker, np.array([1.2, 0.0, 0.0]))

    assert tracker.filters["target"].motion_model is not MotionModel.STATIONARY
    assert tracker.filters["target"].speed > 0.5


def test_stationary_target_is_classified_stationary() -> None:
    """A motionless target must settle into the STATIONARY model."""
    tracker = EntityTracker(filter_enabled=True)
    _stream(tracker, np.array([0.0, 0.0, 0.0]))

    assert tracker.filters["target"].motion_model is MotionModel.STATIONARY
    assert tracker.filters["target"].speed < 0.1


def test_fast_target_escalates_to_a_noisier_model() -> None:
    """A running target must escalate above the default process noise."""
    tracker = EntityTracker(filter_enabled=True)
    _stream(tracker, np.array([4.0, 0.0, 0.0]))

    assert tracker.filters["target"].motion_model in (
        MotionModel.MANEUVERING,
        MotionModel.CONSTANT_ACCELERATION,
    )
    assert tracker.filters["target"].speed > 2.0


def test_model_selection_is_frame_rate_independent() -> None:
    """
    The same motion at 10 Hz and 120 Hz must yield the same model choice and a
    comparable velocity estimate. Time-based smoothing is what makes this hold.
    """
    velocity = np.array([1.5, 0.0, 0.0])

    slow = EntityTracker(filter_enabled=True)
    _stream(slow, velocity, sample_dt=0.1, steps=120, entity="slow")

    fast = EntityTracker(filter_enabled=True)
    _stream(fast, velocity, sample_dt=1.0 / 120.0, steps=120 * 12, entity="fast")

    assert slow.filters["slow"].motion_model is fast.filters["fast"].motion_model
    assert slow.filters["slow"].speed == pytest.approx(
        fast.filters["fast"].speed, abs=0.2
    )


def test_adaptivity_can_be_disabled() -> None:
    """With adaptivity off, the model must stay at its initial value."""
    tracker = EntityTracker(filter_enabled=True, adaptive_motion_model=False)
    _stream(tracker, np.array([0.0, 0.0, 0.0]))

    assert tracker.filters["target"].motion_model is MotionModel.CONSTANT_ACCELERATION


def test_motion_model_state_is_cleaned_up_on_prune() -> None:
    """Pruning must clear the model-selection bookkeeping, not just the filter."""
    tracker = EntityTracker(ttl_seconds=0.0)
    _stream(tracker, np.array([1.0, 0.0, 0.0]), steps=60)

    tracker.prune_stale_entities(current_time=tracker.last_seen["target"] + 1e6)

    assert "target" not in tracker.filters
    assert "target" not in tracker._smoothed_speed
    assert "target" not in tracker._escalation_seconds
    assert "target" not in tracker._position_window
    assert "target" not in tracker._position_history
    assert "target" not in tracker._elapsed_seconds
