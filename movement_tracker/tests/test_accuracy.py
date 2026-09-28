"""
Accuracy Regression Tests for the Prediction Pipeline.

These tests measure absolute prediction error against known ground truth and
verify the behavioural guarantees that matter most:

  * the filter recovers true velocity from position observations alone
  * accuracy does not depend on the telemetry frame rate

An earlier version of this file compared the filter against a "naive" baseline
that was handed the true velocity for free — something no real sensor provides,
since it made the baseline look artificially good. The fair, shared-observation
comparison now lives in `evaluate.py` and is exercised by
`tests/test_evaluate.py`; this file covers the filter's own correctness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from schema import TelemetryInput
from tracker import EntityTracker


def _run_constant_velocity(
    sample_dt: float,
    true_velocity: np.ndarray,
    horizon: float,
    noise_std: float,
    steps: int = 200,
    burn_in: int = 80,
    seed: int = 12345,
    jitter: float = 0.0,
) -> tuple[float, np.ndarray]:
    """
    Streams a constant-velocity target through the pipeline.

    Returns (mean prediction error in metres, final velocity estimate).
    """
    rng = np.random.default_rng(seed)
    tracker = EntityTracker(filter_enabled=True)

    position = np.zeros(3)
    timestamp = 1000.0
    errors: list[float] = []

    for _ in range(steps):
        position = position + true_velocity * sample_dt
        timestamp += sample_dt + (rng.uniform(-jitter, jitter) if jitter else 0.0)

        observed = position + rng.normal(0.0, noise_std, size=3)
        telemetry = TelemetryInput(
            entity_id="accuracy_target",
            x=float(observed[0]),
            y=float(observed[1]),
            z=float(observed[2]),
            timestamp=timestamp,
        )

        prediction = tracker.process_telemetry(telemetry, dt=horizon)
        truth = position + true_velocity * horizon
        predicted = np.array(
            [prediction.predicted_x, prediction.predicted_y, prediction.predicted_z]
        )
        errors.append(float(np.linalg.norm(predicted - truth)))

    return float(np.mean(errors[burn_in:])), tracker.filters["accuracy_target"].velocity


def test_filter_recovers_true_velocity() -> None:
    """
    The pipeline must recover the true velocity from position observations alone.

    A client-supplied velocity is deliberately not trusted as a measurement, so
    this is the only route by which motion can be learned.
    """
    true_velocity = np.array([1.2, -0.4, 0.0])
    _, estimated = _run_constant_velocity(
        sample_dt=0.05, true_velocity=true_velocity, horizon=0.5, noise_std=0.02
    )
    assert np.allclose(estimated, true_velocity, atol=0.1), (
        f"velocity estimate {estimated} should match {true_velocity}"
    )


def test_prediction_error_is_small_for_walking_speed() -> None:
    """A walking-speed target must be predicted to well under half a metre."""
    error, _ = _run_constant_velocity(
        sample_dt=0.1,
        true_velocity=np.array([1.4, 0.0, 0.0]),
        horizon=0.5,
        noise_std=0.05,
    )
    assert error < 0.5, f"prediction error {error:.4f} m exceeds the 0.5 m bound"


def test_accuracy_is_roughly_independent_of_frame_rate() -> None:
    """
    The same motion sampled at very different rates must give comparable
    accuracy. The speed smoothing and model-escalation window are expressed in
    seconds precisely so this holds.
    """
    velocity = np.array([2.0, 0.0, 0.0])
    slow_error, _ = _run_constant_velocity(
        sample_dt=0.1, true_velocity=velocity, horizon=0.5, noise_std=0.02
    )
    fast_error, _ = _run_constant_velocity(
        sample_dt=1.0 / 120.0, true_velocity=velocity, horizon=0.5, noise_std=0.02
    )

    assert slow_error < 0.5 and fast_error < 0.5, (
        f"errors not comparable across frame rates: {slow_error:.4f} m at 10 Hz, "
        f"{fast_error:.4f} m at 120 Hz"
    )


def test_irregular_arrival_intervals_are_handled() -> None:
    """
    Timestamps jitter while the motion stays constant. Because the integration
    step comes from the actual timestamps, accuracy must survive the jitter.
    """
    error, velocity = _run_constant_velocity(
        sample_dt=0.05,
        true_velocity=np.array([1.5, 0.5, 0.0]),
        horizon=0.5,
        noise_std=0.02,
        jitter=0.01,
    )
    assert error < 0.5, f"jittered-arrival error {error:.4f} m is implausible"
    assert np.linalg.norm(velocity) == pytest.approx(np.hypot(1.5, 0.5), abs=0.15)


def test_duplicate_timestamp_does_not_destabilise_filter() -> None:
    """
    Two frames sharing a timestamp give dt == 0, which would produce a singular
    transition if unguarded. The minimum-interval clamp must keep it stable.
    """
    tracker = EntityTracker(filter_enabled=True)
    telemetry = TelemetryInput(
        entity_id="dup_target", x=1.0, y=2.0, z=0.0, timestamp=500.0
    )

    tracker.process_telemetry(telemetry, dt=0.5)
    prediction = tracker.process_telemetry(telemetry, dt=0.5)

    assert np.isfinite(prediction.predicted_x)
    assert np.isfinite(prediction.predicted_y)
    assert np.isfinite(prediction.predicted_z)
