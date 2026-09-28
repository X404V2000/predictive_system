"""
Accuracy Regression Tests for the Prediction Pipeline.

Unlike the variance-only smoke tests, these tests measure absolute prediction
error against known ground truth and assert that the filter actually *improves*
on the naive baseline. This is the class of test that catches integration-step
regressions (e.g. feeding a prediction horizon into the EKF's predict step),
which the tolerance-based tests in test_api.py silently accept.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from schema import TelemetryInput
from tracker import EntityTracker

# --- Ground-truth simulation parameters -----------------------------------
TRUE_VX = 10.0          # metres / second, constant velocity along X
TRUE_VY = -3.0          # metres / second, constant velocity along Y
SAMPLE_DT = 0.1         # telemetry arrives every 100 ms
HORIZON = 0.5           # prediction horizon in seconds
POSITION_NOISE = 0.5    # standard deviation of position measurement noise (m)
STEPS = 80
BURN_IN = 20            # discard filter warm-up samples


def _simulate(filter_enabled: bool, seed: int = 12345, jitter: float = 0.0):
    """
    Runs a deterministic constant-velocity scenario and returns the mean
    Euclidean prediction error (metres) over the post-burn-in samples.
    """
    rng = np.random.default_rng(seed)
    tracker = EntityTracker(filter_enabled=filter_enabled)

    timestamp = 1000.0
    x_true, y_true = 0.0, 0.0
    errors = []

    for _ in range(STEPS):
        x_true += TRUE_VX * SAMPLE_DT
        y_true += TRUE_VY * SAMPLE_DT
        timestamp += SAMPLE_DT + (rng.uniform(-jitter, jitter) if jitter else 0.0)

        telemetry = TelemetryInput(
            entity_id="accuracy_target",
            x=x_true + rng.normal(0.0, POSITION_NOISE),
            y=y_true + rng.normal(0.0, POSITION_NOISE),
            z=0.0,
            vx=TRUE_VX,
            vy=TRUE_VY,
            vz=0.0,
            timestamp=timestamp,
        )

        prediction = tracker.process_telemetry(telemetry, dt=HORIZON)

        # Ground truth position at the prediction horizon
        gt_x = x_true + TRUE_VX * HORIZON
        gt_y = y_true + TRUE_VY * HORIZON
        errors.append(
            float(np.hypot(prediction.predicted_x - gt_x, prediction.predicted_y - gt_y))
        )

    return float(np.mean(errors[BURN_IN:]))


def test_filter_improves_on_naive_baseline() -> None:
    """
    The filtered prediction must be MORE accurate than using raw telemetry.

    This is the central correctness guarantee of the whole pipeline. If the
    integration step is confused with the prediction horizon, the filter
    over-integrates and performs many times worse than the baseline — which is
    precisely the regression this test exists to prevent.
    """
    filtered_error = _simulate(filter_enabled=True)
    naive_error = _simulate(filter_enabled=False)

    assert filtered_error < naive_error, (
        f"Filtered error ({filtered_error:.4f} m) must be lower than the naive "
        f"baseline ({naive_error:.4f} m); the filter is degrading predictions."
    )


def test_filter_error_is_physically_plausible() -> None:
    """
    Absolute accuracy bound: in this clean constant-velocity scenario the
    filtered prediction should be accurate to well under half a metre.
    """
    filtered_error = _simulate(filter_enabled=True)
    assert filtered_error < 0.5, (
        f"Filtered prediction error {filtered_error:.4f} m exceeds the 0.5 m "
        f"plausibility bound for a clean constant-velocity target."
    )


def test_irregular_arrival_intervals_are_handled() -> None:
    """
    Timestamps jitter while the underlying motion stays constant. Because the
    filter derives its integration step from the actual timestamps, it must
    remain accurate despite the irregular arrival cadence.
    """
    jittered_error = _simulate(filter_enabled=True, jitter=0.03)
    assert jittered_error < 0.6, (
        f"Filtered error under timestamp jitter ({jittered_error:.4f} m) is "
        f"implausible; the filter may be assuming a fixed interval."
    )


def test_duplicate_timestamp_does_not_destabilise_filter() -> None:
    """
    Two telemetry frames sharing an identical timestamp give dt == 0, which
    would produce a singular transition model if unguarded. The tracker's
    minimum-interval clamp must keep the filter finite and stable.
    """
    tracker = EntityTracker(filter_enabled=True)
    telemetry = TelemetryInput(
        entity_id="dup_target", x=1.0, y=2.0, z=0.0, vx=1.0, timestamp=500.0
    )

    tracker.process_telemetry(telemetry, dt=0.5)
    prediction = tracker.process_telemetry(telemetry, dt=0.5)  # same timestamp

    assert np.isfinite(prediction.predicted_x)
    assert np.isfinite(prediction.predicted_y)
    assert np.isfinite(prediction.predicted_z)
