"""
Tests for multi-step trajectory prediction.

A trajectory returns the whole future path rather than one point, so callers can
draw where a target is going and how sharply that expectation blurs. These tests
pin down the properties that make it trustworthy:

  * the final point agrees exactly with the single-point output
  * uncertainty grows monotonically and is evaluated per step, not interpolated
  * the path is a true arc when turning, not a straight chord
  * the stated confidence interval is calibrated on real motion, including for a
    seated target whose posture still drifts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from filter import MotionModel
from mocap import DEFAULT_DATA_ROOT, iter_clip_paths, load_clip
from predictor import MAX_TRAJECTORY_STEPS, KinematicPredictor
from schema import TelemetryInput
from tracker import EntityTracker

_HAS_DATA = DEFAULT_DATA_ROOT.exists() and any(DEFAULT_DATA_ROOT.glob("*/*.bvh"))
requires_data = pytest.mark.skipif(
    not _HAS_DATA, reason="CMU MoCap data not downloaded"
)


def _converged_tracker(steps: int = 60) -> EntityTracker:
    tracker = EntityTracker(filter_enabled=True)
    for step in range(steps):
        tracker.process_telemetry(
            TelemetryInput(
                entity_id="t",
                x=1.2 * step * 0.05,
                y=0.9,
                z=0.0,
                timestamp=1000.0 + step * 0.05,
            ),
            dt=1.0,
        )
    return tracker


# --- structure -------------------------------------------------------------


def test_trajectory_is_absent_unless_requested() -> None:
    """The default response must remain a single point."""
    tracker = _converged_tracker()
    out = tracker.process_telemetry(
        TelemetryInput(entity_id="t", x=1.2, y=0.9, z=0.0, timestamp=1004.0),
        dt=1.0,
    )
    assert out.trajectory is None


def test_trajectory_spans_the_horizon_inclusively() -> None:
    """Sampling must start at now and end exactly at the horizon."""
    tracker = _converged_tracker()
    out = tracker.process_telemetry(
        TelemetryInput(entity_id="t", x=1.2, y=0.9, z=0.0, timestamp=1004.0),
        dt=1.0,
        trajectory=True,
        trajectory_step=0.25,
    )

    horizons = [round(p.horizon_seconds, 6) for p in out.trajectory]
    assert horizons == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_final_point_matches_single_point_fields() -> None:
    """
    The last trajectory sample and the scalar output must describe the same
    prediction; if they disagree, one of them is wrong.
    """
    tracker = _converged_tracker()
    out = tracker.process_telemetry(
        TelemetryInput(entity_id="t", x=1.2, y=0.9, z=0.0, timestamp=1004.0),
        dt=1.0,
        trajectory=True,
        trajectory_step=0.2,
    )

    last = out.trajectory[-1]
    assert last.x == pytest.approx(out.predicted_x, abs=1e-12)
    assert last.y == pytest.approx(out.predicted_y, abs=1e-12)
    assert last.z == pytest.approx(out.predicted_z, abs=1e-12)
    assert last.position_std == pytest.approx(out.position_std, abs=1e-12)
    assert last.confidence_radius_95 == pytest.approx(
        out.confidence_radius_95, abs=1e-12
    )


def test_uncertainty_grows_monotonically_along_the_path() -> None:
    """
    Confidence must decay with the horizon, never improve.

    Position error integrates twice over time, so uncertainty is not linear in
    the horizon; sampling it per step is what captures that.
    """
    tracker = _converged_tracker()
    out = tracker.process_telemetry(
        TelemetryInput(entity_id="t", x=1.2, y=0.9, z=0.0, timestamp=1004.0),
        dt=2.0,
        trajectory=True,
        trajectory_step=0.25,
    )

    stds = [p.position_std for p in out.trajectory]
    assert all(a <= b + 1e-12 for a, b in zip(stds, stds[1:]))
    assert stds[-1] > stds[0]


def test_trajectory_rejects_absurd_point_count() -> None:
    """A request for an enormous number of samples must be refused."""
    predictor = KinematicPredictor()
    with pytest.raises(ValueError):
        predictor.predict_trajectory(
            entity_id="t",
            position=np.zeros(3),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw_rate=0.0,
            timestamp=0.0,
            max_horizon=200.0,
            step=0.01,
            covariance_fn=lambda h: np.zeros((3, 3)),
        )


def test_trajectory_rejects_non_positive_arguments() -> None:
    predictor = KinematicPredictor()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            predictor.predict_trajectory(
                entity_id="t",
                position=np.zeros(3),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw_rate=0.0,
                timestamp=0.0,
                max_horizon=bad,
                step=0.1,
                covariance_fn=lambda h: np.zeros((3, 3)),
            )
        with pytest.raises(ValueError):
            predictor.predict_trajectory(
                entity_id="t",
                position=np.zeros(3),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw_rate=0.0,
                timestamp=0.0,
                max_horizon=1.0,
                step=bad,
                covariance_fn=lambda h: np.zeros((3, 3)),
            )


def test_max_trajectory_steps_is_enforced() -> None:
    """
    The documented cap must bound the returned point count.

    The count is inclusive of both endpoints, so a span divided evenly by the
    step yields one MORE point than the naive quotient.
    """
    predictor = KinematicPredictor()

    # Exactly at the cap: count = floor(span/step) + 1, so (MAX-1) steps lands
    # precisely on MAX points.
    at_cap = predictor.predict_trajectory(
        entity_id="t",
        position=np.zeros(3),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        yaw_rate=0.0,
        timestamp=0.0,
        max_horizon=(MAX_TRAJECTORY_STEPS - 1) * 0.05,
        step=0.05,
        covariance_fn=lambda h: np.zeros((3, 3)),
    )
    assert len(at_cap) == MAX_TRAJECTORY_STEPS

    # One point beyond the cap must be refused.
    with pytest.raises(ValueError):
        predictor.predict_trajectory(
            entity_id="t",
            position=np.zeros(3),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw_rate=0.0,
            timestamp=0.0,
            max_horizon=MAX_TRAJECTORY_STEPS * 0.05,
            step=0.05,
            covariance_fn=lambda h: np.zeros((3, 3)),
        )


# --- geometry --------------------------------------------------------------


def test_turning_trajectory_follows_the_arc_not_a_chord() -> None:
    """
    Each sample must lie on the turning circle.

    The path is projected from the original state rather than chained step to
    step, so every point should sit exactly on the arc at its own horizon.
    """
    predictor = KinematicPredictor()
    speed = 2.0
    omega = 0.5
    radius = speed / omega

    points = predictor.predict_trajectory(
        entity_id="t",
        position=np.array([0.0, 0.9, 0.0]),
        velocity=np.array([0.0, 0.0, speed]),
        acceleration=np.zeros(3),
        yaw_rate=omega,
        timestamp=0.0,
        max_horizon=2.0,
        step=0.25,
        covariance_fn=lambda h: np.zeros((3, 3)),
    )

    for point in points:
        distance_from_centre = np.hypot(point.x + radius, point.z)
        assert distance_from_centre == pytest.approx(radius, abs=1e-9)
        # Height is unaffected by a ground-plane turn.
        assert point.y == pytest.approx(0.9, abs=1e-12)


def test_straight_trajectory_is_collinear() -> None:
    """Without turning, successive samples must lie on one straight line."""
    predictor = KinematicPredictor()
    points = predictor.predict_trajectory(
        entity_id="t",
        position=np.zeros(3),
        velocity=np.array([0.0, 0.0, 1.5]),
        acceleration=np.zeros(3),
        yaw_rate=0.0,
        timestamp=0.0,
        max_horizon=1.0,
        step=0.25,
        covariance_fn=lambda h: np.zeros((3, 3)),
    )

    for point in points:
        assert point.z == pytest.approx(1.5 * point.horizon_seconds)
        assert point.x == pytest.approx(0.0)


# --- calibration -----------------------------------------------------------


@requires_data
def test_seated_trajectory_interval_is_calibrated() -> None:
    """
    A seated target's 1 s confidence interval must cover the truth about 95% of
    the time.

    This is the check that exposed the stationary model assuming zero motion. A
    seated person's hip still drifts ~0.15 m/s while shifting posture, so without
    a postural-drift term the interval covered only 58% of outcomes.
    """
    clip = load_clip([p for p in iter_clip_paths() if p.parent.name == "013"][0])
    rng = np.random.default_rng(0)
    noisy = clip.positions + rng.normal(0.0, 0.02, clip.positions.shape)
    tracker = EntityTracker(filter_enabled=True)

    horizon = 1.0
    steps = int(round(horizon / clip.sample_dt))
    inside = 0
    total = 0

    for index in range(80, clip.frame_count - steps, 5):
        out = tracker.process_telemetry(
            TelemetryInput(
                entity_id="s",
                x=float(noisy[index, 0]),
                y=float(noisy[index, 1]),
                z=float(noisy[index, 2]),
                timestamp=float(index) * clip.sample_dt,
            ),
            dt=horizon,
        )
        truth = clip.positions[index + steps]
        error = float(
            np.linalg.norm(
                np.array([out.predicted_x, out.predicted_y, out.predicted_z]) - truth
            )
        )
        total += 1
        if error <= out.confidence_radius_95:
            inside += 1

    coverage = inside / total
    assert coverage > 0.85, (
        f"seated 1 s coverage is {coverage:.1%}; the interval is over-confident "
        f"(postural drift is not being modelled)"
    )
