"""
Tests for prediction uncertainty (calibration and propagation).

A prediction is only useful if its stated confidence means something. These
tests check two properties that are easy to get wrong:

  * uncertainty must GROW with the prediction horizon
  * the reported uncertainty must be CALIBRATED against observed error; at the
    original settings it was 30-60x larger than reality, which made the
    confidence region meaningless
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from filter import ExtendedKalmanFilter3D, MeasurementModel, MotionModel
from mocap import DEFAULT_DATA_ROOT, iter_clip_paths, load_clip
from predictor import CHI2_95_3D
from schema import TelemetryInput
from tracker import EntityTracker

_HAS_DATA = DEFAULT_DATA_ROOT.exists() and any(DEFAULT_DATA_ROOT.glob("*/*.bvh"))
requires_data = pytest.mark.skipif(
    not _HAS_DATA, reason="CMU MoCap data not downloaded"
)


# --- covariance propagation -----------------------------------------------


def test_propagated_covariance_grows_with_horizon() -> None:
    """
    Position uncertainty must increase with the horizon.

    Reporting the present covariance for a long horizon understates the error by
    orders of magnitude, because position error integrates twice over time.
    """
    ekf = ExtendedKalmanFilter3D(
        measurement_noise_std=0.02,
        measurement_model=MeasurementModel.POSITION,
        motion_model=MotionModel.CONSTANT_ACCELERATION,
    )
    for _ in range(50):
        ekf.predict(dt=0.05)
        ekf.update(np.array([0.0, 0.0, 0.0]))

    near = ekf.propagate_position_covariance(0.1)[0, 0]
    mid = ekf.propagate_position_covariance(0.5)[0, 0]
    far = ekf.propagate_position_covariance(2.0)[0, 0]

    assert near < mid < far


def test_propagated_covariance_is_symmetric_positive_semidefinite() -> None:
    """A covariance must be symmetric, and its variances non-negative."""
    ekf = ExtendedKalmanFilter3D(measurement_model=MeasurementModel.POSITION)
    for _ in range(30):
        ekf.predict(dt=0.05)
        ekf.update(np.array([1.0, 2.0, 3.0]))

    covariance = ekf.propagate_position_covariance(0.5)

    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.all(np.diag(covariance) >= 0.0)
    assert np.linalg.eigvalsh(covariance).min() > -1e-12


def test_negative_horizon_is_rejected() -> None:
    ekf = ExtendedKalmanFilter3D()
    with pytest.raises(ValueError):
        ekf.propagate_position_covariance(-0.1)


# --- output shape ----------------------------------------------------------


def test_output_reports_uncertainty_fields() -> None:
    """The prediction output must carry a complete, self-consistent uncertainty."""
    tracker = EntityTracker(filter_enabled=True)
    for step in range(40):
        prediction = tracker.process_telemetry(
            TelemetryInput(
                entity_id="p",
                x=0.05 * step,
                y=0.0,
                z=0.0,
                timestamp=1000.0 + 0.05 * step,
            ),
            dt=0.5,
        )

    assert prediction.sigma_x >= 0.0
    assert prediction.sigma_y >= 0.0
    assert prediction.sigma_z >= 0.0
    assert len(prediction.position_covariance) == 9

    # position_std is the Euclidean norm of the per-axis sigmas.
    expected = float(
        np.sqrt(
            prediction.sigma_x**2
            + prediction.sigma_y**2
            + prediction.sigma_z**2
        )
    )
    assert prediction.position_std == pytest.approx(expected)

    # The 95% radius uses the 3D chi-square quantile, not the 1D value.
    assert prediction.confidence_radius_95 == pytest.approx(
        CHI2_95_3D * prediction.position_std
    )
    assert CHI2_95_3D > 1.96

    # The diagonal of the reported covariance must match the per-axis variances.
    covariance = np.array(prediction.position_covariance).reshape(3, 3)
    assert covariance[0, 0] == pytest.approx(prediction.sigma_x**2)
    assert covariance[1, 1] == pytest.approx(prediction.sigma_y**2)
    assert covariance[2, 2] == pytest.approx(prediction.sigma_z**2)


def test_unfiltered_output_reports_zero_uncertainty() -> None:
    """
    With filtering disabled no uncertainty is estimated; reporting zeros is
    honest, whereas inventing a value would not be.
    """
    tracker = EntityTracker(filter_enabled=False)
    prediction = tracker.process_telemetry(
        TelemetryInput(entity_id="p", x=1.0, y=0.0, z=0.0, timestamp=1000.0),
        dt=0.5,
    )

    assert prediction.position_std == 0.0
    assert prediction.confidence_radius_95 == 0.0
    assert prediction.motion_model == "unfiltered"


# --- calibration against real motion ---------------------------------------


def _measure_calibration(subject: str, horizon: float, horizon_cap: float = 1.0):
    """
    Returns (mean error, mean reported std, coverage) over a real clip.

    Defaults to horizons up to 1 s, where the constant-acceleration model is
    still a fair description of walking.
    """
    clip = load_clip(
        [p for p in iter_clip_paths() if p.parent.name == subject][0]
    )
    steps = int(round(horizon / clip.sample_dt))
    rng = np.random.default_rng(0)
    noisy = clip.positions + rng.normal(0.0, 0.02, clip.positions.shape)

    tracker = EntityTracker(filter_enabled=True)
    errors, stds, inside, total = [], [], 0, 0

    for index in range(60, min(800, clip.frame_count - steps)):
        out = tracker.process_telemetry(
            TelemetryInput(
                entity_id="cal",
                x=float(noisy[index, 0]),
                y=float(noisy[index, 1]),
                z=float(noisy[index, 2]),
                timestamp=float(index) * clip.sample_dt,
            ),
            dt=horizon,
        )
        predicted = np.array(
            [out.predicted_x, out.predicted_y, out.predicted_z]
        )
        error = float(np.linalg.norm(predicted - clip.positions[index + steps]))
        errors.append(error)
        stds.append(out.position_std)
        total += 1
        if error <= out.confidence_radius_95:
            inside += 1

    return float(np.mean(errors)), float(np.mean(stds)), 100.0 * inside / total


@requires_data
def test_uncertainty_is_calibrated_to_observed_error() -> None:
    """
    The reported standard deviation must be within a small factor of the error
    actually observed on real motion.

    This is the test that exposes an uncalibrated uncertainty. At the original
    process-noise settings the reported std was ~40x the real error, so this
    assertion failed by a wide margin.
    """
    error, reported, _ = _measure_calibration("007", 0.5)
    ratio = error / reported

    assert 0.25 < ratio < 2.5, (
        f"reported std {reported:.4f} m is not calibrated against the observed "
        f"error {error:.4f} m (ratio {ratio:.3f}; target order 1)"
    )


@requires_data
def test_uncertainty_is_not_absurdly_pessimistic() -> None:
    """
    A modest AVERAGE over-confidence is acceptable (coverage may legitimately
    fall below 95% when model error dominates), but the region must not be
    hundreds of times too large.
    """
    error, reported, _ = _measure_calibration("013", 0.5)

    assert reported < 10.0 * error, (
        f"reported std {reported:.4f} m dwarfs the observed error {error:.4f} m"
    )
