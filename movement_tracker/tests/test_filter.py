"""
Unit Tests for the 3D Kalman Filter State Estimator.

Validates noise reduction, the position-only measurement model, motion-model
switching, and numerical stability.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from filter import ExtendedKalmanFilter3D, MeasurementModel, MotionModel


def test_ekf_variance_reduction() -> None:
    """Filtered position must have lower variance than the raw measurements."""
    ekf = ExtendedKalmanFilter3D(
        process_noise_std=0.01,
        measurement_noise_std=1.0,
        measurement_model=MeasurementModel.POSITION,
    )

    rng = np.random.default_rng(0)
    true_x = 10.0
    noisy = true_x + rng.normal(0.0, 1.0, size=50)

    filtered = []
    for value in noisy:
        ekf.predict(dt=0.1)
        ekf.update(np.array([value, 0.0, 0.0]))
        filtered.append(float(ekf.position[0]))

    raw_variance = float(np.var(noisy[10:]))
    filtered_variance = float(np.var(filtered[10:]))
    assert filtered_variance < raw_variance


def test_position_model_rejects_wrong_measurement_size() -> None:
    """A 9-value measurement must be refused by the 3-value position model."""
    ekf = ExtendedKalmanFilter3D(measurement_model=MeasurementModel.POSITION)
    with pytest.raises(ValueError):
        ekf.update(np.zeros(9))


def test_full_model_accepts_nine_values() -> None:
    """The full-state model still accepts a 9-value observation."""
    ekf = ExtendedKalmanFilter3D(measurement_model=MeasurementModel.FULL)
    ekf.predict(dt=0.1)
    ekf.update(np.arange(9, dtype=np.float64))


def test_position_only_filter_infers_constant_velocity() -> None:
    """
    Observing position alone must be sufficient to recover a constant velocity.

    This is the core reason for the position-only model: a real position sensor
    never supplies velocity, so the filter must derive it.
    """
    ekf = ExtendedKalmanFilter3D(
        measurement_noise_std=0.02,
        measurement_model=MeasurementModel.POSITION,
        motion_model=MotionModel.CONSTANT_ACCELERATION,
    )

    dt = 0.1
    true_velocity = np.array([1.5, -0.8, 0.0])
    position = np.zeros(3)
    for _ in range(80):
        position = position + true_velocity * dt
        ekf.predict(dt=dt)
        ekf.update(position)

    assert np.allclose(ekf.velocity, true_velocity, atol=0.05)


def test_stationary_model_decays_velocity_to_zero() -> None:
    """
    Under the STATIONARY model, a spurious velocity must decay away.

    Without decay, a random walk in velocity lets the estimate drift away from a
    motionless target.
    """
    ekf = ExtendedKalmanFilter3D(
        measurement_model=MeasurementModel.POSITION,
        motion_model=MotionModel.STATIONARY,
    )
    # Seed a large phantom velocity.
    ekf.x[3, 0] = 5.0

    stationary = np.zeros(3)
    for _ in range(100):
        ekf.predict(dt=0.05)
        ekf.update(stationary)

    assert abs(ekf.velocity[0]) < 0.05


def test_covariance_stays_symmetric_and_positive_definite() -> None:
    """Rounding must not corrupt the covariance over many updates."""
    ekf = ExtendedKalmanFilter3D(measurement_noise_std=0.05)
    rng = np.random.default_rng(3)

    for _ in range(500):
        ekf.predict(dt=0.02)
        ekf.update(rng.normal(0.0, 0.05, size=3))

    assert np.allclose(ekf.P, ekf.P.T, atol=1e-9)
    eigenvalues = np.linalg.eigvalsh(ekf.P)
    assert eigenvalues.min() > 0.0


def test_motion_model_switch_changes_process_noise() -> None:
    """Switching model recomputes Q without discarding the state."""
    ekf = ExtendedKalmanFilter3D()
    ekf.x[0, 0] = 3.0

    noise_before = float(ekf.Q[0, 0])
    ekf.set_motion_model(MotionModel.STATIONARY)
    noise_after = float(ekf.Q[0, 0])

    assert noise_after < noise_before
    assert ekf.x[0, 0] == 3.0


def test_reset_clears_state() -> None:
    ekf = ExtendedKalmanFilter3D()
    ekf.x[0, 0] = 7.0
    ekf.reset()
    assert np.allclose(ekf.x, 0.0)
