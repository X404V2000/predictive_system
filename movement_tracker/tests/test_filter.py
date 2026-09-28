"""
Unit Tests for 3D Extended Kalman Filter State Estimator.

Validates noise variance reduction across sequential spatial measurements.
"""

import numpy as np
import pytest
from filter import ExtendedKalmanFilter3D


def test_ekf_variance_reduction() -> None:
    """
    Ensures EKF state updates reduce position measurement noise variance.
    """
    ekf = ExtendedKalmanFilter3D(process_noise_std=0.01, measurement_noise_std=1.0)

    # True constant state along X axis
    true_x = 10.0
    noisy_measurements = [true_x + np.random.normal(0, 1.0) for _ in range(50)]

    filtered_outputs = []

    for meas in noisy_measurements:
        # Create full 9D observation vector with position X noise
        measurement_vector = np.array([meas, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        ekf.predict(dt=0.1)
        ekf.update(measurement_vector)
        filtered_outputs.append(float(ekf.x[0, 0]))

    # Calculate variance of raw input vs filtered output (ignoring initial burn-in steps)
    raw_variance = float(np.var(noisy_measurements[10:]))
    filtered_variance = float(np.var(filtered_outputs[10:]))

    # Verify filtered variance is lower than raw sensor noise
    assert filtered_variance < raw_variance
