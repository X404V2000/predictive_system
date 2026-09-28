"""
Shared Kinematic Primitives for Motion Analysis.

Pure, dependency-light helpers for turning position observations into the
derived quantities (velocity, acceleration, speed, turning rate) that both the
physical prediction layer and the behavioural feature layer consume.

Keeping these in one place means the dataset loader, the validation harness and
the live pipeline all agree on how motion is derived from raw positions.
"""

import numpy as np


def _central_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """
    Centred finite difference along axis 0, with one-sided differences at the
    boundaries so the output has the same length as the input.
    """
    if values.shape[0] < 3:
        raise ValueError("Need at least 3 samples for a central difference.")

    derivative = np.empty_like(values, dtype=np.float64)
    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    # One-sided first-order estimates at the endpoints.
    derivative[0] = (values[1] - values[0]) / dt
    derivative[-1] = (values[-1] - values[-2]) / dt
    return derivative


def derive_motion(
    positions: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derives velocity and acceleration from a position series.

    Acceleration is the finite difference of the derived velocity, so it is
    inherently noisier than the position signal — the same situation the Kalman
    filter faces with real telemetry.

    Returns (velocities, accelerations), each shaped like `positions`.
    """
    velocities = _central_difference(positions, dt)
    accelerations = _central_difference(velocities, dt)
    return velocities, accelerations


def speed(velocities: np.ndarray) -> np.ndarray:
    """Euclidean speed magnitude per sample."""
    return np.linalg.norm(velocities, axis=-1)


def smooth_positions(positions: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Moving-average smoothing over the time axis.

    Motion-capture exports carry marker jitter; smoothing before differencing
    prevents that jitter from being amplified into acceleration noise.
    """
    if window <= 1:
        return positions
    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd so it stays centred.")

    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(positions, ((pad, pad), (0, 0)), mode="edge")

    smoothed = np.empty_like(positions, dtype=np.float64)
    for axis in range(positions.shape[1]):
        smoothed[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    return smoothed


def yaw_rate(velocities: np.ndarray, dt: float) -> np.ndarray:
    """
    Heading angular rate about the vertical (Z) axis, in rad/s.

    Defined only where horizontal speed is non-negligible; elsewhere it is zero,
    matching the assumption the coordinated-turn predictor makes.
    """
    vx, vy = velocities[:, 0], velocities[:, 1]
    horizontal_speed = np.hypot(vx, vy)

    heading = np.arctan2(vy, vx)
    heading_unwrapped = np.unwrap(heading)
    rate = _central_difference(heading_unwrapped.reshape(-1, 1), dt).ravel()

    # Suppress meaningless heading changes when the target is barely moving.
    rate[horizontal_speed < 1e-3] = 0.0
    return rate


__all__ = [
    "derive_motion",
    "speed",
    "smooth_positions",
    "yaw_rate",
]
