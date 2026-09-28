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


def yaw_rate(
    velocities: np.ndarray,
    dt: float,
    vertical_axis: int = 1,
    min_horizontal_speed: float = 0.5,
    smoothing_seconds: float = 0.2,
) -> np.ndarray:
    """
    Heading angular rate about the vertical axis, in rad/s.

    `vertical_axis` selects which coordinate is up. It defaults to 1 (Y-up), the
    convention used by the CMU BVH exports: the root's Y channel sits at ~0.85 m
    (hip height) and barely varies, while Z carries the walking direction. Using
    X-Y as the ground plane on Y-up data measures heading in a plane containing
    almost no motion, and reports thousands of degrees per second for a straight
    walk.

    Heading is computed from velocity SMOOTHED over `smoothing_seconds`. A person
    walking roughly straight has lateral velocity an order of magnitude smaller
    than forward velocity, so frame-to-frame heading is almost entirely
    measurement noise: on a straight 120 Hz walk, raw heading differencing gave
    122 deg/s where 18 deg/s is correct. Smoothing recovers the real heading.

    Rate is reported as zero while the target is barely moving, where heading is
    undefined and would otherwise be pure noise amplification. The gate is
    deliberately at locomotion speed (0.5 m/s) rather than near zero: a seated
    person still shifts posture at ~0.15 m/s, and computing heading from that
    reported 53 deg/s of "turning" where the truth is none. Gating at 0.5 m/s
    leaves all genuine locomotion untouched (walking is ~1.2 m/s) while reducing
    the seated case to ~0.
    """
    if vertical_axis not in (0, 1, 2):
        raise ValueError(f"vertical_axis must be 0, 1 or 2, got {vertical_axis}")

    ground_axes = [axis for axis in (0, 1, 2) if axis != vertical_axis]
    forward = velocities[:, ground_axes[0]]
    lateral = velocities[:, ground_axes[1]]

    window = max(1, int(round(smoothing_seconds / dt)))
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / window
        forward = np.convolve(forward, kernel, mode="same")
        lateral = np.convolve(lateral, kernel, mode="same")

    horizontal_speed = np.hypot(forward, lateral)

    heading = np.arctan2(lateral, forward)
    heading_unwrapped = np.unwrap(heading)
    rate = _central_difference(heading_unwrapped.reshape(-1, 1), dt).ravel()

    # Suppress meaningless heading changes when the target is barely moving.
    rate = np.where(horizontal_speed < min_horizontal_speed, 0.0, rate)
    return rate


def turn_rate_magnitude(velocities: np.ndarray, dt: float, vertical_axis: int = 1) -> float:
    """Mean absolute heading rate, a compact summary of how much a clip turns."""
    return float(np.abs(yaw_rate(velocities, dt, vertical_axis)).mean())


__all__ = [
    "derive_motion",
    "speed",
    "smooth_positions",
    "yaw_rate",
    "turn_rate_magnitude",
]
