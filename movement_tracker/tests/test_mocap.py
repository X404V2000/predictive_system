"""
Tests for the CMU MoCap (BVH) loader and kinematic primitives.

These tests validate physical plausibility, not just that code runs: a walking
clip must produce human walking speeds, running must exceed walking, and a
stationary clip must stay near zero. This is what caught the earlier dataset's
corruption, where a "sitting" clip appeared to travel 25 metres.

Tests that need the dataset are skipped automatically when it is absent, so the
suite still passes on a machine without the motion-capture download.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from mocap import DEFAULT_DATA_ROOT, iter_clip_paths, load_clip, load_dataset
from physics import derive_motion, smooth_positions, speed, yaw_rate

_HAS_DATA = DEFAULT_DATA_ROOT.exists() and any(DEFAULT_DATA_ROOT.glob("*/*.bvh"))
requires_data = pytest.mark.skipif(
    not _HAS_DATA, reason="CMU MoCap data not downloaded"
)


# --- Kinematic primitives (no data required) ------------------------------


def test_derive_motion_matches_analytic_constant_velocity() -> None:
    """Constant velocity must yield that exact velocity and zero acceleration."""
    dt = 0.1
    frames = 50
    velocity = np.array([2.0, -1.0, 0.5])
    positions = np.outer(np.arange(frames), velocity * dt)

    vel, acc = derive_motion(positions, dt)
    assert np.allclose(vel, velocity, atol=1e-9)
    assert np.allclose(acc, 0.0, atol=1e-9)


def test_derive_motion_matches_analytic_constant_acceleration() -> None:
    """Constant acceleration must be recovered by double differencing."""
    dt = 0.05
    frames = 60
    accel = np.array([0.0, 0.0, -9.81])
    t = np.arange(frames) * dt
    positions = 0.5 * np.outer(t**2, accel)

    _, acc = derive_motion(positions, dt)
    # Interior samples are exact; endpoints use one-sided differences.
    assert np.allclose(acc[2:-2], accel, atol=1e-6)


def test_smoothing_reduces_noise_without_biasing_trend() -> None:
    """Smoothing must lower jitter while preserving the underlying trend."""
    rng = np.random.default_rng(7)
    frames = 200
    trend = np.linspace(0.0, 5.0, frames).reshape(-1, 1) * np.array([[1.0, 0.0, 0.0]])
    noisy = trend + rng.normal(0.0, 0.05, size=trend.shape)

    smoothed = smooth_positions(noisy, window=9)

    noise_before = float(np.std(noisy - trend))
    noise_after = float(np.std(smoothed - trend))
    assert noise_after < noise_before
    # The trend must survive: endpoints stay close to the true trajectory.
    assert np.allclose(smoothed[-1], trend[-1], atol=0.05)


def test_speed_is_euclidean_magnitude() -> None:
    velocities = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    assert np.allclose(speed(velocities), [5.0, 2.0])


def test_yaw_rate_zero_when_stationary() -> None:
    """A stationary target must report no heading change."""
    velocities = np.zeros((20, 3))
    rate = yaw_rate(velocities, dt=0.1)
    assert np.allclose(rate, 0.0)


def test_yaw_rate_positive_for_counterclockwise_turn() -> None:
    """A left-hand (counterclockwise) turn must produce a positive rate."""
    dt = 0.01
    frames = 200
    radius = 5.0
    omega_true = 0.5
    t = np.arange(frames) * dt
    # Circular motion in the X-Y plane, counterclockwise.
    vx = -radius * omega_true * np.sin(omega_true * t)
    vy = radius * omega_true * np.cos(omega_true * t)
    velocities = np.stack([vx, vy, np.zeros_like(vx)], axis=1)

    rate = yaw_rate(velocities, dt)
    assert np.isclose(float(np.median(rate[10:-10])), omega_true, atol=0.05)


# --- BVH loading (requires the dataset) -----------------------------------


@requires_data
def test_bvh_parse_reports_consistent_frame_metadata() -> None:
    """Declared frame count and frame time must be internally consistent."""
    clip = load_clip(next(iter_clip_paths()))
    assert clip.frame_count > 0
    assert clip.positions.shape == (clip.frame_count, 3)
    assert clip.sample_dt > 0
    assert clip.fps == pytest.approx(1.0 / clip.sample_dt)


@requires_data
def test_walk_speed_is_human_plausible() -> None:
    """
    Walking clips must yield human walking speeds (~1-1.5 m/s).

    This is the physical validation that a mis-scaled or corrupted dataset
    fails: an incorrect unit conversion multiplies every speed by a constant
    factor and lands outside this band.
    """
    walking = load_dataset(subjects=("007",))
    assert walking, "expected walking clips in subject 007"

    for clip in walking:
        mean_speed = float(np.linalg.norm(clip.velocities, axis=1).mean())
        assert 0.3 < mean_speed < 2.0, (
            f"{clip.subject}: walking speed {mean_speed:.2f} m/s is not "
            "physically plausible for a human"
        )


@requires_data
def test_running_is_faster_than_walking() -> None:
    """The kinematic ordering of actions must hold in the data."""
    walk_speeds = [
        float(np.linalg.norm(c.velocities, axis=1).mean())
        for c in load_dataset(subjects=("007",))
    ]
    run_speeds = [
        float(np.linalg.norm(c.velocities, axis=1).mean())
        for c in load_dataset(subjects=("009",))
    ]
    assert run_speeds and walk_speeds
    assert min(run_speeds) > max(walk_speeds), (
        f"run ({min(run_speeds):.2f}) must exceed walk ({max(walk_speeds):.2f})"
    )


@requires_data
def test_stationary_clip_does_not_travel() -> None:
    """
    A seated/static clip must not show large translation.

    The earlier Human3.6M export failed exactly this check, reporting up to 25 m
    of travel for a seated subject.
    """
    static = load_dataset(subjects=("013",))
    assert static, "expected a seated clip in subject 013"

    for clip in static:
        travel = float(np.linalg.norm(clip.positions[-1] - clip.positions[0]))
        assert travel < 2.0, (
            f"{clip.subject}: seated clip travelled {travel:.1f} m; the root "
            "trajectory is unreliable"
        )


@requires_data
def test_hip_height_is_human_plausible() -> None:
    """Standing hip height must fall in a realistic band (~0.7-1.2 m)."""
    for clip in load_dataset(limit=6):
        hip_height = float(clip.positions[:, 1].mean())
        assert 0.5 < hip_height < 1.3, (
            f"{clip.subject}: mean hip height {hip_height:.2f} m is implausible; "
            "the unit scale is likely wrong"
        )
