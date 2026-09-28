"""
Tests for the trajectory prediction evaluation harness.

These tests guard the integrity of the benchmark itself. An earlier version
compared a cleanly-fed baseline against a noisily-fed filter, which made the
filter look progressively worse as noise rose — the opposite of reality. The
tests below pin down the properties that a valid benchmark must have.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from evaluate import evaluate_clip
from mocap import DEFAULT_DATA_ROOT, load_clip, iter_clip_paths

_HAS_DATA = DEFAULT_DATA_ROOT.exists() and any(DEFAULT_DATA_ROOT.glob("*/*.bvh"))
requires_data = pytest.mark.skipif(
    not _HAS_DATA, reason="CMU MoCap data not downloaded"
)


def _first_walking_clip():
    for path in iter_clip_paths():
        if path.parent.name == "007":
            return load_clip(path)
    raise AssertionError("no walking clip found")


@requires_data
def test_filter_beats_baselines_on_clean_data() -> None:
    """Even with no sensor noise, filtering must not be worse than the baselines."""
    clip = _first_walking_clip()
    scores = evaluate_clip(clip, horizon=0.5, noise_std=0.0, seed=1)

    assert scores["filtered"][0] < scores["static"][0]
    assert scores["filtered"][0] <= scores["naive"][0] * 1.05


@requires_data
def test_filter_advantage_grows_with_noise() -> None:
    """
    The central property of a Kalman filter: as measurement noise rises, the
    filtered prediction must degrade far more slowly than raw extrapolation.

    This is the check that catches a benchmark which feeds the baselines cleaner
    data than the filter.
    """
    clip = _first_walking_clip()

    low = evaluate_clip(clip, horizon=0.5, noise_std=0.02, seed=3)
    high = evaluate_clip(clip, horizon=0.5, noise_std=0.20, seed=3)

    low_ratio = low["naive"][0] / low["filtered"][0]
    high_ratio = high["naive"][0] / high["filtered"][0]

    assert low_ratio > 2.0, (
        f"filter should be at least 2x better than naive at sigma=0.02, "
        f"got {low_ratio:.2f}x"
    )
    assert high_ratio > low_ratio * 0.8, (
        "filter advantage must not collapse as noise increases"
    )


@requires_data
def test_naive_baseline_degrades_with_noise() -> None:
    """
    Raw finite-difference extrapolation amplifies noise, so its error must rise
    with the noise level. If it does not, the baseline is not seeing the noise.
    """
    clip = _first_walking_clip()

    clean = evaluate_clip(clip, horizon=0.5, noise_std=0.0, seed=5)
    noisy = evaluate_clip(clip, horizon=0.5, noise_std=0.10, seed=5)

    assert noisy["naive"][0] > clean["naive"][0] * 5.0


@requires_data
def test_filter_survives_heavy_noise() -> None:
    """The filtered prediction must stay bounded even at extreme sensor noise."""
    clip = _first_walking_clip()
    scores = evaluate_clip(clip, horizon=0.5, noise_std=0.25, seed=9)

    ade, fde = scores["filtered"]
    assert np.isfinite(ade) and np.isfinite(fde)
    # A walking clip covers a few metres; runaway error would exceed this.
    assert ade < 10.0, f"filtered ADE {ade:.2f} m indicates divergence"


@requires_data
def test_evaluation_is_deterministic_for_a_fixed_seed() -> None:
    """Repeated evaluation with the same seed must yield identical scores."""
    clip = _first_walking_clip()
    first = evaluate_clip(clip, horizon=0.5, noise_std=0.05, seed=42)
    second = evaluate_clip(clip, horizon=0.5, noise_std=0.05, seed=42)
    assert first == second
