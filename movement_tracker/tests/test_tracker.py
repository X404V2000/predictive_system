"""
Unit Tests for Multi-Entity Tracking Manager.

Validates state isolation between independent tracked objects.
"""

import sys
from pathlib import Path

# Resolve parent path to module lookup list
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import TelemetryInput
from tracker import EntityTracker


def test_multi_entity_isolation() -> None:
    """
    Verifies state estimations for Object A do not bleed into Object B.
    """
    tracker = EntityTracker(filter_enabled=True)

    # Telemetry for Object A moving along X axis
    obj_a = TelemetryInput(
        entity_id="drone_alpha",
        x=0.0,
        y=0.0,
        z=0.0,
        vx=10.0,
        vy=0.0,
        vz=0.0,
        timestamp=100.0,
    )

    # Telemetry for Object B moving along Y axis
    obj_b = TelemetryInput(
        entity_id="drone_beta",
        x=50.0,
        y=50.0,
        z=0.0,
        vx=0.0,
        vy=-20.0,
        vz=0.0,
        timestamp=100.0,
    )

    pred_a = tracker.process_telemetry(obj_a, dt=1.0)
    pred_b = tracker.process_telemetry(obj_b, dt=1.0)

    # Ensure separate EKF instances track distinct position matrices
    assert pred_a.entity_id == "drone_alpha"
    assert pred_b.entity_id == "drone_beta"
    assert pred_a.predicted_x != pred_b.predicted_x


def test_warmup_flag_clears_after_enough_samples() -> None:
    """
    The tracker must report when its estimate has not settled.

    The filter starts with a deliberately large covariance because one position
    sample says nothing about velocity, so early projections swing by metres.
    Callers need to know that, rather than being handed an authoritative-looking
    point that was computed before the estimate converged.
    """
    from tracker import WARMUP_SAMPLES, EntityTracker

    tracker = EntityTracker(filter_enabled=True)
    flags = []
    for step in range(WARMUP_SAMPLES + 5):
        result = tracker.process_telemetry(
            TelemetryInput(
                entity_id="w",
                x=1.2 * step * 0.05,
                y=0.9,
                z=0.0,
                timestamp=1000.0 + step * 0.05,
            ),
            dt=0.5,
        )
        flags.append(result.warming_up)

    # The estimate is settled once WARMUP_SAMPLES observations have arrived, so
    # the flag is set for the first (WARMUP_SAMPLES - 1) samples and clears on
    # the one that reaches the threshold.
    assert flags[0] is True, "the first sample cannot have a settled estimate"
    assert flags[WARMUP_SAMPLES - 2] is True
    assert flags[WARMUP_SAMPLES - 1] is False
    assert flags[-1] is False


def test_unfiltered_mode_is_not_reported_as_warming_up() -> None:
    """
    With filtering disabled there is no estimate to converge, so the flag must
    stay clear rather than reporting a transient that cannot occur.
    """
    from tracker import EntityTracker

    tracker = EntityTracker(filter_enabled=False)
    for step in range(3):
        result = tracker.process_telemetry(
            TelemetryInput(
                entity_id="w",
                x=float(step),
                y=1.0,
                z=0.0,
                timestamp=1000.0 + step,
            ),
            dt=0.5,
        )
        assert result.warming_up is False


def test_warmup_state_is_released_when_an_entity_is_pruned() -> None:
    from tracker import EntityTracker

    tracker = EntityTracker(ttl_seconds=0.0)
    tracker.process_telemetry(
        TelemetryInput(entity_id="gone", x=0.0, y=1.0, z=0.0, timestamp=1000.0),
        dt=0.5,
    )
    assert "gone" in tracker._samples_seen

    tracker.prune_stale_entities(current_time=tracker.last_seen["gone"] + 1e6)

    assert "gone" not in tracker._samples_seen
