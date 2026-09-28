"""
Lifecycle Tests for Multi-Entity Tracking Manager.

Validates stale-entity eviction, the pruning throttle, and full cleanup of the
per-entity state registries (filters, predictors, timestamps).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import TelemetryInput
from tracker import EntityTracker


def _telemetry(entity_id: str, timestamp: float = 100.0) -> TelemetryInput:
    return TelemetryInput(
        entity_id=entity_id, x=0.0, y=0.0, z=0.0, vx=1.0, timestamp=timestamp
    )


def test_manual_prune_removes_stale_entity_completely() -> None:
    """
    An entity whose last activity predates the TTL must be removed from every
    registry tracking it, leaving no orphaned state behind.
    """
    tracker = EntityTracker(ttl_seconds=10.0)
    tracker.process_telemetry(_telemetry("alpha"))

    assert "alpha" in tracker.filters
    assert "alpha" in tracker.predictors
    assert "alpha" in tracker.last_seen
    assert "alpha" in tracker.last_measurement_ts

    # Sweep at a wall-clock time far beyond the TTL window
    removed = tracker.prune_stale_entities(current_time=tracker.last_seen["alpha"] + 1e6)

    assert removed == 1
    assert "alpha" not in tracker.filters
    assert "alpha" not in tracker.predictors
    assert "alpha" not in tracker.last_seen
    assert "alpha" not in tracker.last_measurement_ts


def test_prune_preserves_recent_entities() -> None:
    """
    Active entities must survive a pruning sweep; only expired ones are evicted.
    """
    tracker = EntityTracker(ttl_seconds=10.0)
    tracker.process_telemetry(_telemetry("alpha"))
    tracker.process_telemetry(_telemetry("beta"))
    fresh_time = tracker.last_seen["beta"]

    removed = tracker.prune_stale_entities(current_time=fresh_time)

    assert removed == 0
    assert set(tracker.filters) == {"alpha", "beta"}


def test_auto_prune_evicts_entities_growing_without_bound() -> None:
    """
    Feeding many short-lived entity IDs must not grow memory without bound:
    the automatic sweep inside process_telemetry eventually reclaims them.
    """
    tracker = EntityTracker(ttl_seconds=0.0, prune_interval_seconds=0.0)

    for index in range(50):
        tracker.process_telemetry(_telemetry(f"target_{index}"))

    # With ttl=0 and pruning enabled on every call, only the current entity
    # (registered after the sweep) may remain; prior ones must be evicted.
    assert len(tracker.filters) <= 1
    assert len(tracker.predictors) == len(tracker.filters)
    assert len(tracker.last_measurement_ts) == len(tracker.filters)


def test_prune_throttle_respects_interval() -> None:
    """
    The automatic sweep is throttled: repeated calls within the prune interval
    must not re-run eviction.
    """
    tracker = EntityTracker(ttl_seconds=0.0, prune_interval_seconds=10.0)
    tracker.process_telemetry(_telemetry("alpha"))

    # Anchor sweep times to the entity's real activity clock so the staleness
    # comparison is meaningful.
    base = tracker.last_seen["alpha"]

    # Within the throttle window the sweep must be skipped, leaving state intact.
    assert tracker._maybe_prune(now=base + 0.5) == 0
    assert "alpha" in tracker.filters

    # Beyond the window the sweep runs and evicts the expired entity.
    assert tracker._maybe_prune(now=base + 60.0) == 1
    assert "alpha" not in tracker.filters
