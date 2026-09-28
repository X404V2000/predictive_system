"""
Tests for 3D turning motion (coordinated-turn projection).

Three problems motivated these tests, each found by measuring rather than
assuming:

  1. The turn model was DEAD CODE. The tracker forwarded the client-supplied
     `yaw_rate`, which defaults to zero, so the coordinated-turn branch was
     never reached on real telemetry.
  2. The turn was applied in the X-Y plane while the BVH data is Y-up, so the
     heading was measured in a plane containing almost no motion. That reported
     3290 deg/s of turning for a straight walk.
  3. Enabling turning naively made running 40% worse, because fast motion
     saturated the rate cap and injected spurious arcs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from schema import TelemetryInput
from tracker import (
    HEADING_MIN_SPEED,
    MAX_YAW_RATE,
    MIN_TURN_RADIUS,
    EntityTracker,
)


def _stream_circle(
    tracker: EntityTracker,
    radius: float,
    omega: float,
    sample_dt: float = 0.05,
    steps: int = 400,
    entity: str = "turner",
) -> None:
    """Streams a target walking a circle in the X-Z ground plane (Y-up)."""
    speed = radius * omega
    position = np.array([radius, 0.9, 0.0])
    heading = np.pi  # start heading in -X
    timestamp = 1000.0

    for _ in range(steps):
        velocity = np.array([0.0, 0.0, 0.0])
        velocity[0] = speed * np.cos(heading)
        velocity[2] = speed * np.sin(heading)
        position = position + velocity * sample_dt
        heading += omega * sample_dt
        timestamp += sample_dt

        tracker.process_telemetry(
            TelemetryInput(
                entity_id=entity,
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                timestamp=timestamp,
            ),
            dt=0.5,
        )


def test_turn_rate_is_estimated_from_filtered_velocity() -> None:
    """
    Turning must be detected without the client supplying yaw_rate.

    Relying on the wire value left the coordinated-turn projection permanently
    unused on real telemetry, since the field defaults to zero.
    """
    tracker = EntityTracker(filter_enabled=True)
    _stream_circle(tracker, radius=8.0, omega=0.4)

    estimated = tracker._estimate_yaw_rate(
        "turner", tracker.filters["turner"], 0.05
    )
    assert estimated > 0.1, "a circling target must register a non-zero turn rate"


def test_straight_motion_reports_no_turning() -> None:
    """
    A straight walker must not register turning.

    The lateral velocity of someone walking straight is tiny compared with the
    forward velocity, so naive frame differencing invents large turns.
    """
    tracker = EntityTracker(filter_enabled=True)
    position = np.zeros(3)
    timestamp = 1000.0
    for _ in range(400):
        position[2] += 1.2 * 0.05  # walking along +Z at 1.2 m/s
        timestamp += 0.05
        tracker.process_telemetry(
            TelemetryInput(
                entity_id="walker",
                x=float(position[0]),
                y=0.9,
                z=float(position[2]),
                timestamp=timestamp,
            ),
            dt=0.5,
        )

    estimated = tracker._estimate_yaw_rate(
        "walker", tracker.filters["walker"], 0.05
    )
    assert abs(estimated) < 0.1


def test_stationary_target_reports_no_turning() -> None:
    """Heading is undefined without locomotion, so no turn may be reported."""
    tracker = EntityTracker(filter_enabled=True)
    timestamp = 1000.0
    for _ in range(200):
        timestamp += 0.05
        tracker.process_telemetry(
            TelemetryInput(
                entity_id="sitter", x=0.3, y=0.9, z=0.4, timestamp=timestamp
            ),
            dt=0.5,
        )

    estimated = tracker._estimate_yaw_rate(
        "sitter", tracker.filters["sitter"], 0.05
    )
    assert estimated == 0.0


def test_turn_rate_is_capped() -> None:
    """A reported turn rate must never exceed what a body could produce."""
    tracker = EntityTracker(filter_enabled=True)
    # A very tight, very fast circle drives the raw estimate far beyond the cap.
    _stream_circle(tracker, radius=0.5, omega=8.0, steps=200)

    estimated = tracker._estimate_yaw_rate(
        "turner", tracker.filters["turner"], 0.05
    )
    assert abs(estimated) <= MAX_YAW_RATE


def test_tight_turn_radius_is_rejected_as_artefact() -> None:
    """
    A turn radius below MIN_TURN_RADIUS is not a trajectory a body can follow and
    must be rejected rather than projected as an arc.
    """
    tracker = EntityTracker(filter_enabled=True)
    _stream_circle(tracker, radius=0.4, omega=5.0, steps=200)

    estimated = tracker._estimate_yaw_rate(
        "turner", tracker.filters["turner"], 0.05
    )
    assert estimated == 0.0


def test_vertical_axis_configuration_is_validated() -> None:
    with pytest.raises(ValueError):
        EntityTracker(vertical_axis=5)


def test_heading_state_is_cleaned_up_on_prune() -> None:
    """Pruning must release the velocity history used for turn estimation."""
    tracker = EntityTracker(ttl_seconds=0.0)
    _stream_circle(tracker, radius=8.0, omega=0.4, steps=40)

    tracker.prune_stale_entities(current_time=tracker.last_seen["turner"] + 1e6)

    assert "turner" not in tracker.filters
    assert "turner" not in tracker._velocity_history
