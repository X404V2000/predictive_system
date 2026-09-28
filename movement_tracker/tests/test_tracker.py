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
