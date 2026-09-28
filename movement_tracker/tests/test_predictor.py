"""
Unit Tests for Kinematic Prediction Engine.

Validates pure Newtonian motion (CA) and turning motion (CT) calculations.
"""

import sys
from pathlib import Path

# Resolve parent directory to module lookup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from predictor import KinematicPredictor
from schema import TelemetryInput


@pytest.fixture
def predictor() -> KinematicPredictor:
    """
    Instantiates a clean KinematicPredictor instance for direct math validation.
    """
    return KinematicPredictor()


def test_linear_velocity_motion(predictor: KinematicPredictor) -> None:
    """
    Validates position displacement driven strictly by constant linear velocity (zero acceleration).
    """
    telemetry = TelemetryInput(
        entity_id="test_unit",
        x=0.0,
        y=0.0,
        z=0.0,
        vx=10.0,
        vy=0.0,
        vz=0.0,
        ax=0.0,
        ay=0.0,
        az=0.0,
        timestamp=1000.0,
    )

    # Directly execute predictor math computation
    result = predictor.predict_state(
        entity_id=telemetry.entity_id,
        x=telemetry.x,
        y=telemetry.y,
        z=telemetry.z,
        vx=telemetry.vx,
        vy=telemetry.vy,
        vz=telemetry.vz,
        ax=telemetry.ax,
        ay=telemetry.ay,
        az=telemetry.az,
        yaw_rate=telemetry.yaw_rate,
        timestamp=telemetry.timestamp,
        dt=2.0,
    )

    # Verify linear displacement (0 + 10 * 2 = 20)
    assert result.predicted_x == 20.0
    assert result.predicted_y == 0.0
    assert result.predicted_z == 0.0
    assert result.target_timestamp == 1002.0


def test_accelerated_3d_motion(predictor: KinematicPredictor) -> None:
    """
    Validates position displacement combining position, velocity, and acceleration across 3 dimensions.
    """
    telemetry = TelemetryInput(
        entity_id="test_unit",
        x=5.0,
        y=10.0,
        z=-2.0,
        vx=2.0,
        vy=-1.0,
        vz=0.0,
        ax=1.0,
        ay=2.0,
        az=-9.81,
        timestamp=500.0,
    )

    result = predictor.predict_state(
        entity_id=telemetry.entity_id,
        x=telemetry.x,
        y=telemetry.y,
        z=telemetry.z,
        vx=telemetry.vx,
        vy=telemetry.vy,
        vz=telemetry.vz,
        ax=telemetry.ax,
        ay=telemetry.ay,
        az=telemetry.az,
        yaw_rate=telemetry.yaw_rate,
        timestamp=telemetry.timestamp,
        dt=1.0,
    )

    # X calculation: 5.0 + (2.0 * 1) + (0.5 * 1.0 * 1^2) = 7.5
    assert pytest.approx(result.predicted_x, abs=1e-5) == 7.5
    # Y calculation: 10.0 + (-1.0 * 1) + (0.5 * 2.0 * 1^2) = 10.0
    assert pytest.approx(result.predicted_y, abs=1e-5) == 10.0
    # Z calculation: -2.0 + (0.0 * 1) + (0.5 * -9.81 * 1^2) = -6.905
    assert pytest.approx(result.predicted_z, abs=1e-5) == -6.905


def test_coordinated_turn_traces_a_circular_arc(predictor: KinematicPredictor) -> None:
    """
    A turn must follow a circular arc in the ground plane, at constant radius.

    The default vertical axis is Y (the BVH convention), so a walker heading
    along +Z turning with omega traces an arc in the X-Z plane and stays at a
    constant height. Checking radius at several horizons verifies the projection
    is a true arc rather than an approximation.
    """
    speed = 10.0
    omega = np.pi / 2.0
    radius = speed / omega  # 6.366 m

    for dt in (0.25, 1.0, 2.0):
        result = predictor.predict_state(
            entity_id="test_unit",
            x=0.0,
            y=0.9,
            z=0.0,
            vx=0.0,
            vy=0.0,
            vz=speed,
            ax=0.0,
            ay=0.0,
            az=0.0,
            yaw_rate=omega,
            timestamp=1000.0,
            dt=dt,
        )

        # Height is unaffected by a turn in the ground plane.
        assert result.predicted_y == pytest.approx(0.9, abs=1e-9)

        # Distance from the arc centre must equal the turn radius at every dt.
        centre_x = -radius
        measured = np.hypot(
            result.predicted_x - centre_x, result.predicted_z - 0.0
        )
        assert measured == pytest.approx(radius, abs=1e-6)


def test_straight_motion_matches_constant_acceleration(predictor: KinematicPredictor) -> None:
    """With no turn, the projection must reduce to constant acceleration."""
    result = predictor.predict_state(
        entity_id="test_unit",
        x=0.0,
        y=0.9,
        z=0.0,
        vx=1.0,
        vy=0.0,
        vz=2.0,
        ax=0.5,
        ay=0.0,
        az=-0.5,
        yaw_rate=0.0,
        timestamp=1000.0,
        dt=2.0,
    )

    assert result.predicted_x == pytest.approx(0.0 + 1.0 * 2.0 + 0.5 * 0.5 * 4.0)
    assert result.predicted_z == pytest.approx(0.0 + 2.0 * 2.0 + 0.5 * -0.5 * 4.0)
    assert result.predicted_y == pytest.approx(0.9)


def test_z_up_convention_is_supported(predictor: KinematicPredictor) -> None:
    """
    With vertical_axis=2 the ground plane is X-Y, so turning moves Y and the Z
    coordinate stays fixed.
    """
    result = predictor.predict_state(
        entity_id="test_unit",
        x=0.0,
        y=0.0,
        z=1.2,
        vx=2.0,
        vy=0.0,
        vz=0.0,
        ax=0.0,
        ay=0.0,
        az=0.0,
        yaw_rate=0.5,
        timestamp=1000.0,
        dt=1.0,
        vertical_axis=2,
    )

    assert result.predicted_z == pytest.approx(1.2, abs=1e-9)
    assert result.predicted_y != 0.0
