"""
Integration Tests for FastAPI HTTP Endpoints.

Updated to account for EKF state convergence tolerances.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_predict_endpoint_success() -> None:
    """
    Processes a REST payload and returns a well-formed prediction.

    A single frame carries no motion history, so the velocity estimate stays at
    zero and the prediction is the observed position. Velocity is only inferred
    once successive observations arrive (see the streaming test below).
    """
    payload = {
        "entity_id": "test_target",
        "x": 0.0,
        "y": 1.0,
        "z": 2.0,
        "vx": 5.0,
        "vy": 0.0,
        "vz": 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "az": 0.0,
        "timestamp": 1700000000.0,
    }

    response = client.post("/predict?time_horizon=2.0", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["entity_id"] == "test_target"
    assert data["horizon_seconds"] == 2.0
    # First observation: nothing to extrapolate from yet.
    assert pytest.approx(data["predicted_x"], abs=0.05) == 0.0


def test_repeated_rest_calls_infer_velocity_from_position_history() -> None:
    """
    Streaming successive positions must let the pipeline recover the true
    velocity, which is the behaviour a real position sensor relies on.

    The client-supplied `vx` is deliberately wrong here; a correct pipeline
    derives motion from position rather than trusting the wire value.
    """
    entity = "walking_target"
    velocity = 2.0
    dt = 0.2

    prediction = None
    position = 0.0
    for step in range(60):
        position += velocity * dt
        payload = {
            "entity_id": entity,
            "x": position,
            "y": 0.0,
            "z": 0.0,
            "vx": 999.0,  # deliberately absurd; must be ignored
            "timestamp": 1000.0 + step * dt,
        }
        response = client.post("/predict?time_horizon=1.0", json=payload)
        assert response.status_code == 200
        prediction = response.json()

    # True position at +1 s horizon is current position + velocity * 1 s.
    expected = position + velocity * 1.0
    assert pytest.approx(prediction["predicted_x"], abs=0.2) == expected


def test_websocket_telemetry_stream() -> None:
    """
    Validates asynchronous WebSocket bi-directional streaming and that the
    filter accumulates motion across frames.
    """
    with client.websocket_connect("/ws/telemetry") as websocket:
        velocity = 2.0
        dt = 0.2
        position = 0.0
        response_data = None

        for step in range(60):
            position += velocity * dt
            websocket.send_json(
                {
                    "entity_id": "stream_target",
                    "time_horizon": 1.5,
                    "x": position,
                    "y": 0.0,
                    "z": 0.0,
                    "timestamp": 2000.0 + step * dt,
                }
            )
            response_data = websocket.receive_json()

        assert response_data["entity_id"] == "stream_target"
        assert response_data["horizon_seconds"] == 1.5
        expected = position + velocity * 1.5
        assert pytest.approx(response_data["predicted_x"], abs=0.2) == expected

def test_predict_endpoint_invalid_time_horizon() -> None:
    """
    Validates that zero or negative time horizons raise HTTP 400 validation errors.
    """
    payload = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "az": 0.0,
        "timestamp": 1000.0,
    }

    # Execute HTTP POST with invalid negative time_horizon parameter
    response = client.post("/predict?time_horizon=-1.0", json=payload)

    # Ensure system rejects invalid temporal parameters with HTTP 400
    assert response.status_code == 400
    assert response.json()["detail"] == "Time horizon parameter must be greater than zero."


def test_predict_endpoint_rejects_excessive_time_horizon() -> None:
    """
    Horizons beyond the physical validity window must be rejected rather than
    answered with an invented extrapolation.
    """
    payload = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "vx": 1.0,
        "timestamp": 1000.0,
    }

    response = client.post("/predict?time_horizon=60.0", json=payload)

    assert response.status_code == 400
    assert "must not exceed" in response.json()["detail"]


def test_websocket_survives_malformed_frame() -> None:
    """
    A single malformed frame must yield an in-band error object while the
    stream stays open for subsequent valid telemetry.
    """
    with client.websocket_connect("/ws/telemetry") as websocket:
        # Send a frame that is not valid JSON
        websocket.send_text("{not valid json")
        error_frame = websocket.receive_json()
        assert "error" in error_frame

        # The connection must still be usable afterwards
        websocket.send_json(
            {
                "entity_id": "recovered_target",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "vx": 1.0,
                "timestamp": 3000.0,
            }
        )
        recovered = websocket.receive_json()
        assert recovered["entity_id"] == "recovered_target"


def test_websocket_rejects_excessive_horizon_in_band() -> None:
    """
    An out-of-range horizon inside a WebSocket frame is reported in-band and
    does not close the connection.
    """
    with client.websocket_connect("/ws/telemetry") as websocket:
        websocket.send_json(
            {
                "entity_id": "bad_horizon",
                "time_horizon": 99.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "timestamp": 4000.0,
            }
        )
        error_frame = websocket.receive_json()
        assert "error" in error_frame
