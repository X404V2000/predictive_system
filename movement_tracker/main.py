"""
FastAPI Application Entrypoint for Spatial Trajectory Streaming.

Exposes RESTful and WebSocket real-time interfaces.
"""

import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from schema import PredictionOutput, TelemetryInput
from tracker import EntityTracker

app = FastAPI(
    title="Movement Prediction API",
    description="Real-time low-latency trajectory prediction engine using dynamic state filtering.",
    version="2.0.0",
)

# Instantiate multi-entity tracking manager
tracker_manager = EntityTracker(filter_enabled=True)

# Physically defensible extrapolation bounds. Beyond roughly two seconds a pure
# kinematic projection stops describing human motion and starts inventing it, so
# requests outside this window are rejected rather than answered with fiction.
MIN_TIME_HORIZON: float = 1e-3
MAX_TIME_HORIZON: float = 2.0


def _validate_horizon(time_horizon: float) -> float:
    """
    Validates the requested prediction horizon against physical bounds.

    Raises ValueError (a transport-agnostic signal) so that both the REST layer
    and the WebSocket layer can translate the failure into their own error
    format instead of one transport's exception leaking into the other.
    """
    if time_horizon <= 0:
        raise ValueError("Time horizon parameter must be greater than zero.")
    if time_horizon > MAX_TIME_HORIZON:
        raise ValueError(
            f"Time horizon parameter must not exceed {MAX_TIME_HORIZON} seconds; "
            "long-range kinematic extrapolation is not physically meaningful."
        )
    return time_horizon


@app.post("/predict", response_model=PredictionOutput)
async def compute_prediction(telemetry: TelemetryInput, time_horizon: float = 1.0):
    """
    HTTP REST endpoint to execute point-in-time trajectory predictions.
    """
    try:
        _validate_horizon(time_horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return tracker_manager.process_telemetry(telemetry=telemetry, dt=time_horizon)


@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    WebSocket endpoint streaming continuous raw telemetry in and target predictions out.

    A malformed frame is reported back to the client as an error object and the
    stream continues, rather than tearing down the connection on a single bad
    message.
    """
    await websocket.accept()
    try:
        while True:
            # Receive incoming JSON string from client stream
            raw_data = await websocket.receive_text()

            try:
                data_dict = json.loads(raw_data)
                if not isinstance(data_dict, dict):
                    raise ValueError("Telemetry frame must be a JSON object.")

                # Extract optional parameter time_horizon (defaulting to 1.0s)
                time_horizon = float(data_dict.pop("time_horizon", 1.0))
                _validate_horizon(time_horizon)

                # Validate input payload against Pydantic schema
                telemetry = TelemetryInput(**data_dict)
            except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            # Process state update and compute spatial projection
            prediction = tracker_manager.process_telemetry(
                telemetry=telemetry, dt=time_horizon
            )

            # Send serialized prediction back across WebSocket link
            await websocket.send_text(prediction.model_dump_json())

    except WebSocketDisconnect:
        # Handle graceful disconnect
        pass
