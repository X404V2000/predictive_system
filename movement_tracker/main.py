"""
FastAPI Application Entrypoint for Spatial Trajectory Streaming.

Exposes RESTful and WebSocket real-time interfaces.
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
import numpy as np

from schema import (
    ActionConfidence,
    ActionPrediction,
    PoseWindowInput,
    PredictionOutput,
    TelemetryInput,
)
from tracker import EntityTracker

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Trains the action model during startup so no user request pays for it.

    Training takes roughly 25 seconds on this hardware. Doing it lazily on the
    first request made that single request slow enough to look like a failure and
    to trip client timeouts. A failure here is not fatal: the physics endpoints
    work without the dataset, and /predict/action reports 503 with a reason.
    """
    try:
        _get_action_model()
    except RuntimeError:
        # Left for the endpoint to report with a proper status code.
        pass
    yield


app = FastAPI(
    title="Movement Prediction API",
    description=(
        "Trajectory prediction from kinematic filtering, plus action recognition "
        "from skeletal motion."
    ),
    version="2.1.0",
    lifespan=lifespan,
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


MIN_TRAJECTORY_STEP: float = 0.02


def _validate_trajectory_step(step: float, horizon: float) -> float:
    """
    Validates the sampling interval for a requested trajectory.

    Raises ValueError so both transports can render the failure in their own
    format.
    """
    if step < MIN_TRAJECTORY_STEP:
        raise ValueError(
            f"trajectory_step must be at least {MIN_TRAJECTORY_STEP} seconds."
        )
    if step > horizon:
        raise ValueError(
            "trajectory_step must not exceed the prediction horizon, otherwise "
            "the trajectory would contain only its endpoints."
        )
    return step


@app.post("/predict", response_model=PredictionOutput)
async def compute_prediction(
    telemetry: TelemetryInput,
    time_horizon: float = 1.0,
    trajectory: bool = False,
    trajectory_step: float = 0.2,
):
    """
    HTTP REST endpoint to execute point-in-time trajectory predictions.

    Set `trajectory=true` to receive the whole future path sampled every
    `trajectory_step` seconds, each point with its own uncertainty.
    """
    try:
        _validate_horizon(time_horizon)
        if trajectory:
            _validate_trajectory_step(trajectory_step, time_horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return tracker_manager.process_telemetry(
        telemetry=telemetry,
        dt=time_horizon,
        trajectory=trajectory,
        trajectory_step=trajectory_step,
    )


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

                # Optional trajectory request, sampled every trajectory_step.
                want_trajectory = bool(data_dict.pop("trajectory", False))
                trajectory_step = float(data_dict.pop("trajectory_step", 0.2))
                if want_trajectory:
                    _validate_trajectory_step(trajectory_step, time_horizon)

                # Validate input payload against Pydantic schema
                telemetry = TelemetryInput(**data_dict)
            except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            # Process state update and compute spatial projection
            prediction = tracker_manager.process_telemetry(
                telemetry=telemetry,
                dt=time_horizon,
                trajectory=want_trajectory,
                trajectory_step=trajectory_step,
            )

            # Send serialized prediction back across WebSocket link
            await websocket.send_text(prediction.model_dump_json())

    except WebSocketDisconnect:
        # Handle graceful disconnect
        pass


# --- behavioural layer ----------------------------------------------------
#
# The action model is trained once at first use rather than at import, so the
# physics API stays usable when the motion-capture dataset is absent. Training
# takes a second or two; the fitted model is then reused for every request.

_action_model: object | None = None
_action_model_error: str | None = None


def _get_action_model():
    """
    Returns the trained action classifier, training it on first use.

    Raises RuntimeError with an explanatory message when the labelled dataset is
    unavailable, so a caller receives a clear diagnosis instead of an opaque 500.
    """
    global _action_model, _action_model_error

    if _action_model is not None:
        return _action_model
    if _action_model_error is not None:
        raise RuntimeError(_action_model_error)

    try:
        from actions import train_default_model

        _action_model = train_default_model()
    except Exception as exc:  # dataset missing or unreadable
        _action_model_error = (
            f"action model unavailable: {exc}. The behavioural layer needs the "
            "labelled motion-capture dataset under training_data/cmu_mocap."
        )
        raise RuntimeError(_action_model_error) from exc

    return _action_model


MIN_ACTION_FRAMES: int = 10



def _joint_array(payload: PoseWindowInput) -> np.ndarray:
    """
    Converts the request's nested lists into a validated (frames, joints, 3) array.

    Raises ValueError for shapes the feature extractor cannot consume, so the
    caller gets a 400 rather than a stack trace.
    """
    try:
        positions = np.asarray(payload.positions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"positions must be numeric: {exc}") from exc

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            "positions must be shaped (frames, joints, 3); got "
            f"{tuple(positions.shape)}"
        )
    if positions.shape[0] < MIN_ACTION_FRAMES:
        raise ValueError(
            f"at least {MIN_ACTION_FRAMES} frames are required to characterise "
            f"motion; got {positions.shape[0]}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions contain non-finite values")

    return positions


def _reorder_joints(positions: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """
    Reorders a supplied skeleton into the training joint order by name.

    Accepting names lets a caller send a differently ordered skeleton, which is
    the common case for cameras and pose estimators that do not use the CMU
    layout. A missing joint is an error: silently substituting one would produce
    plausible-looking but wrong features.
    """
    from features import JOINT_ORDER

    lookup = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in JOINT_ORDER if name not in lookup]
    if missing:
        raise ValueError(
            f"joint_names is missing {len(missing)} required joint(s): "
            f"{', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
        )

    order = [lookup[name] for name in JOINT_ORDER]
    return positions[:, order, :]


@app.post("/predict/action", response_model=ActionPrediction)
async def predict_action(payload: PoseWindowInput):
    """
    Classifies the action being performed in a window of skeletal motion.

    Returns a calibrated probability distribution over coarse actions rather than
    a bare label, so a caller can tell a confident recognition from a guess.
    """
    try:
        model = _get_action_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        positions = _joint_array(payload)
        if payload.joint_names is not None:
            positions = _reorder_joints(positions, payload.joint_names)

        # The root trajectory is the hip joint, read from the supplied skeleton
        # rather than asked for separately, so the two can never disagree.
        from features import ROOT_JOINT, features_from_arrays

        root = positions[:, ROOT_JOINT, :]
        values = features_from_arrays(root, positions, payload.sample_dt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probabilities = model.predict_proba(values)
    ranked = sorted(
        (
            {"action": name, "probability": float(probabilities[index])}
            for index, name in enumerate(model.classes)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )

    from actions import LOW_CONFIDENCE_THRESHOLD, MODEL_VERSION

    top = ranked[0]
    return ActionPrediction(
        entity_id=payload.entity_id,
        action=top["action"],
        confidence=top["probability"],
        distribution=[ActionConfidence(**item) for item in ranked],
        window_seconds=float(positions.shape[0] * payload.sample_dt),
        model_version=MODEL_VERSION,
        low_confidence=top["probability"] < LOW_CONFIDENCE_THRESHOLD,
    )
