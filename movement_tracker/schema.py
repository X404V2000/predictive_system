"""
Data Validation Schemas for Movement Predictive Software.

Supports multi-target entity tracking over real-time protocols.
"""

from pydantic import BaseModel, ConfigDict, Field


class TelemetryInput(BaseModel):
    """
    Represents spatial telemetry from a tracked target entity.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(
        default="default_entity",
        description="Unique string identifier for tracked object",
    )
    x: float = Field(..., description="X coordinate in meters relative to origin")
    y: float = Field(..., description="Y coordinate in meters relative to origin")
    z: float = Field(..., description="Z coordinate in meters relative to origin")
    vx: float = Field(
        default=0.0, description="Instantaneous velocity along X-axis in m/s"
    )
    vy: float = Field(
        default=0.0, description="Instantaneous velocity along Y-axis in m/s"
    )
    vz: float = Field(
        default=0.0, description="Instantaneous velocity along Z-axis in m/s"
    )
    ax: float = Field(
        default=0.0, description="Instantaneous acceleration along X-axis in m/s^2"
    )
    ay: float = Field(
        default=0.0, description="Instantaneous acceleration along Y-axis in m/s^2"
    )
    az: float = Field(
        default=0.0, description="Instantaneous acceleration along Z-axis in m/s^2"
    )
    yaw_rate: float = Field(
        default=0.0, description="Angular velocity around Z-axis in rad/s"
    )
    timestamp: float = Field(
        ..., description="Unix epoch timestamp associated with the reading"
    )


class TrajectoryPoint(BaseModel):
    """
    One sample along a predicted future path.

    Each point carries its own uncertainty because confidence decays with the
    horizon, and it decays non-linearly: position error integrates twice over
    time. A caller drawing a future path can therefore show not just where the
    target is expected to be, but how sharply that expectation blurs.
    """

    model_config = ConfigDict(extra="forbid")

    horizon_seconds: float = Field(
        ..., ge=0.0, description="Seconds ahead of the current observation"
    )
    timestamp: float = Field(
        ..., description="Absolute time corresponding to this point"
    )
    x: float = Field(..., description="Predicted X coordinate, metres")
    y: float = Field(..., description="Predicted Y coordinate, metres")
    z: float = Field(..., description="Predicted Z coordinate, metres")

    sigma_x: float = Field(
        ..., ge=0.0, description="Position standard deviation along X, metres"
    )
    sigma_y: float = Field(
        ..., ge=0.0, description="Position standard deviation along Y, metres"
    )
    sigma_z: float = Field(
        ..., ge=0.0, description="Position standard deviation along Z, metres"
    )
    position_std: float = Field(
        ..., ge=0.0, description="Combined position standard deviation, metres"
    )
    confidence_radius_95: float = Field(
        ..., ge=0.0, description="Radius of the 95% confidence sphere, metres"
    )


class PredictionOutput(BaseModel):
    """
    Represents extrapolated future spatial metrics for a targeted entity.

    A prediction is a distribution, not a point. The uncertainty fields describe
    how confident the estimate is at the horizon; they grow with the horizon
    because position error accumulates as the projection lengthens.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(
        ..., description="Unique string identifier for tracked object"
    )
    predicted_x: float = Field(
        ..., description="Extrapolated X coordinate at horizon time"
    )
    predicted_y: float = Field(
        ..., description="Extrapolated Y coordinate at horizon time"
    )
    predicted_z: float = Field(
        ..., description="Extrapolated Z coordinate at horizon time"
    )
    horizon_seconds: float = Field(
        ..., description="Time interval into the future calculated"
    )
    target_timestamp: float = Field(
        ..., description="Target timestamp corresponding to the prediction"
    )

    # --- uncertainty -------------------------------------------------------
    sigma_x: float = Field(
        ..., ge=0.0, description="Position standard deviation along X at the horizon, metres"
    )
    sigma_y: float = Field(
        ..., ge=0.0, description="Position standard deviation along Y at the horizon, metres"
    )
    sigma_z: float = Field(
        ..., ge=0.0, description="Position standard deviation along Z at the horizon, metres"
    )
    position_std: float = Field(
        ...,
        ge=0.0,
        description=(
            "Radius of one standard deviation: sqrt(sigma_x^2 + sigma_y^2 + "
            "sigma_z^2), metres"
        ),
    )
    confidence_radius_95: float = Field(
        ...,
        ge=0.0,
        description=(
            "Radius of the 95% confidence sphere around the predicted point, "
            "metres. Derived from the propagated covariance."
        ),
    )
    position_covariance: list[float] = Field(
        ...,
        description=(
            "The 3x3 position covariance at the horizon, in row-major order "
            "(9 values). Full information for clients that need anisotropy."
        ),
    )
    motion_model: str = Field(
        ...,
        description=(
            "Active motion model used for the projection: stationary, "
            "constant_velocity, constant_acceleration or maneuvering. Larger "
            "process noise means a longer assumed correlation, hence wider "
            "uncertainty."
        ),
    )

    warming_up: bool = Field(
        default=False,
        description=(
            "True while the filter is still converging, which takes about 20 "
            "samples. Early projections are unreliable because the initial "
            "covariance is large: one position sample says nothing about velocity. "
            "Treat the point as indicative, not actionable, until this clears."
        ),
    )

    trajectory: list[TrajectoryPoint] | None = Field(
        default=None,
        description=(
            "Optional full future path sampled from now to the horizon, each "
            "point with its own uncertainty. Present only when the caller "
            "requests a trajectory; the single-point fields above always "
            "describe the final horizon."
        ),
    )


class ActionConfidence(BaseModel):
    """Probability assigned to one candidate action."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., description="Action class name")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Posterior probability of this action"
    )


class PoseWindowInput(BaseModel):
    """
    A window of skeletal motion supplied for action classification.

    A window rather than a single frame is required because the motion features
    describe how the body moves OVER TIME: locomotion speed, vertical travel,
    cadence and turn rate are all undefined for one instant.

    Positions are expected as (frames, joints, 3) in METRES, in the same joint
    order the model was trained on (CMU 31-joint skeleton, Hips first, Y up).
    Joint order can be supplied via `joint_names` and will otherwise be assumed
    to match the training skeleton exactly.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(..., description="Identifier for the tracked subject")
    positions: list[list[list[float]]] = Field(
        ...,
        description=(
            "Joint world positions, shaped (frames, joints, 3), in metres. "
            "Use the CMU 31-joint order unless joint_names is given."
        ),
    )
    sample_dt: float = Field(
        ...,
        gt=0.0,
        description="Seconds between consecutive frames, required for velocity features",
    )
    joint_names: list[str] | None = Field(
        default=None,
        description=(
            "Optional joint names matching the frame axis. When given, the "
            "expected joints are looked up by name so a differently ordered "
            "skeleton can be supplied."
        ),
    )


class ActionPrediction(BaseModel):
    """
    Distribution over coarse actions for one observation window.

    The full distribution is returned rather than a single label because a
    consumer deciding how to respond needs to know when the classifier is
    uncertain. Confidence is calibrated: on held-out subjects the stated
    probability tracks observed accuracy to within a few points.
    """

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(..., description="Identifier for the tracked subject")
    action: str = Field(..., description="Most likely action")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of the most likely action"
    )
    distribution: list[ActionConfidence] = Field(
        ...,
        description="Full posterior over all known actions, highest probability first",
    )
    window_seconds: float = Field(
        ..., gt=0.0, description="Duration of motion the request described"
    )
    model_version: str = Field(
        ..., description="Identifier of the model that produced this prediction"
    )
    low_confidence: bool = Field(
        ...,
        description=(
            "True when the top probability is below the model's reliability "
            "floor, meaning the result should be treated as advisory only."
        ),
    )
