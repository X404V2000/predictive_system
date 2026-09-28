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

    trajectory: list[TrajectoryPoint] | None = Field(
        default=None,
        description=(
            "Optional full future path sampled from now to the horizon, each "
            "point with its own uncertainty. Present only when the caller "
            "requests a trajectory; the single-point fields above always "
            "describe the final horizon."
        ),
    )
