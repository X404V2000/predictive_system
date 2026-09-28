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


class PredictionOutput(BaseModel):
    """
    Represents extrapolated future spatial metrics for a targeted entity.
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
