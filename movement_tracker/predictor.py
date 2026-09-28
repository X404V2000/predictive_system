"""
Kinematic Movement Prediction Engine.

Projects a filtered state forward in time using linear (constant-acceleration)
or turning (coordinated-turn) dynamic models, and reports the growth of
positional uncertainty over the projection.

Two output shapes are supported:

  * a single position at one horizon (`predict_state`)
  * a full trajectory sampled across a span of horizons (`predict_trajectory`),
    each step carrying its own uncertainty

The trajectory form matters because a single point hides how confidence decays.
Projecting 0.1 s ahead and 2 s ahead are very different claims, and a caller
drawing a future path needs the whole curve, not one sample of it.
"""

from typing import Callable

import numpy as np

from schema import PredictionOutput, TelemetryInput, TrajectoryPoint

# Multiplier applied to the position standard deviation to obtain a 95%
# confidence radius. For an isotropic 3D Gaussian the radius containing 95% of
# the probability mass is chi-square distributed with 3 degrees of freedom, whose
# 0.95 quantile is 2.7955. Using the 1D value of 1.96 here would understate the
# region by ~30%.
CHI2_95_3D: float = 2.7955

# Maximum number of samples a trajectory may contain. Guards against a caller
# requesting an enormous number of points and stalling the service.
MAX_TRAJECTORY_STEPS: int = 200


class KinematicPredictor:
    """
    Computes linear and turning 3D movement projections from state parameters.
    """

    # -- projection core ---------------------------------------------------

    @staticmethod
    def _project(
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        yaw_rate: float,
        dt: float,
        vertical_axis: int,
    ) -> np.ndarray:
        """
        Projects one state forward by `dt` seconds.

        Turning is applied in the plane perpendicular to `vertical_axis`, and
        vertical motion is projected independently because a turn does not rotate
        gravity. Keeping this in one place means the single-step and multi-step
        paths can never disagree.
        """
        ground_axes = [axis for axis in (0, 1, 2) if axis != vertical_axis]
        e1, e2 = ground_axes[0], ground_axes[1]
        up = vertical_axis

        projected = np.empty(3, dtype=np.float64)

        if abs(yaw_rate) > 1e-5:
            # Coordinated turn: constant speed along a circular arc in the ground
            # plane. Rotating the horizontal velocity pair by omega*dt places the
            # target on the arc, matching how a person follows a curved path.
            angle = yaw_rate * dt
            sin_a, cos_a = np.sin(angle), np.cos(angle)
            v1, v2 = velocity[e1], velocity[e2]

            projected[e1] = position[e1] + (v1 * sin_a + v2 * (cos_a - 1.0)) / yaw_rate
            projected[e2] = position[e2] + (v1 * (1.0 - cos_a) + v2 * sin_a) / yaw_rate
        else:
            # Constant acceleration in the ground plane.
            projected[e1] = (
                position[e1] + velocity[e1] * dt + 0.5 * acceleration[e1] * dt**2
            )
            projected[e2] = (
                position[e2] + velocity[e2] * dt + 0.5 * acceleration[e2] * dt**2
            )

        # Vertical motion is independent of the turn.
        projected[up] = (
            position[up] + velocity[up] * dt + 0.5 * acceleration[up] * dt**2
        )
        return projected

    @staticmethod
    def _summarise_covariance(
        position_covariance: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Reduces a 3x3 covariance to the reported uncertainty quantities.

        Returns (covariance, per-axis sigmas, combined std, 95% radius).
        """
        if position_covariance is None:
            covariance = np.zeros((3, 3), dtype=np.float64)
        else:
            covariance = np.asarray(position_covariance, dtype=np.float64)
            if covariance.shape != (3, 3):
                raise ValueError(
                    f"position_covariance must be 3x3, got {covariance.shape}"
                )
            # Guard against tiny negative variances from floating-point error
            # before taking square roots.
            covariance = 0.5 * (covariance + covariance.T)

        variances = np.clip(np.diag(covariance), 0.0, None)
        sigmas = np.sqrt(variances)
        position_std = float(np.sqrt(float(variances.sum())))
        return covariance, sigmas, position_std, float(CHI2_95_3D * position_std)

    # -- single-step output ------------------------------------------------

    def predict_state(
        self,
        entity_id: str,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        ax: float,
        ay: float,
        az: float,
        yaw_rate: float,
        timestamp: float,
        dt: float,
        position_covariance: np.ndarray | None = None,
        motion_model: str = "constant_acceleration",
        vertical_axis: int = 1,
        warming_up: bool = False,
    ) -> PredictionOutput:
        """
        Projects the state forward by `dt` seconds and reports the result with
        its uncertainty.

        `position_covariance` is the 3x3 position covariance already propagated
        to the horizon. When omitted (for example when filtering is disabled), a
        zero covariance is reported, which correctly signals "no estimate of
        uncertainty" rather than inventing one.

        `vertical_axis` selects which coordinate is up. Turning is applied in the
        plane perpendicular to it, so the model works under either the Y-up
        convention used by BVH exports or a Z-up convention.
        """
        position = np.array([x, y, z], dtype=np.float64)
        velocity = np.array([vx, vy, vz], dtype=np.float64)
        acceleration = np.array([ax, ay, az], dtype=np.float64)

        projected = self._project(
            position, velocity, acceleration, yaw_rate, dt, vertical_axis
        )
        covariance, sigmas, position_std, radius_95 = self._summarise_covariance(
            position_covariance
        )

        return PredictionOutput(
            entity_id=entity_id,
            predicted_x=float(projected[0]),
            predicted_y=float(projected[1]),
            predicted_z=float(projected[2]),
            horizon_seconds=dt,
            target_timestamp=timestamp + dt,
            sigma_x=float(sigmas[0]),
            sigma_y=float(sigmas[1]),
            sigma_z=float(sigmas[2]),
            position_std=position_std,
            confidence_radius_95=radius_95,
            position_covariance=[float(value) for value in covariance.ravel()],
            motion_model=motion_model,
            warming_up=warming_up,
        )

    # -- multi-step trajectory ---------------------------------------------

    def predict_trajectory(
        self,
        entity_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        yaw_rate: float,
        timestamp: float,
        max_horizon: float,
        step: float,
        covariance_fn: Callable[[float], np.ndarray],
        motion_model: str = "constant_acceleration",
        vertical_axis: int = 1,
    ) -> list[TrajectoryPoint]:
        """
        Samples the projection across `max_horizon` at intervals of `step`,
        including both endpoints.

        Each point carries its own uncertainty, obtained from `covariance_fn`, a
        callable taking a horizon in seconds and returning the 3x3 position
        covariance expected there. Uncertainty is evaluated per step rather than
        interpolated from the final value, because position error integrates
        twice over time and so does not grow linearly with the horizon.

        Every point is projected from the ORIGINAL state rather than by chaining
        one step onto the previous output. Chaining would compound the
        turn-model approximation along the path and drift away from the true arc.
        """
        if max_horizon <= 0:
            raise ValueError("max_horizon must be greater than zero.")
        if step <= 0:
            raise ValueError("step must be greater than zero.")

        count = int(np.floor(max_horizon / step)) + 1
        if count > MAX_TRAJECTORY_STEPS:
            raise ValueError(
                f"trajectory would contain {count} points, exceeding the limit of "
                f"{MAX_TRAJECTORY_STEPS}; increase `step` or reduce `max_horizon`."
            )

        points: list[TrajectoryPoint] = []
        for index in range(count):
            horizon = min(index * step, max_horizon)

            projected = self._project(
                position, velocity, acceleration, yaw_rate, horizon, vertical_axis
            )
            _, sigmas, position_std, radius_95 = self._summarise_covariance(
                covariance_fn(horizon)
            )

            points.append(
                TrajectoryPoint(
                    horizon_seconds=float(horizon),
                    timestamp=float(timestamp + horizon),
                    x=float(projected[0]),
                    y=float(projected[1]),
                    z=float(projected[2]),
                    sigma_x=float(sigmas[0]),
                    sigma_y=float(sigmas[1]),
                    sigma_z=float(sigmas[2]),
                    position_std=position_std,
                    confidence_radius_95=radius_95,
                )
            )

        return points
