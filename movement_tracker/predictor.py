"""
Kinematic Movement Prediction Engine.

Projects a filtered state forward in time using linear (constant-acceleration)
or turning (coordinated-turn) dynamic models, and reports the growth of
positional uncertainty over the projection.
"""

import numpy as np

from schema import PredictionOutput, TelemetryInput

# Multiplier applied to the position standard deviation to obtain a 95%
# confidence radius. For an isotropic 3D Gaussian the radius containing 95% of
# the probability mass is chi-square distributed with 3 degrees of freedom, whose
# 0.95 quantile is 2.7955. Using the 1D value of 1.96 here would understate the
# region by ~30%.
CHI2_95_3D: float = 2.7955


class KinematicPredictor:
    """
    Computes linear and turning 3D movement projections from state parameters.
    """

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
        convention used by BVH exports or a Z-up convention. Vertical motion is
        projected independently of the turn, because turning happens in the
        ground plane and does not rotate gravity.
        """
        omega = yaw_rate

        components = np.array([x, y, z], dtype=np.float64)
        velocity = np.array([vx, vy, vz], dtype=np.float64)
        acceleration = np.array([ax, ay, az], dtype=np.float64)

        ground_axes = [axis for axis in (0, 1, 2) if axis != vertical_axis]
        up = vertical_axis
        e1, e2 = ground_axes[0], ground_axes[1]

        projected = np.empty(3, dtype=np.float64)

        if abs(omega) > 1e-5:
            # Coordinated turn: constant speed along a circular arc in the ground
            # plane. Rotating the horizontal pair by omega*dt places the target
            # on the arc, matching how a person follows a curved walking path.
            angle = omega * dt
            sin_a, cos_a = np.sin(angle), np.cos(angle)

            v1, v2 = velocity[e1], velocity[e2]
            # Rotation in the (e1, e2) plane about the vertical axis.
            projected[e1] = components[e1] + (v1 * sin_a + v2 * (cos_a - 1.0)) / omega
            projected[e2] = components[e2] + (v1 * (1.0 - cos_a) + v2 * sin_a) / omega
        else:
            # Constant acceleration in the ground plane.
            projected[e1] = (
                components[e1]
                + velocity[e1] * dt
                + 0.5 * acceleration[e1] * dt**2
            )
            projected[e2] = (
                components[e2]
                + velocity[e2] * dt
                + 0.5 * acceleration[e2] * dt**2
            )

        # Vertical motion is independent of the turn.
        projected[up] = (
            components[up] + velocity[up] * dt + 0.5 * acceleration[up] * dt**2
        )

        pred_x, pred_y, pred_z = float(projected[0]), float(projected[1]), float(projected[2])

        future_timestamp: float = timestamp + dt

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
        # Radius covering 95% of a 3D Gaussian of this scale.
        confidence_radius_95 = float(CHI2_95_3D * position_std)

        return PredictionOutput(
            entity_id=entity_id,
            predicted_x=float(pred_x),
            predicted_y=float(pred_y),
            predicted_z=float(pred_z),
            horizon_seconds=dt,
            target_timestamp=future_timestamp,
            sigma_x=float(sigmas[0]),
            sigma_y=float(sigmas[1]),
            sigma_z=float(sigmas[2]),
            position_std=position_std,
            confidence_radius_95=confidence_radius_95,
            position_covariance=[float(value) for value in covariance.ravel()],
            motion_model=motion_model,
        )
