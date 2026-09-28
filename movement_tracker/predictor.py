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
    ) -> PredictionOutput:
        """
        Projects the state forward by `dt` seconds and reports the result with
        its uncertainty.

        `position_covariance` is the 3x3 position covariance already propagated
        to the horizon. When omitted (for example when filtering is disabled), a
        zero covariance is reported, which correctly signals "no estimate of
        uncertainty" rather than inventing one.
        """
        omega = yaw_rate

        if abs(omega) > 1e-5:
            # Coordinated Turn model: constant speed along a circular arc.
            sin_w_dt = np.sin(omega * dt)
            cos_w_dt = np.cos(omega * dt)

            pred_x = x + (vx / omega) * sin_w_dt - (vy / omega) * (1.0 - cos_w_dt)
            pred_y = y + (vx / omega) * (1.0 - cos_w_dt) + (vy / omega) * sin_w_dt
            pred_z = z + (vz * dt) + (0.5 * az * (dt**2))
        else:
            # Constant Acceleration model.
            pred_x = x + (vx * dt) + (0.5 * ax * (dt**2))
            pred_y = y + (vy * dt) + (0.5 * ay * (dt**2))
            pred_z = z + (vz * dt) + (0.5 * az * (dt**2))

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
