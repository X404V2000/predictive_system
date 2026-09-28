"""
Kinematic Movement Prediction Engine.

Computes multi-model kinematic spatial transformations.
"""

import numpy as np
from schema import PredictionOutput, TelemetryInput


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
    ) -> PredictionOutput:
        """
        Projects state positions forward using linear (CA) or turning (CT) dynamic models.
        """
        omega = yaw_rate

        # Branch logic based on turning threshold
        if abs(omega) > 1e-5:
            # Coordinated Turn Model (CT)
            sin_w_dt = np.sin(omega * dt)
            cos_w_dt = np.cos(omega * dt)

            pred_x = x + (vx / omega) * sin_w_dt - (vy / omega) * (1.0 - cos_w_dt)
            pred_y = y + (vx / omega) * (1.0 - cos_w_dt) + (vy / omega) * sin_w_dt
            pred_z = z + (vz * dt) + (0.5 * az * (dt**2))
        else:
            # Constant Acceleration Model (CA)
            pred_x = x + (vx * dt) + (0.5 * ax * (dt**2))
            pred_y = y + (vy * dt) + (0.5 * ay * (dt**2))
            pred_z = z + (vz * dt) + (0.5 * az * (dt**2))

        future_timestamp: float = timestamp + dt

        return PredictionOutput(
            entity_id=entity_id,
            predicted_x=float(pred_x),
            predicted_y=float(pred_y),
            predicted_z=float(pred_z),
            horizon_seconds=dt,
            target_timestamp=future_timestamp,
        )
