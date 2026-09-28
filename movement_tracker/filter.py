"""
Extended Kalman Filter (EKF) for 3D Movement State Estimation.

Filters noisy observations to produce optimised continuous estimates of the
spatial state vector (position, velocity, acceleration) prior to projection.

Two design choices matter and both are configurable:

1. Measurement model. A position-sensing system observes POSITION ONLY. Feeding
   a finite-difference velocity derived from noisy positions injects large
   phantom velocity: at 120 Hz, 2 cm of position noise becomes ~2.4 m/s of
   apparent velocity. The default model therefore observes position only and
   lets the filter infer velocity from position history.

2. Motion model. A single fixed process noise cannot serve both a seated person
   and a runner. `MotionModel` selects the process-noise profile, and the
   tracker chooses it automatically from the observed motion.
"""

from enum import Enum

import numpy as np


class MeasurementModel(str, Enum):
    """Which components of the state the sensor observes."""

    POSITION = "position"
    FULL = "full"


class MotionModel(str, Enum):
    """
    Process-noise profiles describing how rapidly the target's acceleration is
    expected to change.

    STATIONARY              the target is essentially still
    CONSTANT_VELOCITY       smooth motion with little acceleration change
    CONSTANT_ACCELERATION   general human motion (default)
    MANEUVERING             running or sharp turns
    """

    STATIONARY = "stationary"
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    MANEUVERING = "maneuvering"


# Process-noise standard deviation per motion model. Smaller values mean the
# filter trusts its smoothness assumption more, which is what a stationary
# target needs in order to reject phantom velocity.
_MOTION_MODEL_NOISE: dict[MotionModel, float] = {
    MotionModel.STATIONARY: 0.01,
    MotionModel.CONSTANT_VELOCITY: 0.05,
    MotionModel.CONSTANT_ACCELERATION: 0.30,
    MotionModel.MANEUVERING: 1.00,
}

# Time constant (seconds) over which velocity decays under the STATIONARY model.
# Short enough to reject phantom motion quickly, long enough not to fight a
# genuine slow movement during the moment before the model switches.
STATIONARY_CORRELATION_TIME: float = 0.5


class ExtendedKalmanFilter3D:
    """
    State estimator tracking 9 variables:
        X = [x, y, z, vx, vy, vz, ax, ay, az]^T

    The transition is linear, so this is a standard Kalman filter; the name is
    retained for compatibility.
    """

    def __init__(
        self,
        process_noise_std: float | None = None,
        measurement_noise_std: float = 0.02,
        measurement_model: MeasurementModel = MeasurementModel.POSITION,
        motion_model: MotionModel = MotionModel.CONSTANT_ACCELERATION,
    ) -> None:
        """
        Initialises the state, covariance and noise models.

        `process_noise_std` overrides the motion model's default when supplied.
        """
        self.dim_x: int = 9
        self.measurement_model: MeasurementModel = measurement_model

        # Current state estimate [9x1] and covariance [9x9]. Initial uncertainty
        # is large because one observation says nothing about velocity.
        self.x: np.ndarray = np.zeros((9, 1), dtype=np.float64)
        self.P: np.ndarray = np.eye(9, dtype=np.float64) * 100.0

        self.measurement_noise_std: float = measurement_noise_std

        self.set_motion_model(motion_model, process_noise_std)

        if measurement_model is MeasurementModel.POSITION:
            self.H: np.ndarray = np.zeros((3, 9), dtype=np.float64)
            self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
            self.R: np.ndarray = np.eye(3, dtype=np.float64) * (
                measurement_noise_std**2
            )
        else:
            self.H = np.eye(9, dtype=np.float64)
            self.R = np.eye(9, dtype=np.float64) * (measurement_noise_std**2)

        # Cached identity, reused by every update. Allocating it per call was a
        # measurable share of the per-frame cost.
        self._identity: np.ndarray = np.eye(self.dim_x, dtype=np.float64)

    # -- configuration -----------------------------------------------------

    def set_motion_model(
        self, motion_model: MotionModel, process_noise_std: float | None = None
    ) -> None:
        """
        Switches the process-noise profile used by the prediction step.

        Recomputes Q in place, so an existing filter can adapt to a change in the
        observed motion without discarding its accumulated state.
        """
        self.motion_model = motion_model
        std = (
            process_noise_std
            if process_noise_std is not None
            else _MOTION_MODEL_NOISE[motion_model]
        )
        self.Q: np.ndarray = np.eye(9, dtype=np.float64) * (std**2)

    def _transition(self, dt: float) -> np.ndarray:
        """
        Builds the 9x9 state transition matrix.

        For moving models this is constant acceleration. For the STATIONARY
        model, velocity and acceleration decay geometrically toward zero, which
        is a first-order Gauss-Markov (Ornstein-Uhlenbeck) process. Merely
        reducing process noise is not enough: a random walk in velocity still
        drifts, so the estimate wanders away from a truly motionless target.
        """
        F = np.eye(9, dtype=np.float64)

        if self.motion_model is MotionModel.STATIONARY:
            # A short correlation time means velocity forgets itself quickly,
            # which is the correct prior for someone who is not moving.
            decay = float(np.exp(-dt / STATIONARY_CORRELATION_TIME))
            F[3, 3] = F[4, 4] = F[5, 5] = decay
            F[6, 6] = F[7, 7] = F[8, 8] = decay
            return F

        # Position advances by velocity plus the acceleration term.
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        F[0, 6] = F[1, 7] = F[2, 8] = 0.5 * dt * dt
        # Velocity advances by acceleration.
        F[3, 6] = F[4, 7] = F[5, 8] = dt
        return F

    # -- filter steps ------------------------------------------------------

    def predict(self, dt: float) -> None:
        """
        Propagates the estimate and covariance forward by `dt` seconds.

        `dt` is the elapsed time since the previous measurement, NOT a forward
        prediction horizon.
        """
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = (F @ self.P @ F.T) + self.Q
        # Repeated updates accumulate asymmetry through rounding, which can
        # eventually produce an invalid gain.
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, measurement: np.ndarray) -> None:
        """
        Corrects the estimate with a new observation.

        `measurement` is length 3 for the position model, length 9 for full.

        The gain is obtained by solving S K^T = (H P^T)^T rather than inverting
        S explicitly: solving is both faster and numerically better behaved.
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(-1, 1)

        expected_dim = 3 if self.measurement_model is MeasurementModel.POSITION else 9
        if z.shape[0] != expected_dim:
            raise ValueError(
                f"{self.measurement_model.value} measurement model expects "
                f"{expected_dim} values, got {z.shape[0]}"
            )

        innovation = z - (self.H @ self.x)

        # H P H^T is small (3x3 for position, 9x9 for full) and symmetric.
        PHt = self.P @ self.H.T
        S = (self.H @ PHt) + self.R
        # K = P H^T S^-1  <=>  S K^T = (P H^T)^T
        K = np.linalg.solve(S, PHt.T).T

        self.x += K @ innovation

        # Covariance update. The Joseph form (I-KH)P(I-KH)^T + KRK^T is the most
        # robust, but costs two 9x9 multiplies per frame. For the position model
        # H selects the leading 3 rows, so K H is sparse and the standard form
        # (I - K H) P is both cheaper and exact here. Symmetrising afterwards
        # guards against accumulated rounding error in either branch.
        i_kh = self._identity - (K @ self.H)
        self.P = (i_kh @ self.P) + (K @ self.R @ K.T)
        self.P = 0.5 * (self.P + self.P.T)

    # -- convenience -------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6, 0].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def reset(self) -> None:
        """Clears the estimate back to its initial uninformative state."""
        self.x = np.zeros((9, 1), dtype=np.float64)
        self.P = np.eye(9, dtype=np.float64) * 100.0
