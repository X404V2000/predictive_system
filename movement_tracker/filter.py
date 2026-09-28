"""
Extended Kalman Filter (EKF) for 3D Movement State Estimation.

Filters noisy raw sensor inputs to produce optimized continuous estimations 
of spatial state vectors (position, velocity, acceleration) prior to projection.
"""

import numpy as np


class ExtendedKalmanFilter3D:
    """
    State Estimator tracking 9 state variables:
    State Vector X = [x, y, z, vx, vy, vz, ax, ay, az]^T
    """

    def __init__(
        self,
        process_noise_std: float = 0.1,
        measurement_noise_std: float = 0.5,
    ) -> None:
        """
        Initializes state covariance matrices and noise dynamics parameters.
        """
        # Define 9D state dimension (3 pos, 3 vel, 3 acc)
        self.dim_x: int = 9

        # Initialize current state estimate vector [9x1] with zeros
        self.x: np.ndarray = np.zeros((9, 1), dtype=np.float64)

        # Initialize state covariance matrix P [9x9] with high initial uncertainty
        self.P: np.ndarray = np.eye(9, dtype=np.float64) * 500.0

        # Define process noise covariance matrix Q [9x9]
        self.Q: np.ndarray = np.eye(9, dtype=np.float64) * (process_noise_std**2)

        # Define measurement noise covariance matrix R [9x9]
        self.R: np.ndarray = np.eye(9, dtype=np.float64) * (measurement_noise_std**2)

        # Define measurement mapping matrix H [9x9] (direct observation model)
        self.H: np.ndarray = np.eye(9, dtype=np.float64)

    def predict(self, dt: float) -> None:
        """
        Predicts state estimate and covariance forward by time interval dt.
        F matrix maps kinematic transitions across dt.
        """
        # Construct dynamic 9x9 State Transition Matrix F
        F = np.eye(9, dtype=np.float64)

        # Apply velocity integration to position: x = x + vx*dt + 0.5*ax*dt^2
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[0, 6] = 0.5 * (dt**2)
        F[1, 7] = 0.5 * (dt**2)
        F[2, 8] = 0.5 * (dt**2)

        # Apply acceleration integration to velocity: vx = vx + ax*dt
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        # Update state vector extrapolation: X_k = F * X_{k-1}
        self.x = F @ self.x

        # Update covariance uncertainty extrapolation: P_k = F * P_{k-1} * F^T + Q
        self.P = (F @ self.P @ F.T) + self.Q

    def update(self, measurement: np.ndarray) -> None:
        """
        Updates state estimate and covariance matrix with incoming raw telemetry measurement [9x1].
        """
        # Format measurement as column matrix
        z = measurement.reshape((9, 1))

        # Calculate measurement residual (innovation): y = z - H * x
        y = z - (self.H @ self.x)

        # Calculate residual covariance matrix: S = H * P * H^T + R
        S = (self.H @ self.P @ self.H.T) + self.R

        # Calculate optimal Kalman Gain matrix: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate vector using Kalman Gain: x = x + K * y
        self.x = self.x + (K @ y)

        # Update state error covariance matrix: P = (I - K * H) * P
        I = np.eye(self.dim_x, dtype=np.float64)
        self.P = (I - (K @ self.H)) @ self.P
