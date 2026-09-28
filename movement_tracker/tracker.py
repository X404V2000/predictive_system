"""
Multi-Entity Motion Tracker Manager with Memory Lifecycles.

Manages state tracking, isolated EKF filters, automatic motion-model selection
and memory pruning for multi-entity targets.
"""

import time

import numpy as np

from filter import ExtendedKalmanFilter3D, MeasurementModel, MotionModel
from predictor import KinematicPredictor
from schema import PredictionOutput, TelemetryInput

# Speed thresholds (m/s) at which the tracker switches motion model. Hysteresis
# is applied: the filter must fall clearly below a threshold before dropping to
# a quieter model, otherwise measurement noise makes the model flap every frame.
STATIONARY_SPEED_ENTER: float = 0.15
STATIONARY_SPEED_EXIT: float = 0.30
WALKING_SPEED_ENTER: float = 1.80
WALKING_SPEED_EXIT: float = 1.50

# Time constant (seconds) for the low-pass filter applied to the speed estimate
# that drives model selection. Expressed in TIME, not frames, so the tracker
# behaves identically at 10 Hz and 120 Hz. A frame-based equivalent made the
# filter converge beautifully at 10 Hz and oscillate at 120 Hz.
SPEED_SMOOTHING_TIME_CONSTANT: float = 0.5

# Minimum duration (seconds) of sustained evidence before escalating to a
# noisier model. Escalation is reluctant and de-escalation immediate, because a
# wrongly-noisy model invents motion that is not present.
ESCALATION_SECONDS: float = 0.5

# Duration over which displacement is measured to infer motion. Measuring over a
# fixed time (rather than between consecutive samples) is what makes the speed
# estimate independent of the telemetry frame rate: noise-induced apparent speed
# is sqrt(2)*sigma/window regardless of how often samples arrive.
SPEED_WINDOW_SECONDS: float = 0.4

# Duration over which the heading change is measured to estimate turn rate.
# Comparing across a fixed duration rather than consecutive samples matters: at
# 120 Hz a straight walker's lateral velocity is almost entirely noise.
HEADING_WINDOW_SECONDS: float = 0.4

# Speed below which heading is undefined, so no turn is reported. Set at
# locomotion speed because a seated person shifting posture moves at ~0.15 m/s
# and would otherwise register large spurious rotation.
HEADING_MIN_SPEED: float = 0.5

# Magnitude cap on reported turn rate (rad/s). A human cannot rotate their
# heading faster than roughly this; anything larger is an estimation artefact.
MAX_YAW_RATE: float = 3.0

# Minimum credible turn radius (metres). A person turning a corner walks a radius
# of several metres; a smaller radius means the heading estimate is changing
# faster than a body could follow, so the turn is rejected as an artefact.
MIN_TURN_RADIUS: float = 3.0

# Ordering of motion models by how much process noise they inject, used to tell
# an escalation (noisier) from a de-escalation (quieter).
_MODEL_NOISE_ORDER: dict[MotionModel, int] = {
    MotionModel.STATIONARY: 0,
    MotionModel.CONSTANT_VELOCITY: 1,
    MotionModel.CONSTANT_ACCELERATION: 2,
    MotionModel.MANEUVERING: 3,
}


class EntityTracker:
    """
    Manages isolated EKF state instances and lifetime cycles for target IDs.
    """

    def __init__(
        self,
        filter_enabled: bool = True,
        ttl_seconds: float = 300.0,
        min_measurement_dt: float = 1e-3,
        max_measurement_dt: float = 5.0,
        prune_interval_seconds: float = 1.0,
        measurement_noise_std: float = 0.02,
        adaptive_motion_model: bool = True,
        vertical_axis: int = 1,
    ) -> None:
        """
        Initialises the entity registry, kinematics engine and eviction policy.

        min_measurement_dt / max_measurement_dt clamp the elapsed interval between
        consecutive measurements before it reaches the filter, guarding against
        duplicate timestamps (dt == 0) and long telemetry gaps.

        prune_interval_seconds throttles automatic stale-entity eviction so the
        O(n) sweep does not run on every telemetry frame.

        measurement_noise_std is the assumed sensor noise on position, in metres.
        It sets the filter's measurement covariance and should reflect reality:
        too small and the filter ignores its own smoothing, too large and it
        ignores the sensor.

        adaptive_motion_model enables automatic process-noise selection based on
        the filter's own speed estimate.

        vertical_axis selects which coordinate is up (1 = Y-up, the BVH
        convention; 2 = Z-up). Turning is measured in the plane perpendicular to
        it.
        """
        if vertical_axis not in (0, 1, 2):
            raise ValueError(f"vertical_axis must be 0, 1 or 2, got {vertical_axis}")

        self.filter_enabled: bool = filter_enabled
        self.ttl_seconds: float = ttl_seconds
        self.min_measurement_dt: float = min_measurement_dt
        self.max_measurement_dt: float = max_measurement_dt
        self.prune_interval_seconds: float = prune_interval_seconds
        self.measurement_noise_std: float = measurement_noise_std
        self.adaptive_motion_model: bool = adaptive_motion_model
        self.vertical_axis: int = vertical_axis

        self.predictors: dict[str, KinematicPredictor] = {}
        self.filters: dict[str, ExtendedKalmanFilter3D] = {}

        # Wall-clock (monotonic) time of last activity, used solely for TTL eviction.
        self.last_seen: dict[str, float] = {}

        # Telemetry timestamp of the previous sample, used to derive the true
        # integration interval between measurements. Distinct from last_seen.
        self.last_measurement_ts: dict[str, float] = {}

        # Smoothed speed per entity, used for motion-model selection so that
        # per-frame measurement noise cannot drive the decision.
        self._smoothed_speed: dict[str, float] = {}

        # Accumulated seconds of evidence supporting a noisier model.
        self._escalation_seconds: dict[str, float] = {}

        # Rolling window of filtered velocity vectors, used to estimate the
        # heading rate for the coordinated-turn projection.
        self._velocity_history: dict[str, list[np.ndarray]] = {}

        # Rolling window of filtered positions and their elapsed times, used to
        # measure displacement over a fixed duration independently of the
        # velocity the stationary model suppresses.
        self._position_window: dict[str, list[np.ndarray]] = {}
        self._position_history: dict[str, list[float]] = {}
        self._elapsed_seconds: dict[str, float] = {}

        # Monotonic timestamp of the most recent pruning sweep (throttle state).
        self._last_prune_time: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    def _maybe_prune(self, now: float) -> int:
        """
        Runs stale-entity eviction at most once per prune_interval_seconds.

        Returns the number of entities removed (0 when throttled).
        """
        if (now - self._last_prune_time) < self.prune_interval_seconds:
            return 0
        self._last_prune_time = now
        return self.prune_stale_entities(now)

    def prune_stale_entities(self, current_time: float) -> int:
        """
        Removes tracked entities that have exceeded the inactivity threshold.
        Returns the count of pruned entities.
        """
        stale_ids = [
            entity_id
            for entity_id, last_timestamp in self.last_seen.items()
            if (current_time - last_timestamp) > self.ttl_seconds
        ]

        for entity_id in stale_ids:
            del self.filters[entity_id]
            del self.predictors[entity_id]
            del self.last_seen[entity_id]
            self.last_measurement_ts.pop(entity_id, None)
            self._smoothed_speed.pop(entity_id, None)
            self._escalation_seconds.pop(entity_id, None)
            self._position_window.pop(entity_id, None)
            self._position_history.pop(entity_id, None)
            self._elapsed_seconds.pop(entity_id, None)
            self._velocity_history.pop(entity_id, None)

        return len(stale_ids)

    # -- motion-model selection -------------------------------------------

    @staticmethod
    def _select_motion_model(current: MotionModel, speed: float) -> MotionModel:
        """
        Chooses a process-noise profile from a smoothed speed estimate.

        Selecting directly from the instantaneous estimate creates a feedback
        loop: measurement noise inflates the estimated speed, which selects a
        noisier model, which trusts measurements more, which inflates the
        estimate further. The filter then latches into MANEUVERING and can never
        return to STATIONARY.

        Two guards prevent that:
          * the caller supplies a smoothed speed, not the raw estimate
          * escalation requires sustained evidence (see ESCALATION_SECONDS)
        """
        if current is MotionModel.STATIONARY:
            # Stay stationary until motion is convincingly present.
            if speed >= STATIONARY_SPEED_EXIT:
                return (
                    MotionModel.MANEUVERING
                    if speed >= WALKING_SPEED_ENTER
                    else MotionModel.CONSTANT_ACCELERATION
                )
            return MotionModel.STATIONARY

        if current is MotionModel.MANEUVERING:
            if speed > WALKING_SPEED_EXIT:
                return MotionModel.MANEUVERING
            if speed < STATIONARY_SPEED_ENTER:
                return MotionModel.STATIONARY
            return MotionModel.CONSTANT_ACCELERATION

        # CONSTANT_VELOCITY / CONSTANT_ACCELERATION: move either way as needed.
        if speed < STATIONARY_SPEED_ENTER:
            return MotionModel.STATIONARY
        if speed >= WALKING_SPEED_ENTER:
            return MotionModel.MANEUVERING
        return MotionModel.CONSTANT_ACCELERATION

    def _update_motion_model(
        self, entity_id: str, ekf: ExtendedKalmanFilter3D, dt: float
    ) -> None:
        """
        Updates the observed-motion estimate and, when justified, switches the
        filter's motion model.

        Motion is judged from DISPLACEMENT OVER A FIXED TIME WINDOW, not from
        displacement between consecutive frames and not from the filter's own
        velocity estimate.

        Why not consecutive frames: the apparent speed induced by measurement
        noise is sqrt(2)*sigma/dt. At sigma = 2 cm that is 0.28 m/s at 10 Hz but
        3.4 m/s at 120 Hz, so a motionless target looks like a sprinter purely
        because the sensor is fast. Measuring across a fixed window of seconds
        divides the same noise by a fixed duration, making the metric
        rate-independent.

        Why not the filter's velocity: the STATIONARY model decays velocity
        toward zero, so using it to decide when to leave STATIONARY creates a
        one-way trap where genuinely walking targets are suppressed to a
        standstill.

        Escalation to a noisier model requires ESCALATION_SECONDS of sustained
        evidence; de-escalation to a quieter model is immediate. This asymmetry
        is deliberate: a model that is too noisy manufactures motion, whereas one
        that is too quiet merely lags a genuine change briefly.
        """
        current_position = ekf.position
        if entity_id not in self._position_window:
            self._position_window[entity_id] = []
            self._position_history[entity_id] = []
            self._elapsed_seconds[entity_id] = 0.0

        timestamps = self._position_history[entity_id]
        positions = self._position_window[entity_id]

        timestamps.append(self._elapsed_seconds[entity_id])
        positions.append(current_position.copy())
        self._elapsed_seconds[entity_id] += dt

        if len(positions) < 2:
            return

        # Drop samples older than the measurement window.
        cutoff = self._elapsed_seconds[entity_id] - SPEED_WINDOW_SECONDS
        while len(timestamps) > 2 and timestamps[1] < cutoff:
            timestamps.pop(0)
            positions.pop(0)

        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return

        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        instantaneous = displacement / elapsed

        # Time-constant form of an exponential moving average: alpha grows with
        # dt so the effective memory is a fixed duration, not a fixed frame count.
        alpha = 1.0 - float(np.exp(-dt / SPEED_SMOOTHING_TIME_CONSTANT))
        previous = self._smoothed_speed.get(entity_id, instantaneous)
        smoothed = (alpha * instantaneous) + ((1.0 - alpha) * previous)
        self._smoothed_speed[entity_id] = smoothed

        candidate = self._select_motion_model(ekf.motion_model, smoothed)
        if candidate is ekf.motion_model:
            self._escalation_seconds[entity_id] = 0.0
            return

        escalating = (
            _MODEL_NOISE_ORDER[candidate] > _MODEL_NOISE_ORDER[ekf.motion_model]
        )

        if escalating:
            elapsed = self._escalation_seconds.get(entity_id, 0.0) + dt
            self._escalation_seconds[entity_id] = elapsed
            if elapsed < ESCALATION_SECONDS:
                return
        else:
            self._escalation_seconds[entity_id] = 0.0

        ekf.set_motion_model(candidate)
        self._escalation_seconds[entity_id] = 0.0

    # -- turn-rate estimation ---------------------------------------------

    def _estimate_yaw_rate(
        self,
        entity_id: str,
        ekf: ExtendedKalmanFilter3D,
        dt: float | None,
    ) -> float:
        """
        Estimates the heading rate (rad/s) from the filtered velocity history.

        The client-supplied `yaw_rate` defaults to zero, so relying on it leaves
        the coordinated-turn projection permanently inactive. Turning is instead
        measured from how the filtered velocity direction changes over time.

        Two guards make the estimate usable:

          * Heading is compared across samples separated by a fixed DURATION,
            not across consecutive samples. At 120 Hz the lateral velocity of a
            straight walker is dominated by noise; differencing frame to frame
            reported 122 deg/s where the truth is 18 deg/s.
          * No turn is reported below locomotion speed, where heading is
            undefined. A seated person shifting posture would otherwise register
            large spurious rotation.
        """
        history = self._velocity_history.setdefault(entity_id, [])
        history.append(ekf.velocity.copy())

        rate_dt = dt if dt and dt > 0 else 0.01
        max_samples = max(2, int(HEADING_WINDOW_SECONDS / rate_dt) + 2)
        while len(history) > max_samples:
            history.pop(0)

        if len(history) < 2 or not dt or dt <= 0:
            return 0.0

        ground_axes = [axis for axis in (0, 1, 2) if axis != self.vertical_axis]
        e1, e2 = ground_axes[0], ground_axes[1]

        # Average velocity over the window suppresses per-sample jitter.
        mean_velocity = np.mean(np.stack(history), axis=0)
        horizontal_speed = float(np.hypot(mean_velocity[e1], mean_velocity[e2]))
        if horizontal_speed < HEADING_MIN_SPEED:
            return 0.0

        # Compare heading at the start and end of the window, yielding a rate
        # over a fixed duration rather than over one sample interval.
        first, last = history[0], history[-1]
        first_heading = np.arctan2(first[e2], first[e1])
        last_heading = np.arctan2(last[e2], last[e1])
        delta = float(
            np.arctan2(
                np.sin(last_heading - first_heading),
                np.cos(last_heading - first_heading),
            )
        )
        elapsed = dt * (len(history) - 1)
        if elapsed <= 0.0:
            return 0.0

        rate = delta / elapsed

        # Credibility gate on turn RADIUS. The coordinated-turn model describes a
        # constant-speed circular arc, which only fits a person following a
        # curved path. A turn radius below a few metres means the heading is
        # changing far faster than any body could follow, so the signal is an
        # estimation artefact rather than motion.
        #
        # This matters most at speed: on a running clip, 30% of samples saturated
        # the rate cap, and applying those arcs made prediction 40% worse
        # (ADE 0.97 -> 1.38 m). Walking clips, where turning is genuine, were
        # unaffected by the gate.
        if rate != 0.0:
            radius = horizontal_speed / abs(rate)
            if radius < MIN_TURN_RADIUS:
                return 0.0

        return float(np.clip(rate, -MAX_YAW_RATE, MAX_YAW_RATE))

    # -- main entry point --------------------------------------------------

    def process_telemetry(
        self, telemetry: TelemetryInput, dt: float = 1.0
    ) -> PredictionOutput:
        """
        Updates the filter for a target entity and computes a future prediction.

        Time semantics — two DIFFERENT physical quantities that must not be
        conflated:

          * measurement_dt : elapsed telemetry time since this entity's previous
                             sample. The correct integration step for the filter's
                             `predict()`, derived from the incoming timestamps.
          * dt             : the forward prediction horizon, used ONLY to
                             extrapolate the filtered state into the future.

        Passing the prediction horizon into the filter's integration step
        over-integrates the state and silently corrupts the covariance.
        """
        entity_id = telemetry.entity_id
        now = time.monotonic()
        self.last_seen[entity_id] = now

        # Opportunistic, throttled eviction of expired entities.
        self._maybe_prune(now)

        if entity_id not in self.filters:
            self.filters[entity_id] = ExtendedKalmanFilter3D(
                measurement_noise_std=self.measurement_noise_std,
                measurement_model=MeasurementModel.POSITION,
                motion_model=MotionModel.CONSTANT_ACCELERATION,
            )
            self.predictors[entity_id] = KinematicPredictor()

        ekf = self.filters[entity_id]
        predictor = self.predictors[entity_id]

        # Derive the true interval since this entity's previous measurement.
        previous_ts = self.last_measurement_ts.get(entity_id)
        if previous_ts is None:
            measurement_dt = None
        else:
            measurement_dt = float(
                np.clip(
                    telemetry.timestamp - previous_ts,
                    self.min_measurement_dt,
                    self.max_measurement_dt,
                )
            )
        self.last_measurement_ts[entity_id] = telemetry.timestamp

        if self.filter_enabled:
            # The sensor observes POSITION only. Velocity supplied on the wire is
            # deliberately not used as a measurement: a finite-difference velocity
            # computed from noisy positions carries far more error than the value
            # it claims to represent, and trusting it injects phantom motion.
            position_measurement = np.array(
                [telemetry.x, telemetry.y, telemetry.z], dtype=np.float64
            )

            # First sample has no elapsed interval, so only correct the state.
            if measurement_dt is not None:
                ekf.predict(dt=measurement_dt)

            ekf.update(measurement=position_measurement)

            if self.adaptive_motion_model and measurement_dt is not None:
                self._update_motion_model(entity_id, ekf, measurement_dt)

            state = ekf.x.flatten()
            x, y, z = state[0], state[1], state[2]
            vx, vy, vz = state[3], state[4], state[5]
            ax, ay, az = state[6], state[7], state[8]

            # Propagate the covariance to the prediction horizon, so the reported
            # uncertainty reflects the projection rather than the present moment.
            position_covariance = ekf.propagate_position_covariance(dt)
            motion_model = ekf.motion_model.value

            # Estimate the turn rate from filtered velocity rather than trusting
            # the wire value, which defaults to zero. Without this the
            # coordinated-turn branch of the predictor is never reached.
            yaw_rate = self._estimate_yaw_rate(entity_id, ekf, measurement_dt)
        else:
            x, y, z = telemetry.x, telemetry.y, telemetry.z
            vx, vy, vz = telemetry.vx, telemetry.vy, telemetry.vz
            ax, ay, az = telemetry.ax, telemetry.ay, telemetry.az
            # Filtering is off, so no uncertainty is estimated. Reporting zeros
            # is honest; inventing a value would not be.
            position_covariance = None
            motion_model = "unfiltered"
            yaw_rate = telemetry.yaw_rate

        return predictor.predict_state(
            entity_id=entity_id,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            ax=ax,
            ay=ay,
            az=az,
            yaw_rate=yaw_rate,
            timestamp=telemetry.timestamp,
            dt=dt,
            position_covariance=position_covariance,
            motion_model=motion_model,
            vertical_axis=self.vertical_axis,
        )
