"""
Multi-Entity Motion Tracker Manager with Memory Lifecycles.

Manages state tracking, isolated EKF filters, and memory pruning for multi-entity targets.
"""

import time
import numpy as np
from filter import ExtendedKalmanFilter3D
from predictor import KinematicPredictor
from schema import PredictionOutput, TelemetryInput


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
    ) -> None:
        """
        Initializes entity map registry, kinematics engine, and eviction TTL.

        min_measurement_dt / max_measurement_dt clamp the elapsed interval between
        consecutive measurements before it is fed to the filter, guarding against
        duplicate timestamps (dt == 0) and long telemetry gaps.

        prune_interval_seconds throttles automatic stale-entity eviction so the
        O(n) sweep does not run on every telemetry frame.
        """
        self.filter_enabled: bool = filter_enabled
        self.ttl_seconds: float = ttl_seconds
        self.min_measurement_dt: float = min_measurement_dt
        self.max_measurement_dt: float = max_measurement_dt
        self.prune_interval_seconds: float = prune_interval_seconds
        self.predictors: dict[str, KinematicPredictor] = {}
        self.filters: dict[str, ExtendedKalmanFilter3D] = {}

        # Wall-clock (monotonic) time of last activity, used solely for TTL eviction.
        self.last_seen: dict[str, float] = {}

        # Telemetry timestamp of the previous sample, used to derive the true
        # integration interval between measurements for the filter. This is a
        # distinct quantity from self.last_seen.
        self.last_measurement_ts: dict[str, float] = {}

        # Monotonic timestamp of the most recent pruning sweep (throttle state).
        self._last_prune_time: float = 0.0

    def _maybe_prune(self, now: float) -> int:
        """
        Runs stale-entity eviction at most once per prune_interval_seconds.

        Returns the number of entities removed during this call (0 if the sweep
        was skipped by the throttle).
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

        return len(stale_ids)

    def process_telemetry(
        self, telemetry: TelemetryInput, dt: float = 1.0
    ) -> PredictionOutput:
        """
        Updates the EKF for a target entity and computes a future spatial prediction.

        Time semantics — these are two DIFFERENT physical quantities and must not
        be conflated:

          * measurement_dt : elapsed telemetry time since this entity's previous
                             sample. This is the correct integration step for the
                             filter's `predict()` and is derived from the incoming
                             timestamps, not from the caller.
          * dt             : the forward prediction horizon. Used ONLY to
                             extrapolate the filtered state into the future.

        Passing the prediction horizon into the filter's integration step
        over-integrates the state and silently corrupts the covariance.
        """
        entity_id = telemetry.entity_id
        now = time.monotonic()
        self.last_seen[entity_id] = now

        # Opportunistic, throttled eviction of expired entities.
        self._maybe_prune(now)

        # Register new target entity if unknown
        is_new_entity = entity_id not in self.filters
        if is_new_entity:
            self.filters[entity_id] = ExtendedKalmanFilter3D()
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
            raw_measurement = np.array(
                [
                    telemetry.x,
                    telemetry.y,
                    telemetry.z,
                    telemetry.vx,
                    telemetry.vy,
                    telemetry.vz,
                    telemetry.ax,
                    telemetry.ay,
                    telemetry.az,
                ],
                dtype=np.float64,
            )

            # First sample for an entity: no interval exists yet, so only correct
            # the initial state rather than integrating a fictitious step.
            if measurement_dt is not None:
                ekf.predict(dt=measurement_dt)
            ekf.update(measurement=raw_measurement)

            state = ekf.x.flatten()
            x, y, z = state[0], state[1], state[2]
            vx, vy, vz = state[3], state[4], state[5]
            ax, ay, az = state[6], state[7], state[8]
        else:
            x, y, z = telemetry.x, telemetry.y, telemetry.z
            vx, vy, vz = telemetry.vx, telemetry.vy, telemetry.vz
            ax, ay, az = telemetry.ax, telemetry.ay, telemetry.az

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
            yaw_rate=telemetry.yaw_rate,
            timestamp=telemetry.timestamp,
            dt=dt,
        )
