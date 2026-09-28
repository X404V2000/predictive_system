# Predictive Movement System

A real-time engine that estimates a moving target's physical state and projects
it forward in time. It is the foundation of a two-layer human-movement
prediction system.

**Layer 1 — Physical (implemented):** given noisy telemetry, predict *where* the
body will be over the next 0.1–2 s, using an Extended Kalman Filter plus
kinematic motion models.

**Layer 2 — Behavioural (planned):** given the physical state, predict *what
action* comes next (walking, turning, sitting, falling) as a probability
distribution, learned from motion-capture data.

## Architecture

```
Telemetry (REST / WebSocket)
   → schema.py     Pydantic input/output validation
   → tracker.py    per-entity filters, motion-model selection, lifecycle + TTL
   → filter.py     9-state Kalman filter  [x,y,z, vx,vy,vz, ax,ay,az]
   → predictor.py  Constant-Acceleration / Coordinated-Turn projection
   → PredictionOutput
```

### Measurement and motion models

Two choices carry most of the accuracy:

- **Position-only measurement.** A position sensor does not observe velocity.
  Feeding the filter a finite-difference velocity derived from noisy positions
  injects large phantom motion: at 120 Hz, 2 cm of noise becomes ~2.4 m/s of
  apparent velocity. The filter therefore observes position only and infers
  velocity from position history.
- **Automatic motion-model selection.** One process-noise setting cannot serve
  both a seated person and a runner. The tracker selects between `stationary`,
  `constant_velocity`, `constant_acceleration` and `maneuvering`, judging motion
  from displacement over a fixed time window. Under the `stationary` model
  velocity decays geometrically toward zero (a Gauss-Markov process), which is
  what stops a motionless target from accumulating drift.

## Time handling (important)

The pipeline distinguishes quantities that are easy to conflate:

| Quantity | Meaning | Used by |
|---|---|---|
| `measurement_dt` | elapsed telemetry time between consecutive samples for one entity | the filter's `predict()` integration step |
| `dt` | the forward prediction horizon | extrapolation only |
| `SPEED_WINDOW_SECONDS` | fixed duration over which displacement is measured | motion-model selection |

Passing the prediction horizon into the filter's integration step over-integrates
the state and silently corrupts the covariance.

All timing is expressed in **seconds rather than frames**. Noise-induced apparent
speed is `sqrt(2)*sigma/dt`, so from 2 cm of noise a motionless target appears to
move at 0.28 m/s when sampled at 10 Hz but 3.4 m/s at 120 Hz. Measuring
displacement over a fixed window divides that noise by a fixed duration, making
classification and velocity estimation independent of the frame rate.

## Uncertainty

A prediction is a distribution, not a point. Every response carries:

| Field | Meaning |
|---|---|
| `sigma_x/y/z` | per-axis position standard deviation at the horizon, metres |
| `position_std` | `sqrt(sigma_x^2 + sigma_y^2 + sigma_z^2)` |
| `confidence_radius_95` | radius of the 95% confidence sphere |
| `position_covariance` | full 3x3 covariance, row-major (9 values) |
| `motion_model` | active model, which sets the process-noise profile |

Two points matter:

- **Uncertainty is propagated, not copied.** The covariance is integrated
  forward through the motion model to the horizon. Position error integrates
  twice over time, so a 1 s projection is far less certain than a 0.1 s one;
  reporting the present covariance would understate it by orders of magnitude.
- **Process noise scales with the step.** The discrete white-noise-acceleration
  model gives position variance growing as `dt^4` and velocity as `dt^2`. A
  constant diagonal `Q`, independent of `dt`, is dimensionally wrong and makes
  long-horizon uncertainty explode quadratically faster than reality.

`tools/calibrate.py` measures the reported uncertainty against the error actually
observed on real motion capture, which is how the process-noise values were set.
At realistic noise the reported standard deviation is within roughly a factor of
two of observed error.

## Turning motion

A person following a curved path is projected along a circular arc
(coordinated-turn) rather than a straight line. Two details decide whether this
helps or hurts:

- **The ground plane depends on the up axis.** `vertical_axis` selects it
  (default 1 = Y-up, the BVH convention; use 2 for Z-up). Turning is applied
  perpendicular to that axis and vertical motion is projected independently.
  Applying a Z-up assumption to Y-up data measures heading in a plane containing
  almost no motion and reports thousands of degrees per second for a straight walk.
- **Turn-rate artefacts must be rejected.** The rate is estimated from
  filtered velocity over a fixed duration, then gated: no turn below locomotion
  speed (heading is undefined when barely moving) and none below a
  `MIN_TURN_RADIUS` of 3 m (a smaller radius is not a path a body can follow).
  Without the radius gate, running saturated the rate cap and prediction became
  40% worse (ADE 0.97 → 1.38 m).

On the CMU clips the turn model improves walking accuracy by ~9-10% (ADE
0.413 → 0.377 m) while leaving running and seated clips unchanged.

### Known limit at long horizons

Beyond ~1 s the constant-acceleration assumption still degrades: over a 2 s
window a walking person changes heading by 10-20 degrees. The turn model
recovers part of this (1 s calibration ratio 0.414 → 0.379), but at 2 s real
error still outgrows predicted error, so the confidence region is over-confident
there. This is model error, not measurement noise, and is why the prediction
horizon is capped at 2 s.



## Running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest                                   # 73 tests
uvicorn main:app --reload                # http://127.0.0.1:8000/docs
```

### Benchmark against real motion capture

```bash
python3 tools/benchmark.py                       # ADE/FDE vs baselines, all clips
python3 tools/benchmark.py --horizon 0.25
python3 tools/benchmark.py --noise 0.10          # heavier sensor noise
python3 tools/benchmark.py --subjects 007 009    # walking + running only
```

### Calibration against real motion capture

```bash
python3 tools/calibrate.py     # reported uncertainty vs observed error
```

On real CMU walking clips at a realistic 0.02 m sensor noise the filtered
prediction beats raw extrapolation by roughly 6x in ADE, and the gap widens with
noise: raw differencing amplifies measurement noise directly, while the filter
does not.


### REST

```bash
curl -X POST "http://127.0.0.1:8000/predict?time_horizon=0.5" \
  -H "Content-Type: application/json" \
  -d '{"entity_id":"p1","x":0,"y":0,"z":0,"vx":5,"timestamp":1700000000}'
```

### WebSocket

Connect to `ws://127.0.0.1:8000/ws/telemetry` and send the same JSON object per
frame (optionally including `time_horizon`). Malformed frames receive an in-band
`{"error": ...}` object and do not close the stream.

## Limits

- Prediction horizon is bounded to 2 s — beyond that a purely kinematic
  projection stops describing human motion.
- Prediction is a point estimate; uncertainty output is planned (Phase B1).
- Motion models currently cover yaw rotation only (Phase B3 adds pitch/roll).

## Data

`training_data/cmu_mocap/` holds CMU Graphics Lab Motion Capture trials in BVH
format (the cgspeed conversion): walking (007), turning (008), running (009) and
seated (013, 020) clips at 120 fps. The loader reads each file's own skeleton
hierarchy, so joint semantics and frame timing come from the data rather than
assumptions.

Two details the loader encodes explicitly:

- **Unit scale:** one raw BVH unit is 0.05 m, verified against anatomy (the leg
  chain measures 17.01 units for a ~0.85 m hip-to-ankle length). Treating the
  values as plain centimetres understates every speed by 5x.
- **Root trajectory:** only the root joint's global translation is used for
  trajectory prediction; joint rotations are retained for the behavioural layer.

`tests/test_mocap.py` asserts physical plausibility (walk 0.3–2.0 m/s, run faster
than walk, seated clips travel < 2 m), which is what detects dataset corruption.

## Tools

```bash
python3 tools/camera_check.py                    # probe the webcam
python3 tools/camera_check.py --manual-exposure 100   # restore full FPS in dim light
python3 tools/camera_check.py --preview          # live window, press q to quit
```

A UVC camera in dim light lengthens its exposure and silently drops its frame
rate. On this machine the webcam reported 30 FPS but delivered 7.5 FPS until
exposure was pinned, at which point it delivered the full 30 FPS.

