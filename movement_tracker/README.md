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


## Running

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest                                   # 36 tests
uvicorn main:app --reload                # http://127.0.0.1:8000/docs
```

### Benchmark against real motion capture

```bash
python3 tools/benchmark.py                       # ADE/FDE vs baselines, all clips
python3 tools/benchmark.py --horizon 0.25
python3 tools/benchmark.py --noise 0.10          # heavier sensor noise
python3 tools/benchmark.py --subjects 007 009    # walking + running only
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

