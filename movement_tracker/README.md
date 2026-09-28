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
   → tracker.py    per-entity EKF instances, lifecycle + TTL eviction
   → filter.py     9-state Extended Kalman Filter  [x,y,z, vx,vy,vz, ax,ay,az]
   → predictor.py  Constant-Acceleration / Coordinated-Turn projection
   → PredictionOutput
```

## The two time quantities (important)

The pipeline distinguishes two physical quantities that are easy to conflate:

| Quantity | Meaning | Used by |
|---|---|---|
| `measurement_dt` | elapsed telemetry time between consecutive samples for one entity | the filter's `predict()` integration step |
| `dt` | the forward prediction horizon | extrapolation only |

Passing the prediction horizon into the filter's integration step over-integrates
the state and silently corrupts the covariance. `tests/test_accuracy.py` guards
against this: with the bug present the filtered prediction error rises to
~3.7 m; correct, it is ~0.18 m.

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

