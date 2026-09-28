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
pytest                                   # 20 tests
uvicorn main:app --reload                # http://127.0.0.1:8000/docs
```

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

`training_data/h3.6m/` holds Human3.6M motion capture (33 joints × 3 coords at
50 fps, world coordinates in metres; clip filenames are the action labels). It is
git-ignored and used for Phase B/C validation and action-model training.
