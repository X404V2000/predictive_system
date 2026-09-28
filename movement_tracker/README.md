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
- **Even a "stationary" person drifts.** A seated or standing human shifts
  posture, moving the hip at roughly 0.15 m/s. Treating a motionless target as
  exactly motionless made the 1 s interval over-confident: 95% coverage fell to
  58% on seated clips. Modelling postural drift as a velocity random walk
  restores coverage to 97% while leaving walking accuracy unchanged.

`tools/calibrate.py` measures the reported uncertainty against the error actually
observed on real motion capture, which is how the process-noise values were set.
At realistic noise the reported standard deviation is within roughly a factor of
two of observed error.

## Running the service

### Locally

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements-dev.txt
venv/bin/uvicorn main:app --reload        # http://127.0.0.1:8000/docs
```

### With Docker

```bash
docker compose up --build                       # physics endpoints
docker compose --profile actions up --build     # + action recognition
```

The dataset is **not** baked into the image. It is ~160 MB and re-downloadable,
and the physics endpoints work without it: `/predict/action` returns 503 with an
explanation rather than failing obscurely. The `actions` profile mounts it
read-only at `/app/training_data`.

Two details worth knowing when deploying:

- **Startup takes about 25 seconds** while the action model trains. The healthcheck
  allows a 60 s start period for this; without it an orchestrator can kill a
  container that is merely still warming up.
- **Dependencies are pinned exactly** in `requirements.txt`. An unpinned service
  can change behaviour between builds without any code changing, which would make
  the accuracy figures here impossible to reproduce.

## Live demo

Runs the whole pipeline continuously: pose source -> trajectory (WHERE) ->
action (WHAT).

```bash
python3 tools/live_demo.py                              # synthetic walker
python3 tools/live_demo.py --source mocap --clip 013/13_05
python3 tools/live_demo.py --speed 2.0 --frames 300 --realtime
python3 tools/live_demo.py --json                       # machine-readable
```

Output shows both layers, plus a `warmup` marker:

```
frame  time    now (x,y,z)                 ahead   sigma    motion model          | action
[    2] t=  0.07s warmup now=(  0.007, 0.914,  0.080) ahead=0.613m sd=2.492 constant_acceleration | collecting...
[   62] t=  2.07s        now=( -0.013, 0.880,  2.480) ahead=0.607m sd=0.223 constant_acceleration | locomote    p=0.82
[   67] t=  2.23s        now=(  0.001, 0.903,  2.680) ahead=0.600m sd=0.223 constant_acceleration | locomote    p=0.49 (!)
```

### Real webcam pose: prepared, not yet wired

The pieces for a real webcam skeleton are in place and verified, but the live
loop is not connected to them yet.

```bash
bash tools/download_models.sh     # fetch and checksum-verify the weights (17 MB)
```

| Piece | Status |
|---|---|
| System OpenCV 4.10 with DNN | already installed; exposed to the venv by a `.pth` file, so **no packages are added** |
| Camera from the venv | **30.4 FPS** at 640x480, verified |
| BlazePose pose model | 5.3 MB, SHA-256 verified against the upstream LFS pointer |
| Person detector | 12.0 MB, SHA-256 verified |
| `blazepose.py` mapping | 33 BlazePose landmarks -> 31 CMU joints, verified exact against hand-computed geometry |

**Why MediaPipe was not installed.** `pip install mediapipe` pulls 16 packages
including an 82 MB OpenCV build, duplicating what is already present, on a
dual-core machine with 4 GB free. The OpenCV DNN route reaches the same model
with no new dependency at all.

#### Measured performance ceiling

Running both models on every frame is **not viable on this CPU**:

| Stage | Time | Rate |
|---|---|---|
| Pose inference | 102 ms | ~10 FPS |
| Person detection | 239 ms | ~4 FPS |
| **Both, per frame** | **341 ms** | **2.7 FPS** |

A 2-second action window would take 5 seconds to fill at that rate. The pose
model has a **fixed 256x256 input** and cannot be downscaled to trade accuracy
for speed, so the mitigation must be temporal: run the detector every N frames
and pose from a tracked crop. That recovers roughly 10 FPS, which is usable but
visibly choppy.

#### Accuracy is expected to drop, and will be measured

BlazePose and the CMU skeleton differ in topology: BlazePose has **no spine
chain** and separate hips rather than a single root, so `blazepose.py`
interpolates those joints and fills unobserved finger joints at the wrist.

Features derived from that mapping are therefore approximate, and the action
classifier should **not** be assumed to hold its 81.6% motion-capture accuracy on
webcam input. That gap must be measured before any number is quoted.

#### Landmark layout, established empirically

The indices were read off the model's own output on the reference diagram rather
than taken from memory, which caught two things:

- **Hips are landmarks 23 and 24**, not 11 and 12 (those are the shoulders).
- **Rows 33-38 are padding.** Their visibility is exactly 0.0 for both a real
  person and pure noise, so they are dropped. An early draft also treated the
  coordinates as normalised `[0, 1]` when they are actually **pixels of the
  256x256 model input**, which produced a skeleton 600 m tall.

### Pose sources are pluggable

`pose.py` defines one interface with swappable drivers, so the pipeline does not
depend on how skeletons are obtained. Two ship today:

- **`SyntheticPoseSource`** — a stylised but anatomically valid gait (hip 0.90 m,
  head 0.65 m above hips, feet on the floor, travel at the requested speed). It
  exists so the pipeline runs end to end without a pose model.
- **`MocapPlaybackSource`** — replays a recorded clip as a live stream, for
  exercising the pipeline against real skeletons.

**A real webcam pose estimator is NOT included.** Full-body pose from an RGB
camera needs a trained model (MediaPipe, MoveNet, RTMPose), none of which is
installed. Adding one means implementing a single interface:

```python
class MyWebcamPose:
    name = "webcam"
    def start(self): ...
    def read(self) -> PoseFrame | None:   # (31, 3) metres, CMU order
        ...
    def close(self): ...
```

`reorder_to_cmu()` maps another skeleton's joint names into the required order.
A source that emits the wrong joint order or units will fail loudly on the first
frame rather than producing plausible-looking wrong predictions.

### Warm-up

The filter starts with a large covariance because one position sample says
nothing about velocity. Projections made before about 20 samples are therefore
unreliable, and the response carries `warming_up: true` for that period rather
than presenting an unearned confident-looking point. On real motion capture the
reported sigma falls from ~9 m to under 1 m over those first samples.

## Action recognition

A second layer classifies *what* the body is doing, from 18 motion features
(locomotion, vertical motion, turning, cadence, posture) computed from the root
trajectory and joint positions.

```bash
python3 tools/evaluate_actions.py    # accuracy, confusion matrix, per-class recall
```

Measured with **leave-one-subject-out**, which is the honest protocol here because
class and subject are confounded: most action classes appear in only two of the
five subjects, so a random clip split would leak subject identity.

| Metric | Value |
|---|---|
| Accuracy | **81.6%** |
| Majority baseline | 20.7% |
| Calibration gap | **2.3 points** |

Calibration needed deliberate work. At the numerically minimal covariance ridge
the model claimed 98.3% mean confidence while being correct 85.1% of the time,
with 86% of predictions at 99%+ certainty: every mistake was a confident one. A
ridge of 1.0 cuts the gap to 2.3 points for about 3.5 points of raw accuracy,
which is the right trade for probabilities that downstream code acts on.

### Endpoints

```
POST /predict         # trajectory prediction from a tracked point
POST /predict/action  # action recognition from a window of skeletal motion
WS   /ws/telemetry    # streaming trajectory prediction
```

`/predict/action` accepts joint world positions in metres and returns a ranked,
calibrated distribution:

```bash
uvicorn main:app --reload
curl -X POST http://127.0.0.1:8000/predict/action \
  -H "Content-Type: application/json" \
  -d '{"entity_id":"p1","sample_dt":0.0333,
       "positions":[[[0,0.9,0],[0,0.85,0], ...]]}'
```

Two details matter for callers:

- **A window is required, not a frame.** Speed, cadence and turn rate are
  undefined for a single instant; at least 10 frames are needed to characterise
  motion.
- **`joint_names` lets you send any joint order.** The CMU 31-joint layout is
  assumed by default. Supply names (for example from a pose estimator) and the
  payload is reordered by name. A missing joint is rejected rather than
  substituted, because a wrong joint produces plausible-looking nonsense.

The model is trained once at startup (~25 s), so no request pays for it;
afterwards `/predict/action` answers in about 25 ms.

### What does NOT work yet: predicting the NEXT action

`transitions.py` implements a first-order Markov model over segmented windows, and
it does **not** add predictive value. Leave-one-subject-out:

| | next-action accuracy |
|---|---|
| transition model | 64.9% |
| always predict one class | 64.4% |
| per-state majority successor | 64.9% |

The model's most likely next action is *exactly* the per-state majority successor
(6 of 6 states, both folds), so it re-derives a counting rule rather than learning
structure. The cause is the class definition: one coarse class absorbs 60-86% of
windows and 64% of all transitions, so every row's maximum is "stay put". Window
duration does not fix this (0.5 s to 4.0 s all show zero lift, and shorter windows
make the skew worse).

Making this work requires mutually exclusive action states with comparable
frequencies, which is a data decision rather than a tuning fix.

## Trajectory (multi-step) output

By default a prediction is a single point at one horizon. Adding
`trajectory=true` returns the whole future path:

```bash
curl -X POST "http://127.0.0.1:8000/predict?time_horizon=1.0&trajectory=true&trajectory_step=0.25" \
  -H "Content-Type: application/json" \
  -d '{"entity_id":"p1","x":0,"y":0.9,"z":0,"timestamp":1700000000}'
```

Each point carries its own position and uncertainty. Three properties hold:

- **The final point equals the scalar fields.** One prediction described two
  ways; they can never disagree.
- **Uncertainty is evaluated per step**, not interpolated from the final value.
  Position error integrates twice over time, so it does not grow linearly.
- **Every point is projected from the current state**, not chained onto the
  previous sample. Chaining would compound the turn approximation and drift off
  the true arc.

`trajectory_step` must be at least 0.02 s and no larger than the horizon, and a
trajectory is capped at 200 points.

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
pytest                                   # 157 tests
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

