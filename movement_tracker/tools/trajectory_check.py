"""Measure trajectory accuracy at every step against real mocap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import evaluate as ev
from mocap import iter_clip_paths, load_clip
from schema import TelemetryInput

# Horizons verified. Capped at 1.0 s because only the seated clips are long
# enough to verify 2 s projections, and seated motion is unrepresentative.
STEPS = [0.1, 0.25, 0.5, 0.75, 1.0]
NOISE = 0.02


def run(subject: str):
    clip = load_clip([p for p in iter_clip_paths() if p.parent.name == subject][0])
    rng = np.random.default_rng(0)
    noisy = clip.positions + rng.normal(0, NOISE, clip.positions.shape)
    tracker = ev.EntityTracker(filter_enabled=True)

    errors = {h: [] for h in STEPS}
    stds = {h: [] for h in STEPS}
    inside = {h: 0 for h in STEPS}
    counts = {h: 0 for h in STEPS}

    # Keep only horizons the clip is long enough to verify.
    span = int(max(STEPS) / clip.sample_dt)
    last = clip.frame_count - span
    if last <= 80:
        return None

    for i in range(80, last):
        out = tracker.process_telemetry(
            TelemetryInput(
                entity_id="t",
                x=float(noisy[i, 0]),
                y=float(noisy[i, 1]),
                z=float(noisy[i, 2]),
                timestamp=float(i) * clip.sample_dt,
            ),
            dt=max(STEPS),
            trajectory=True,
            trajectory_step=0.1,
        )
        by_horizon = {round(p.horizon_seconds, 3): p for p in out.trajectory}
        for h in STEPS:
            point = by_horizon.get(round(h, 3))
            if point is None:
                continue
            k = int(round(h / clip.sample_dt))
            if i + k >= clip.frame_count:
                continue
            truth = clip.positions[i + k]
            err = float(
                np.linalg.norm(
                    [point.x - truth[0], point.y - truth[1], point.z - truth[2]]
                )
            )
            errors[h].append(err)
            stds[h].append(point.position_std)
            counts[h] += 1
            if err <= point.confidence_radius_95:
                inside[h] += 1
    return errors, stds, inside, counts


print(f"{'subject':<6} {'horizon':>8} {'mean err':>10} {'mean std':>10} {'ratio':>7} {'95% cov':>8} {'n':>6}")
print("-" * 62)
for subject in ["007", "008", "013"]:
    result = run(subject)
    if result is None:
        print(f"{subject:<6} (clip too short to verify)")
        continue
    errors, stds, inside, counts = result
    for h in STEPS:
        if not errors[h]:
            continue
        e = float(np.mean(errors[h]))
        s = float(np.mean(stds[h]))
        cov = 100.0 * inside[h] / counts[h]
        ratio = e / s if s > 0 else float("inf")
        print(f"{subject:<6} {h:>8.2f} {e:>10.4f} {s:>10.4f} {ratio:>7.3f} {cov:>7.1f}% {counts[h]:>6}")
