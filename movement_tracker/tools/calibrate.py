"""Measure the reported uncertainty against observed error on real mocap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import evaluate as ev
from mocap import iter_clip_paths, load_clip
from schema import TelemetryInput

HORIZONS = [0.2, 0.5, 1.0, 2.0]
NOISE = 0.02


def run(subject: str, horizon: float):
    clip = load_clip([p for p in iter_clip_paths() if p.parent.name == subject][0])
    hs = int(round(horizon / clip.sample_dt))
    if hs < 1:
        return None
    rng = np.random.default_rng(0)
    noisy = clip.positions + rng.normal(0, NOISE, clip.positions.shape)
    trk = ev.EntityTracker(filter_enabled=True)

    errs, stds, inside, total = [], [], 0, 0
    for i in range(60, min(900, clip.frame_count - hs)):
        out = trk.process_telemetry(
            TelemetryInput(
                entity_id="e",
                x=float(noisy[i, 0]),
                y=float(noisy[i, 1]),
                z=float(noisy[i, 2]),
                timestamp=float(i) * clip.sample_dt,
            ),
            dt=horizon,
        )
        pred = np.array([out.predicted_x, out.predicted_y, out.predicted_z])
        err = float(np.linalg.norm(pred - clip.positions[i + hs]))
        errs.append(err)
        stds.append(out.position_std)
        total += 1
        if err <= out.confidence_radius_95:
            inside += 1
    return float(np.mean(errs)), float(np.mean(stds)), 100.0 * inside / total


print(f"{'subject':<6} {'horizon':>7} {'actual err':>11} {'reported std':>13} {'ratio':>7} {'95% cov':>8}")
print("-" * 60)
for subject in ["007", "013"]:
    for horizon in HORIZONS:
        result = run(subject, horizon)
        if result is None:
            continue
        err, std, cov = result
        ratio = err / std if std > 0 else float("inf")
        print(
            f"{subject:<6} {horizon:>7.2f} {err:>11.4f} {std:>13.4f} {ratio:>7.3f} {cov:>7.1f}%"
        )
