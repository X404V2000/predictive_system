"""
Trajectory Prediction Evaluation Metrics and Harness.

Replays real motion-capture trajectories through the prediction pipeline and
scores the result against ground truth and naive baselines.

Metrics follow the standard trajectory-prediction definitions:
    ADE  Average Displacement Error — mean Euclidean distance between predicted
         and true position across all evaluated time steps.
    FDE  Final Displacement Error — Euclidean distance at the final horizon.

Two baselines provide context, because a raw error number means little alone:
    static   assume the target does not move (error grows with speed)
    naive    extrapolate the raw, unfiltered telemetry forward kinematically

A prediction is only useful if it beats both.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mocap import MotionClip
from schema import TelemetryInput
from tracker import EntityTracker


class Baseline(str, Enum):
    """Reference predictors to compare the filtered pipeline against."""

    STATIC = "static"
    NAIVE = "naive"


@dataclass
class EvaluationResult:
    """Aggregated accuracy for one predictor over one or more clips."""

    predictor: str
    horizon: float
    ade: float
    fde: float
    samples: int
    per_action: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.predictor:<12} horizon={self.horizon:.2f}s  "
            f"ADE={self.ade:6.3f} m  FDE={self.fde:6.3f} m  n={self.samples}"
        )


def _static_prediction(positions: np.ndarray, index: int, horizon_steps: int) -> np.ndarray:
    """Baseline: the target is assumed to remain where it is."""
    return positions[index]


def _naive_prediction(
    noisy_positions: np.ndarray, index: int, dt: float, horizon: float
) -> np.ndarray:
    """
    Baseline: extrapolate position with a raw finite-difference velocity, with no
    filtering.

    Operates on the same noisy observation stream the filter receives, so the
    comparison isolates the effect of filtering rather than of noise.
    """
    current = noisy_positions[index]
    if index < 1:
        return current
    velocity = (current - noisy_positions[index - 1]) / dt
    return current + velocity * horizon


def _pipeline_prediction(
    tracker: EntityTracker,
    clip: MotionClip,
    index: int,
    horizon: float,
    noisy_position: np.ndarray,
    previous_noisy: np.ndarray | None,
) -> np.ndarray:
    """
    Runs one telemetry frame through the real pipeline and returns the predicted
    position.

    Only the POSITION is observed. Velocity is derived by finite difference of
    the noiseless observation stream, and acceleration is left unknown, because a
    position-sensing system has no privileged access to clean derivatives.
    Supplying clean velocity would hand the filter information no real sensor
    provides, and would flatter the filter in the benchmark.
    """
    noisy = noisy_position

    if previous_noisy is None:
        velocity = np.zeros(3, dtype=np.float64)
    else:
        velocity = (noisy - previous_noisy) / clip.sample_dt

    telemetry = TelemetryInput(
        entity_id="eval_target",
        x=float(noisy[0]),
        y=float(noisy[1]),
        z=float(noisy[2]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
        vz=float(velocity[2]),
        timestamp=float(index) * clip.sample_dt,
    )

    prediction = tracker.process_telemetry(telemetry, dt=horizon)
    return np.array(
        [prediction.predicted_x, prediction.predicted_y, prediction.predicted_z]
    )


def evaluate_clip(
    clip: MotionClip,
    horizon: float = 0.5,
    noise_std: float = 0.02,
    seed: int = 0,
    warmup: int = 20,
) -> dict[str, tuple[float, float]]:
    """
    Evaluates all predictors on a single clip.

    Returns a mapping of predictor name -> (ADE, FDE) in metres.
    """
    dt = clip.sample_dt
    horizon_steps = int(round(horizon / dt))
    if horizon_steps < 1:
        raise ValueError("Horizon is shorter than one sample interval.")

    rng = np.random.default_rng(seed)

    filtered = EntityTracker(filter_enabled=True)

    errors: dict[str, list[float]] = {
        "static": [],
        "naive": [],
        "filtered": [],
    }

    # One shared noisy observation stream, so every predictor sees identical
    # measurements. Comparing predictors fed different data would be meaningless.
    noisy_positions = clip.positions + rng.normal(
        0.0, noise_std, size=clip.positions.shape
    )

    for index in range(warmup, clip.frame_count - horizon_steps):
        truth = clip.positions[index + horizon_steps]

        previous_noisy = noisy_positions[index - 1] if index >= 1 else None

        predictions = {
            "static": _static_prediction(noisy_positions, index, horizon_steps),
            "naive": _naive_prediction(noisy_positions, index, dt, horizon),
            "filtered": _pipeline_prediction(
                filtered, clip, index, horizon, noisy_positions[index], previous_noisy
            ),
        }

        for name, predicted in predictions.items():
            errors[name].append(float(np.linalg.norm(predicted - truth)))

    results: dict[str, tuple[float, float]] = {}
    for name, series in errors.items():
        if not series:
            results[name] = (float("nan"), float("nan"))
            continue
        results[name] = (float(np.mean(series)), float(series[-1]))
    return results


def evaluate_dataset(
    clips: list[MotionClip],
    horizon: float = 0.5,
    noise_std: float = 0.02,
    seed: int = 0,
    actions: dict[str, str] | None = None,
) -> dict[str, EvaluationResult]:
    """
    Evaluates every predictor across a set of clips and aggregates ADE/FDE.

    `actions` optionally maps a clip subject id to an action label, enabling
    per-action breakdowns.
    """
    per_predictor: dict[str, list[float]] = {"static": [], "naive": [], "filtered": []}
    per_predictor_fde: dict[str, list[float]] = {
        "static": [],
        "naive": [],
        "filtered": [],
    }
    per_action: dict[str, dict[str, list[float]]] = {}

    for clip_index, clip in enumerate(clips):
        scores = evaluate_clip(clip, horizon, noise_std, seed=seed + clip_index)
        for name, (ade, fde) in scores.items():
            if np.isfinite(ade):
                per_predictor[name].append(ade)
                per_predictor_fde[name].append(fde)
            if actions is not None:
                label = actions.get(clip.subject, clip.subject)
                per_action.setdefault(label, {}).setdefault(name, []).append(ade)

    results: dict[str, EvaluationResult] = {}
    for name in per_predictor:
        series = per_predictor[name]
        if not series:
            continue
        breakdown = {
            label: float(np.mean(values.get(name, [float("nan")])))
            for label, values in per_action.items()
        }
        results[name] = EvaluationResult(
            predictor=name,
            horizon=horizon,
            ade=float(np.mean(series)),
            fde=float(np.mean(per_predictor_fde[name])),
            samples=len(series),
            per_action=breakdown,
        )
    return results


__all__ = [
    "Baseline",
    "EvaluationResult",
    "evaluate_clip",
    "evaluate_dataset",
]
