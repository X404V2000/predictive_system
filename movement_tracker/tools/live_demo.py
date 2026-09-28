#!/usr/bin/env python3
"""
Live Movement Prediction Demo.

Runs the complete pipeline continuously and prints what each layer concludes:

    pose source -> trajectory prediction (WHERE)  -> action recognition (WHAT)

Two layers are shown because they answer different questions and need different
inputs:

  * **Trajectory** consumes the HIP trajectory only. It needs no skeleton and so
    works with any motion source, including a simple tracked point.
  * **Action** consumes the full skeleton over a window. It reports nothing until
    enough frames have accumulated, because speed, cadence and turn rate are
    meaningless for a single instant.

Usage:
    python3 tools/live_demo.py                          # synthetic walker
    python3 tools/live_demo.py --source mocap --clip 013/13_05
    python3 tools/live_demo.py --speed 2.0 --frames 300
    python3 tools/live_demo.py --json                   # machine-readable lines

The pose source is pluggable. A real webcam pose estimator can be added by
implementing the PoseSource interface in pose.py; nothing in this loop changes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from features import ROOT_JOINT, features_from_arrays
from pose import MocapPlaybackSource, PoseSourceError, SyntheticPoseSource
from schema import TelemetryInput
from tracker import EntityTracker

# Action recognition needs a window of motion. Two seconds matches the window the
# model was trained on; using a different duration would feed it inputs unlike
# its training data.
ACTION_WINDOW_SECONDS: float = 2.0

# Blend over this many recent frames when estimating the hip's velocity so the
# prediction is not driven by single-frame jitter.
VELOCITY_WINDOW: int = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live movement prediction demo.")
    parser.add_argument(
        "--source",
        choices=("synthetic", "mocap"),
        default="synthetic",
        help="where pose comes from (default: synthetic)",
    )
    parser.add_argument(
        "--clip",
        default="013/13_05",
        help="subject/trial to replay when --source mocap (default: 013/13_05)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second")
    parser.add_argument(
        "--speed", type=float, default=1.2, help="synthetic walking speed in m/s"
    )
    parser.add_argument(
        "--horizon", type=float, default=0.5, help="prediction horizon in seconds"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=150,
        help="frames to process before stopping; 0 runs until the source ends",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace the loop to the requested FPS instead of running flat out",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON lines")
    return parser.parse_args()


def build_source(args: argparse.Namespace):
    if args.source == "mocap":
        from mocap import DEFAULT_DATA_ROOT

        subject, name = args.clip.split("/")
        path = DEFAULT_DATA_ROOT / subject / f"{name}.bvh"
        if not path.exists():
            raise PoseSourceError(f"clip not found: {path}")
        return MocapPlaybackSource(path=path, speed_multiplier=1.0, loop=True)

    return SyntheticPoseSource(fps=args.fps, speed=args.speed)


def format_line(
    index: int,
    timestamp: float,
    predicted: tuple[float, float, float],
    current: tuple[float, float, float],
    displacement: float,
    std: float,
    motion_model: str,
    warming_up: bool,
    action: str | None,
    confidence: float | None,
    low_confidence: bool,
) -> str:
    """
    Renders one status line.

    The distance column is how far the target is projected to TRAVEL, not an
    error. A live source has no future reference, so an "error" here would only
    measure the horizon distance itself (about 0.6 m for a walker at 0.5 s), which
    looks alarming while carrying no information. Accuracy is measured properly in
    tools/benchmark.py against recorded motion.
    """
    action_text = (
        f"{action:<11} p={confidence:.2f}{' (!)' if low_confidence else '    '}"
        if action
        else f"{'collecting...':<11}          "
    )
    status = "warmup" if warming_up else "      "
    return (
        f"[{index:5d}] t={timestamp:6.2f}s {status} "
        f"now=({current[0]:7.3f},{current[1]:6.3f},{current[2]:7.3f}) "
        f"ahead={displacement:5.3f}m sd={std:5.3f} "
        f"{motion_model:<20} | {action_text}"
    )


def main() -> int:
    args = parse_args()

    try:
        source = build_source(args)
        source.start()
    except PoseSourceError as exc:
        print(f"[FAIL] {exc}")
        return 1

    # The tracker is keyed by entity, so a single "live" entity is used here. A
    # multi-person source would supply one id per person and get one EKF each.
    tracker = EntityTracker(filter_enabled=True, vertical_axis=1)
    entity_id = "live"

    try:
        from actions import LOW_CONFIDENCE_THRESHOLD, train_default_model

        action_model = train_default_model()
    except Exception as exc:
        print(f"[warn] action model unavailable ({exc}); showing trajectory only")
        action_model = None

    if not args.json:
        print("Live movement prediction")
        print(f"  source   : {source.name}")
        print(f"  horizon  : {args.horizon}s")
        print(f"  action   : 2.0s window, "
              f"{'enabled' if action_model else 'DISABLED'}")
        print()
        print(
            f"{'frame':<7}{'time':<8}{'now (x,y,z)':<28}"
            f"{'ahead':<8}{'sigma':<9}{'motion model':<22}| action"
        )
        print("-" * 118)

    action_buffer: list[np.ndarray] = []
    action_frame_dt = 1.0 / max(args.fps, 1e-6)
    window_frames = max(2, int(round(ACTION_WINDOW_SECONDS * args.fps)))

    # Velocity is derived here from consecutive hip positions rather than taken
    # from the source, because the source contract only guarantees positions.
    previous_hip: np.ndarray | None = None

    index = 0
    settled_sigmas: list[float] = []
    low_confidence_count = 0
    action_calls = 0

    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            hip = frame.positions[ROOT_JOINT]
            if previous_hip is None:
                velocity = np.zeros(3)
            else:
                velocity = (hip - previous_hip) / action_frame_dt
            previous_hip = hip.copy()

            # Telemetry carries the hip position and its measured velocity. The
            # tracker filters the position and re-derives its own state; the
            # velocity here only seeds the first sample.
            prediction = tracker.process_telemetry(
                TelemetryInput(
                    entity_id=entity_id,
                    x=float(hip[0]),
                    y=float(hip[1]),
                    z=float(hip[2]),
                    vx=float(velocity[0]),
                    vy=float(velocity[1]),
                    vz=float(velocity[2]),
                    timestamp=frame.timestamp,
                ),
                dt=args.horizon,
            )

            # Ground truth for the error column: where the hip actually is at the
            # horizon. Only available because this is a replay; a live camera has
            # no such reference, so the column is informational.
            action_buffer.append(frame.positions)
            if len(action_buffer) > window_frames:
                action_buffer.pop(0)

            action_name: str | None = None
            confidence: float | None = None
            low_confidence = False

            if action_model is not None and len(action_buffer) == window_frames:
                window = np.stack(action_buffer)
                values = features_from_arrays(
                    window[:, ROOT_JOINT, :], window, action_frame_dt
                )
                probabilities = action_model.predict_proba(values)
                best = int(np.argmax(probabilities))
                action_name = action_model.classes[best]
                confidence = float(probabilities[best])
                low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
                action_calls += 1
                low_confidence_count += int(low_confidence)

            predicted = (
                prediction.predicted_x,
                prediction.predicted_y,
                prediction.predicted_z,
            )
            displacement = float(
                np.linalg.norm(np.asarray(predicted) - np.asarray(hip))
            )

            if args.json:
                print(
                    json.dumps(
                        {
                            "frame": index,
                            "timestamp": round(frame.timestamp, 4),
                            "source": frame.source,
                            "predicted": [round(v, 4) for v in predicted],
                            "horizon_seconds": prediction.horizon_seconds,
                            "sigma": round(prediction.position_std, 4),
                            "confidence_radius_95": round(
                                prediction.confidence_radius_95, 4
                            ),
                            "motion_model": prediction.motion_model,
                            "warming_up": prediction.warming_up,
                            "action": action_name,
                            "action_confidence": (
                                round(confidence, 4) if confidence is not None else None
                            ),
                            "low_confidence": low_confidence,
                        }
                    ),
                    flush=True,
                )
            else:
                print(
                    format_line(
                        index,
                        frame.timestamp,
                        predicted,
                        tuple(hip),
                        displacement,
                        prediction.position_std,
                        prediction.motion_model,
                        prediction.warming_up,
                        action_name,
                        confidence,
                        low_confidence,
                    ),
                    flush=True,
                )

            if not prediction.warming_up:
                settled_sigmas.append(float(prediction.position_std))

            index += 1
            if args.frames and index >= args.frames:
                break
            if args.realtime:
                time.sleep(max(0.0, 1.0 / max(args.fps, 1e-6)))

    except KeyboardInterrupt:
        print("\n[stopped]")
    except PoseSourceError as exc:
        print(f"\n[FAIL] {exc}")
        return 1
    finally:
        source.close()

    if not args.json:
        print("-" * 118)
        print(f"frames processed      : {index}")
        print(f"action classifications: {action_calls}")
        if action_calls:
            print(
                f"  low confidence       : {low_confidence_count} "
                f"({100 * low_confidence_count / action_calls:.0f}%)"
            )
        if settled_sigmas:
            print(
                f"mean reported sigma   : {float(np.mean(settled_sigmas)):.4f} m "
                "(settled frames only)"
            )
        print()
        print("Note: 'ahead' is how far the target is projected to travel, not an")
        print("error. A live source has no future reference, so accuracy is measured")
        print("in tools/benchmark.py against recorded motion instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
