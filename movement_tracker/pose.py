"""
Pose Source Abstraction.

Decouples the prediction pipeline from HOW skeletal motion is obtained, so the
same live loop works with any source: a webcam pose model, a motion-capture
playback, or a synthetic driver.

Why an abstraction rather than a direct dependency: obtaining full-body pose from
an RGB webcam requires a trained model (MediaPipe, MoveNet, RTMPose and similar)
which is not installed, and the pipeline should not be blocked on that choice.
The live demo drives a synthetic source today, and a real pose estimator can be
added later by implementing one interface.

The contract is deliberately narrow. A source yields skeletons in the SAME joint
order and units the action model was trained on (CMU 31-joint, metres, Y up),
because a source that silently emitted a different layout would produce
plausible-looking but wrong predictions. Sources that natively use another layout
must reorder before yielding.

numpy and the standard library only.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from features import JOINT_ORDER, N_JOINTS


class PoseSourceError(RuntimeError):
    """Raised when a pose source cannot start or deliver frames."""


@dataclass(frozen=True)
class PoseFrame:
    """
    One observation of a skeleton.

    Attributes:
        positions  (joints, 3) joint world positions in METRES
        timestamp  seconds, monotonic within a session
        source     identifier of the driver that produced the frame
    """

    positions: np.ndarray
    timestamp: float
    source: str

    @property
    def joint_count(self) -> int:
        return int(self.positions.shape[0])


@runtime_checkable
class PoseSource(Protocol):
    """Anything that can deliver a stream of skeleton frames."""

    name: str

    def start(self) -> None:
        """Prepares the source. Must be safe to call once."""

    def read(self) -> PoseFrame | None:
        """Returns the next frame, or None when the source is exhausted."""

    def close(self) -> None:
        """Releases any held resources. Must be safe to call twice."""


def validate_frame(frame: PoseFrame) -> None:
    """
    Checks a frame satisfies the contract the action model requires.

    Failing loudly here is important: a wrong joint count or a non-finite value
    would otherwise flow into the feature extractor and surface as a confusing
    error much later, or worse, as a plausible but meaningless prediction.
    """
    if frame.positions.shape != (N_JOINTS, 3):
        raise PoseSourceError(
            f"pose frame must be ({N_JOINTS}, 3), got {tuple(frame.positions.shape)}. "
            "Sources using another skeleton must reorder to the CMU layout first."
        )
    if not np.all(np.isfinite(frame.positions)):
        raise PoseSourceError("pose frame contains non-finite coordinates")


# --- synthetic driver ------------------------------------------------------


@dataclass
class SyntheticPoseSource:
    """
    Generates anatomically plausible walking skeletons over time.

    This exists so the live pipeline can be exercised end to end without a pose
    model, and so its behaviour is deterministic and testable. It is NOT a
    substitute for real pose estimation: it produces a stylised gait, not
    observations of a person.

    The skeleton is built from a fixed rest pose with sinusoidal joint motion, so
    it moves like a walker: hips advance, legs swing out of phase, arms
    counter-swing, and the torso bobs vertically.
    """

    name: str = "synthetic"
    fps: float = 30.0
    speed: float = 1.2  # metres per second of forward travel
    stride_hz: float = 0.9  # gait cycles per second
    duration: float | None = None
    seed: int = 0

    _frames_emitted: int = 0
    _started: bool = False

    # Rest pose heights in metres, from the verified CMU skeleton (Hips first).
    # Only the joints the synthetic gait animates are listed; the rest are placed
    # at a neutral offset so the skeleton has a sensible silhouette.
    _REST_HEIGHTS: tuple[float, ...] = (
        0.90,  # Hips
        0.88, 0.86, 0.48, 0.06, 0.03,       # left leg chain
        0.88, 0.86, 0.48, 0.06, 0.03,       # right leg chain
        0.95, 1.05, 1.15, 1.35, 1.38, 1.55,  # spine to head
        1.34, 1.30, 1.16, 1.02,             # left arm chain
        1.02, 1.02, 1.02,
        1.34, 1.30, 1.16, 1.02,             # right arm chain
        1.02, 1.02, 1.02,
    )

    def start(self) -> None:
        if len(self._REST_HEIGHTS) != N_JOINTS:
            raise PoseSourceError(
                f"rest pose defines {len(self._REST_HEIGHTS)} joints, "
                f"expected {N_JOINTS}"
            )
        self._frames_emitted = 0
        self._started = True

    def close(self) -> None:
        self._started = False

    def read(self) -> PoseFrame | None:
        if not self._started:
            raise PoseSourceError("start() must be called before read()")

        t = self._frames_emitted / self.fps
        if self.duration is not None and t > self.duration:
            return None

        positions = np.zeros((N_JOINTS, 3), dtype=np.float64)
        rest = np.asarray(self._REST_HEIGHTS, dtype=np.float64)

        # Forward travel along +Z, with a small lateral sway.
        forward = self.speed * t
        phase = 2.0 * np.pi * self.stride_hz * t
        bob = 0.02 * np.sin(2.0 * phase)

        positions[:, 1] = rest + bob
        positions[:, 2] = forward

        # Legs swing in antiphase, arms counter-swing. Amplitude shrinks with
        # height so the whole chain rotates plausibly rather than shearing.
        leg_swing = 0.28 * np.sin(phase)
        positions[2:6, 2] += leg_swing       # left leg chain
        positions[7:11, 2] -= leg_swing      # right leg chain
        positions[18:21, 2] -= leg_swing * 0.7  # left arm chain
        positions[25:28, 2] += leg_swing * 0.7  # right arm chain

        # Lateral sway, largest at the hips.
        sway = 0.03 * np.sin(phase)
        positions[:, 0] = sway * (positions[:, 1] / max(rest.max(), 1e-6))

        frame = PoseFrame(
            positions=positions,
            timestamp=t,
            source=self.name,
        )
        validate_frame(frame)

        self._frames_emitted += 1
        return frame


# --- motion-capture playback driver ---------------------------------------


@dataclass
class MocapPlaybackSource:
    """
    Replays a recorded motion-capture clip as a live pose stream.

    Useful for exercising the pipeline against REAL skeletons without a pose
    model, and for regression testing the live loop against known actions.
    """

    path: object  # pathlib.Path to a BVH file
    name: str = "mocap"
    speed_multiplier: float = 1.0
    loop: bool = True

    _positions: np.ndarray | None = None
    _dt: float = 0.0
    _index: int = 0
    _elapsed: float = 0.0

    def start(self) -> None:
        from features import load_joint_positions

        positions, _skeleton = load_joint_positions(self.path)
        self._positions = np.asarray(positions, dtype=np.float64)
        self._dt = 1.0 / 120.0  # CMU cgspeed files are 120 Hz
        self._index = 0
        self._elapsed = 0.0

    def close(self) -> None:
        self._positions = None

    def read(self) -> PoseFrame | None:
        if self._positions is None:
            raise PoseSourceError("start() must be called before read()")

        if self._index >= self._positions.shape[0]:
            if not self.loop:
                return None
            self._index = 0

        frame = PoseFrame(
            positions=self._positions[self._index].copy(),
            timestamp=self._elapsed,
            source=self.name,
        )
        validate_frame(frame)

        self._index += 1
        self._elapsed += self._dt / max(self.speed_multiplier, 1e-6)
        return frame


def reorder_to_cmu(positions: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """
    Reorders a skeleton into the CMU joint order by name.

    Provided so a future pose driver with its own layout can reuse the same
    mapping logic the API uses, rather than inventing another. A missing joint is
    an error, since substituting one silently yields wrong predictions.
    """
    lookup = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in JOINT_ORDER if name not in lookup]
    if missing:
        raise PoseSourceError(
            f"joint_names is missing {len(missing)} required joint(s): "
            f"{', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else "")
        )
    return positions[:, [lookup[name] for name in JOINT_ORDER], :]


__all__ = [
    "MocapPlaybackSource",
    "PoseFrame",
    "PoseSource",
    "PoseSourceError",
    "SyntheticPoseSource",
    "reorder_to_cmu",
    "validate_frame",
]
