"""
Motion Feature Extraction for Behaviour Recognition.

Turns a motion-capture clip into a fixed-length feature vector describing HOW
the body is moving, suitable for an action classifier.

Features are chosen to separate the behaviours actually present in the data
(walking, running, sitting, climbing, jumping, gesturing) rather than to be
generic. Each captures a distinct physical property:

    speed_*        locomotion magnitude
    vertical_*     how much the body rises and falls (jumping, climbing, sitting)
    turning_*      how much the heading changes (walking a curve vs straight line)
    cadence_*      rhythm of the motion (stepping vs holding still)
    posture_*      body configuration, especially hip and hand height

All features are scale-free where possible (ratios, coefficients of variation)
so that a tall and a short subject produce comparable vectors. Features are
computed with numpy only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mocap import MotionClip
from physics import derive_motion, yaw_rate

# Imported at runtime, not under TYPE_CHECKING. The return annotation of
# load_joint_positions names Skeleton, and a TYPE_CHECKING-only import leaves that
# name undefined at runtime, so typing.get_type_hints() raises NameError even though
# the lint gate passes. skeleton.py imports nothing from this module, so there is no
# cycle to avoid.
from skeleton import Skeleton

# Joint indices in the CMU skeleton, verified by parsing the BVH hierarchy
# directly rather than assumed. The cgspeed conversion defines 31 joints in this
# order, beginning Hips, LHipJoint, LeftUpLeg, LeftLeg, LeftFoot, ...
#
# Getting these wrong is silent and damaging: an earlier draft used index 6 for
# the left foot (actually RHipJoint) and index 12 for the right foot (actually
# Spine), which would have fed hip and spine heights to the classifier labelled
# as feet.
JOINT_ORDER: tuple[str, ...] = (
    "Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftFingerBase", "LeftHandIndex1", "LThumb",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightFingerBase", "RightHandIndex1", "RThumb",
)
N_JOINTS: int = len(JOINT_ORDER)

ROOT_JOINT: int = JOINT_ORDER.index("Hips")
LEFT_FOOT: int = JOINT_ORDER.index("LeftFoot")
RIGHT_FOOT: int = JOINT_ORDER.index("RightFoot")
LEFT_HAND: int = JOINT_ORDER.index("LeftHand")
RIGHT_HAND: int = JOINT_ORDER.index("RightHand")
HEAD: int = JOINT_ORDER.index("Head")

# Feature names, in the order produced by `extract_features`.
FEATURE_NAMES: tuple[str, ...] = (
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_cv",
    "vertical_range",
    "vertical_rate",
    "vertical_cv",
    "turning_mean_abs",
    "turning_std",
    "turning_fraction",
    "cadence",
    "foot_height_mean",
    "hip_height_mean",
    "hip_height_range",
    "hand_height_mean",
    "hand_height_range",
    "posture_spread",
    "hand_foot_gap",
)


@dataclass(frozen=True)
class ClipFeatures:
    """Feature vector for one clip, with the label metadata alongside."""

    clip_id: str
    subject: str
    label: str
    values: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, (float(v) for v in self.values)))


# Forward kinematics is the dominant cost in feature extraction: it parses the
# hierarchy and composes 31 joints across every frame. Windowed analysis calls it
# once per window over the SAME file, so the full-file result is cached and
# sliced per window instead of recomputed.
_FK_CACHE: dict[Path, tuple[np.ndarray, object]] = {}


def load_joint_positions(
    clip: "MotionClip | Path",
) -> tuple[np.ndarray, "Skeleton"]:
    """
    Recovers 3D world positions for every joint of a clip, in metres.

    BVH stores only ROTATIONS for non-root joints, so positions must be derived
    by forward kinematics rather than read from the file. Doing this correctly
    required resolving three format details, each verified empirically:

      * the cgspeed files define 31 joints, and only the root carries position
        channels (96 channels total: 6 for the root, 3 for each other joint)
      * rotations are stored in DEGREES, not radians
      * a child's orientation is composed with its parent's, so world rotations
        accumulate down the chain

    Returns (positions, skeleton) with positions shaped (frames, joints, 3).
    """
    from skeleton import load_joint_positions as _fk

    path = clip.path if isinstance(clip, MotionClip) else Path(clip)

    cached = _FK_CACHE.get(path)
    if cached is None:
        full_positions, skeleton = _fk(path, metres=True)
        _FK_CACHE[path] = (full_positions, skeleton)
    else:
        full_positions, skeleton = cached

    positions = full_positions

    # Honour a sub-clip's frame window, so joint data lines up with the root
    # trajectory the clip carries. Skipping this would silently mismatch lengths.
    if isinstance(clip, MotionClip) and clip.frame_range is not None:
        start, end = clip.frame_range
        positions = positions[start:end]

    return positions, skeleton


def _joint_heights(positions: np.ndarray, joint: int) -> np.ndarray:
    """Vertical coordinate of one joint across all frames, in metres."""
    return positions[:, joint, 1]


def _safe_cv(values: np.ndarray) -> float:
    """Coefficient of variation, guarding against division by ~zero."""
    mean = float(np.mean(np.abs(values)))
    if mean < 1e-6:
        return 0.0
    return float(np.std(values) / mean)


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """
    Fraction of samples where the signal changes direction about its mean.

    Used as a cadence proxy: periodic motion such as stepping or jumping crosses
    its own mean far more often than a sustained movement.
    """
    if signal.size < 3:
        return 0.0
    centred = signal - np.mean(signal)
    signs = np.sign(centred)
    signs[signs == 0] = 1
    crossings = np.count_nonzero(np.diff(signs) != 0)
    return float(crossings / (signal.size - 1))


def features_from_arrays(
    root_positions: np.ndarray,
    joint_positions: np.ndarray,
    sample_dt: float,
) -> np.ndarray:
    """
    Computes the feature vector from raw motion arrays.

    This is the single implementation shared by training (which reads BVH files)
    and serving (which receives joints over the API). Keeping one code path is
    essential: if the two computed features even slightly differently, the model
    would be evaluated on inputs unlike those it was trained on.

    Args:
        root_positions  (frames, 3) root/hip trajectory in metres
        joint_positions (frames, joints, 3) joint positions in metres
        sample_dt       seconds between frames

    Returns an (n_features,) array matching FEATURE_NAMES.
    """
    root_positions = np.asarray(root_positions, dtype=np.float64)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)

    if root_positions.ndim != 2 or root_positions.shape[1] != 3:
        raise ValueError(
            f"root_positions must be (frames, 3), got {root_positions.shape}"
        )
    if joint_positions.ndim != 3 or joint_positions.shape[2] != 3:
        raise ValueError(
            f"joint_positions must be (frames, joints, 3), got {joint_positions.shape}"
        )
    if joint_positions.shape[1] != N_JOINTS:
        raise ValueError(
            f"expected {N_JOINTS} joints, got {joint_positions.shape[1]}. "
            "Supply joints in the CMU order or map them by name first."
        )
    if root_positions.shape[0] != joint_positions.shape[0]:
        raise ValueError(
            f"frame count mismatch: root has {root_positions.shape[0]}, "
            f"joints have {joint_positions.shape[0]}"
        )
    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive")

    dt = float(sample_dt)
    velocities, _accelerations = derive_motion(root_positions, dt)
    speed = np.linalg.norm(velocities, axis=1)

    vertical = root_positions[:, 1]
    vertical_velocity = np.gradient(vertical, dt)

    turn = yaw_rate(velocities, dt)
    turning_fraction = float(np.mean(np.abs(turn) > 0.2)) if turn.size else 0.0

    foot_height = np.minimum(
        joint_positions[:, LEFT_FOOT, 1], joint_positions[:, RIGHT_FOOT, 1]
    )
    hand_height = np.mean(
        [joint_positions[:, LEFT_HAND, 1], joint_positions[:, RIGHT_HAND, 1]], axis=0
    )
    hip_height = joint_positions[:, ROOT_JOINT, 1]

    limb_columns = [
        joint_positions[:, j, :]
        for j in (LEFT_FOOT, RIGHT_FOOT, LEFT_HAND, RIGHT_HAND)
    ]
    centre = np.mean(limb_columns, axis=0)
    posture_spread = float(
        np.mean([np.linalg.norm(limb - centre, axis=1).mean() for limb in limb_columns])
    )
    hand_foot_gap = float(np.mean(hand_height - foot_height))

    values = np.array(
        [
            float(np.mean(speed)),
            float(np.std(speed)),
            float(np.max(speed)),
            _safe_cv(speed),
            float(np.ptp(vertical)),
            float(np.mean(np.abs(vertical_velocity))),
            _safe_cv(vertical),
            float(np.mean(np.abs(turn))),
            float(np.std(turn)),
            turning_fraction,
            _zero_crossing_rate(vertical),
            float(np.mean(foot_height)),
            float(np.mean(hip_height)),
            float(np.ptp(hip_height)),
            float(np.mean(hand_height)),
            float(np.ptp(hand_height)),
            posture_spread,
            hand_foot_gap,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite feature value computed")

    return values


def extract_features(clip: MotionClip, label: str) -> ClipFeatures:
    """
    Computes the feature vector for a clip.

    Delegates to `features_from_arrays` so that the training path (reading BVH)
    and the serving path (receiving joints over the API) run identical maths. A
    divergence between the two would mean the model is served inputs unlike those
    it was trained on.
    """
    joints, _skeleton = load_joint_positions(clip)
    values = features_from_arrays(clip.positions, joints, clip.sample_dt)

    return ClipFeatures(
        clip_id=clip.subject,
        subject=clip.subject.split("/")[0],
        label=label,
        values=values,
    )


__all__ = [
    "FEATURE_NAMES",
    "ROOT_JOINT",
    "ClipFeatures",
    "extract_features",
    "load_joint_positions",
]
