"""
CMU Motion Capture (BVH) Loader.

Reads BVH motion-capture files from the CMU Graphics Lab Motion Capture Database
(cgspeed BVH conversion) and reduces them to the root-joint trajectory used by
the prediction pipeline.

Why BVH rather than the earlier Human3.6M export:
    BVH files state their own skeleton hierarchy, bone offsets, frame count and
    frame time. The root joint carries explicit global X/Y/Z position channels,
    so the trajectory is stated by the file rather than inferred. This makes the
    data verifiable instead of assumption-dependent.

BVH structure:
    HIERARCHY
        ROOT Hips
            CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
            JOINT ...  (bone offsets in centimetres)
    MOTION
        Frames: <N>
        Frame Time: <seconds>
        <N lines of whitespace-separated channel values>

Only the root joint's translation is extracted here. Joint rotations are
retained in `rotations` for the behavioural layer (Phase C), which will need
pose orientation rather than just global position.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from physics import derive_motion

DEFAULT_DATA_ROOT: Path = Path(__file__).resolve().parent / "training_data" / "cmu_mocap"

# The cgspeed BVH conversion scales the skeleton so that one raw unit equals
# 5 centimetres (0.05 m). This was verified against anatomy: the left leg chain
# (UpLeg -> Leg -> Foot) measures 17.01 raw units, and a human hip-to-ankle
# length is ~0.85 m, giving 0.85 / 17.01 = 0.05 m per unit. Interpreting these
# values as plain centimetres understates every speed and distance by 5x.
METRES_PER_UNIT: float = 0.05

# The root joint name used across the CMU skeleton.
ROOT_JOINT_NAME: str = "Hips"


@dataclass(frozen=True)
class MotionClip:
    """
    A motion-capture clip reduced to the quantities the pipeline consumes.

    Attributes:
        path            source BVH file
        subject         subject/trial identifier, e.g. "007/07_01"
        positions       (frames, 3) root-joint world positions in metres
        velocities      (frames, 3) finite-difference root velocities in m/s
        accelerations   (frames, 3) finite-difference root accelerations in m/s^2
        rotations       (frames, n_rot) raw root channel values after position,
                        retained for pose/orientation features
        sample_dt       seconds between consecutive frames
        frame_count     number of frames
        frame_range     (start, end) window this clip covers within the source
                        file, or None for the whole file. Sub-clips need this so
                        that joint data read from disk is sliced consistently
                        with the stored root trajectory; otherwise forward
                        kinematics would return the full clip's joints and the
                        two would silently disagree in length.
    """

    path: Path
    subject: str
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    rotations: np.ndarray
    sample_dt: float
    frame_range: tuple[int, int] | None = None

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def duration(self) -> float:
        return self.frame_count * self.sample_dt

    @property
    def fps(self) -> float:
        return 1.0 / self.sample_dt


def _read_bvh_motion(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Parses a BVH file's MOTION section.

    Returns (motion_array, frame_time, root_channel_indices) where root_channel_indices
    are the column positions of the root's X/Y/Z position channels within each frame.
    """
    header: list[str] = []
    motion_lines: list[str] = []
    in_motion = False

    with path.open("r", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_motion:
                if stripped.upper().startswith("MOTION"):
                    in_motion = True
                    continue
                header.append(stripped)
            else:
                if not stripped:
                    continue
                motion_lines.append(stripped)

    if not motion_lines:
        raise ValueError(f"{path.name}: no MOTION data found.")

    # First two motion lines are the "Frames:" and "Frame Time:" declarations.
    frame_count = int(motion_lines[0].split()[-1])
    frame_time = float(motion_lines[1].split()[-1])
    data_lines = motion_lines[2:]

    if len(data_lines) != frame_count:
        raise ValueError(
            f"{path.name}: declared {frame_count} frames but found "
            f"{len(data_lines)} data lines."
        )

    motion = np.array(
        [[float(value) for value in line.split()] for line in data_lines],
        dtype=np.float64,
    )

    # Locate the ROOT's position channels in the header to find their columns.
    root_position_columns = _find_root_translation_channels(header, path)
    return motion, frame_time, root_position_columns


def _find_root_translation_channels(
    header: list[str], path: Path
) -> np.ndarray:
    """
    Determines which motion columns hold the root joint's global position.

    Walks the HIERARCHY in order, accumulating each joint's channel count until
    the ROOT joint's channels are found. The root is always first, so its
    position channels occupy the leading columns — but the code verifies the
    channel names rather than assuming their order.
    """
    channel_names: list[str] | None = None

    for index, line in enumerate(header):
        if line.startswith("CHANNELS"):
            tokens = line.split()
            count = int(tokens[1])
            names = tokens[2 : 2 + count]
            # The first CHANNELS line belongs to the ROOT joint.
            channel_names = names
            break

    if channel_names is None:
        raise ValueError(f"{path.name}: no CHANNELS declaration found in HIERARCHY.")

    position_columns = [
        i
        for i, name in enumerate(channel_names)
        if name.lower() in {"xposition", "yposition", "zposition"}
    ]

    if len(position_columns) != 3:
        raise ValueError(
            f"{path.name}: expected 3 root position channels, found "
            f"{len(position_columns)} ({channel_names})."
        )

    return np.array(position_columns, dtype=np.int64)


def load_clip(path: Path) -> MotionClip:
    """
    Loads one BVH clip and derives root-joint kinematics.

    Root positions are converted from centimetres to metres. Velocity and
    acceleration are finite differences of position, matching how the live
    pipeline must infer motion from position observations alone.
    """
    motion, frame_time, position_columns = _read_bvh_motion(path)

    # Extract and convert the root trajectory to metres.
    positions = motion[:, position_columns] * METRES_PER_UNIT
    positions = np.ascontiguousarray(positions, dtype=np.float64)

    velocities, accelerations = derive_motion(positions, frame_time)

    # Remaining channels (root rotations, then all descendant joints) are kept
    # for behavioural features; position columns are dropped to avoid duplication.
    mask = np.ones(motion.shape[1], dtype=bool)
    mask[position_columns] = False
    rotations = np.ascontiguousarray(motion[:, mask], dtype=np.float64)

    subject = f"{path.parent.name}/{path.stem}"

    return MotionClip(
        path=path,
        subject=subject,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        rotations=rotations,
        sample_dt=frame_time,
    )


def iter_clip_paths(
    data_root: Path = DEFAULT_DATA_ROOT,
    subjects: tuple[str, ...] | None = None,
) -> Iterator[Path]:
    """Yields BVH clip paths, optionally filtered by subject directory."""
    if not data_root.exists():
        raise FileNotFoundError(
            f"Motion capture data not found at {data_root}. "
            "Place CMU BVH files under training_data/cmu_mocap/<subject>/."
        )

    for path in sorted(data_root.glob("*/*.bvh")):
        if subjects is not None and path.parent.name not in subjects:
            continue
        yield path


def load_dataset(
    data_root: Path = DEFAULT_DATA_ROOT,
    subjects: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[MotionClip]:
    """Loads multiple clips, optionally capped by `limit`."""
    clips: list[MotionClip] = []
    for index, path in enumerate(iter_clip_paths(data_root, subjects)):
        if limit is not None and index >= limit:
            break
        clips.append(load_clip(path))
    return clips


__all__ = [
    "DEFAULT_DATA_ROOT",
    "ROOT_JOINT_NAME",
    "MotionClip",
    "load_clip",
    "load_dataset",
    "iter_clip_paths",
]
