"""
BVH Skeleton Parsing and Forward Kinematics.

Recovers 3D joint world positions from a BVH file.

This is necessary rather than optional: in a BVH motion block only the ROOT joint
carries position channels. Every other joint carries three ROTATION channels, and
its position must be derived by composing the hierarchy of bone offsets and
rotations from the root outward (forward kinematics).

Verified layout for the CMU cgspeed files:
    31 joints, 96 channels total
    Hips      : 6 channels (Xposition Yposition Zposition Zrotation Yrotation Xrotation)
    all others: 3 channels (Zrotation Yrotation Xrotation)
Channel order within a joint is Z, then Y, then X, which is the order the
rotations must be composed.

Everything here uses numpy only.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Unit scale of the cgspeed conversion, verified in mocap.py against anatomy.
METRES_PER_UNIT: float = 0.05


@dataclass(frozen=True)
class Joint:
    """One joint of the parsed skeleton."""

    name: str
    parent: int
    offset: np.ndarray  # bone vector to this joint, in the PARENT's local frame
    channel_slice: slice  # columns of the motion block belonging to this joint
    is_root: bool


@dataclass(frozen=True)
class Skeleton:
    """A parsed BVH skeleton: joint topology plus channel layout."""

    joints: tuple[Joint, ...]
    channel_count: int
    frame_time: float
    frame_count: int

    def index_of(self, name: str) -> int:
        for i, joint in enumerate(self.joints):
            if joint.name == name:
                return i
        raise KeyError(f"joint {name!r} not present in skeleton")


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Builds a rotation matrix from Z, Y, X Euler angles in that composition order.

    BVH channel order is Zrotation, Yrotation, Xrotation, and the rotations
    compose in the order the channels are declared, so R = Rz @ Ry @ Rx.
    """
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz_m @ ry_m @ rx_m


def parse_skeleton(path: Path) -> Skeleton:
    """
    Parses the HIERARCHY and MOTION header of a BVH file.

    Returns the joint topology and channel layout, without reading the motion
    data itself, so the skeleton can be reused across frames.
    """
    text = path.read_text(errors="replace")
    if "MOTION" not in text:
        raise ValueError(f"{path.name}: no MOTION section found")

    hierarchy, motion = text.split("MOTION", 1)

    # --- hierarchy -------------------------------------------------------
    names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channel_counts: list[int] = []
    stack: list[int] = []

    # Braces appear both alone and trailing a JOINT line, so they are counted
    # rather than matched per line. Treating every "}" as closing a joint, or
    # missing a trailing "{", silently corrupts the parent links.
    pending_parent: int = -1
    for raw in hierarchy.splitlines():
        line = raw.strip()

        match = re.match(r"(ROOT|JOINT)\s+(\S+)", line)
        if match:
            # A joint's parent is the joint whose block currently encloses it.
            parents.append(stack[-1] if stack else -1)
            names.append(match.group(2))
            offsets.append(np.zeros(3))
            channel_counts.append(0)
            pending_parent = len(names) - 1
            if "{" in line:
                stack.append(pending_parent)
            continue

        if line.startswith("OFFSET") and names:
            offsets[-1] = np.array([float(v) for v in line.split()[1:4]])
            continue
        if line.startswith("CHANNELS") and names:
            channel_counts[-1] = int(line.split()[1])
            continue

        # Account for every brace on the line, in order.
        for char in line:
            if char == "{":
                stack.append(pending_parent)
            elif char == "}":
                if stack:
                    stack.pop()

    if not names:
        raise ValueError(f"{path.name}: no joints parsed from HIERARCHY")

    # --- motion header ---------------------------------------------------
    motion_lines = [ln.strip() for ln in motion.splitlines() if ln.strip()]
    frame_count = int(motion_lines[0].split()[-1])
    frame_time = float(motion_lines[1].split()[-1])

    # Build the channel slice for each joint, in declaration order.
    joints: list[Joint] = []
    cursor = 0
    for index, name in enumerate(names):
        count = channel_counts[index]
        joints.append(
            Joint(
                name=name,
                parent=parents[index],
                offset=offsets[index],
                channel_slice=slice(cursor, cursor + count),
                is_root=(index == 0),
            )
        )
        cursor += count

    return Skeleton(
        joints=tuple(joints),
        channel_count=cursor,
        frame_time=frame_time,
        frame_count=frame_count,
    )


def forward_kinematics(
    skeleton: Skeleton, motion: np.ndarray
) -> np.ndarray:
    """
    Computes world positions for every joint across every frame.

    `motion` is the (frames, channels) numeric block. Returns (frames, joints, 3)
    positions in the same units as the input (i.e. raw BVH units; multiply by
    METRES_PER_UNIT for metres).

    The root's position channels give its world location directly. Every other
    joint is reached by walking outward from the root, composing each bone's
    rotation and translating by its offset.
    """
    frames = motion.shape[0]
    n_joints = len(skeleton.joints)

    rotations = np.zeros((frames, n_joints, 3, 3), dtype=np.float64)
    local_offsets = np.zeros((frames, n_joints, 3), dtype=np.float64)
    world = np.zeros((frames, n_joints, 3), dtype=np.float64)

    for index, joint in enumerate(skeleton.joints):
        block = motion[:, joint.channel_slice]

        if joint.is_root:
            # 6 channels: position (x, y, z) then Euler rotations (z, y, x).
            local_offsets[:, index, :] = block[:, 0:3]
            euler = block[:, 3:6]
        else:
            # Joints translate by their fixed bone offset in the parent frame.
            local_offsets[:, index, :] = joint.offset
            euler = block[:, 0:3]

        # The cgspeed conversion stores rotations in DEGREES, not radians: values
        # span roughly -90..+90 and reach magnitudes (e.g. 21) that are
        # impossible for radians. Feeding them straight into cos/sin misplaces
        # every joint away from the root.
        euler = np.radians(euler)

        # Vectorised Euler -> matrix over all frames.
        rz, ry, rx = euler[:, 0], euler[:, 1], euler[:, 2]
        cz, sz = np.cos(rz), np.sin(rz)
        cy, sy = np.cos(ry), np.sin(ry)
        cx, sx = np.cos(rx), np.sin(rx)

        rotations[:, index, 0, 0] = cz * cy
        rotations[:, index, 0, 1] = cz * sy * sx - sz * cx
        rotations[:, index, 0, 2] = cz * sy * cx + sz * sx
        rotations[:, index, 1, 0] = sz * cy
        rotations[:, index, 1, 1] = sz * sy * sx + cz * cx
        rotations[:, index, 1, 2] = sz * sy * cx - cz * sx
        rotations[:, index, 2, 0] = -sy
        rotations[:, index, 2, 1] = cy * sx
        rotations[:, index, 2, 2] = cy * cx

    # Compose outward, accumulating the parent's WORLD rotation. Composing each
    # joint's local rotation in isolation is wrong: a child's orientation is
    # relative to its parent, so world_rotation[child] = world_rotation[parent]
    # @ local_rotation[child]. Using local rotations alone placed the head below
    # the hips, because the spine's upward bone offsets were never inherited.
    #
    # The parsed order guarantees a parent appears before its children, so one
    # forward pass suffices.
    world_rotations = np.zeros_like(rotations)
    for index, joint in enumerate(skeleton.joints):
        if joint.parent < 0:
            world_rotations[:, index] = rotations[:, index]
            world[:, index, :] = local_offsets[:, index, :]
            continue

        # World orientation of this joint is its parent's world orientation
        # composed with its own local rotation.
        parent_world = world_rotations[:, joint.parent]
        world_rotations[:, index] = parent_world @ rotations[:, index]

        # Its position is the parent's position plus the bone offset expressed in
        # the parent's world frame.
        rotated = np.einsum("fij,fj->fi", parent_world, local_offsets[:, index, :])
        world[:, index, :] = world[:, joint.parent] + rotated

    return world


def load_joint_positions(path: Path, metres: bool = True) -> tuple[np.ndarray, Skeleton]:
    """
    Convenience wrapper: reads a BVH file and returns joint world positions.

    Returns (positions, skeleton) where positions has shape (frames, joints, 3).
    """
    skeleton = parse_skeleton(path)

    text = path.read_text(errors="replace")
    motion_block = text.split("MOTION", 1)[1]
    lines = [ln.strip() for ln in motion_block.splitlines() if ln.strip()]
    data_lines = lines[2 : 2 + skeleton.frame_count]

    motion = np.array(
        [[float(v) for v in ln.split()] for ln in data_lines], dtype=np.float64
    )
    if motion.shape[1] != skeleton.channel_count:
        raise ValueError(
            f"{path.name}: expected {skeleton.channel_count} channels, "
            f"got {motion.shape[1]}"
        )

    positions = forward_kinematics(skeleton, motion)
    if metres:
        positions = positions * METRES_PER_UNIT
    return positions, skeleton


__all__ = [
    "Joint",
    "Skeleton",
    "forward_kinematics",
    "load_joint_positions",
    "parse_skeleton",
]
