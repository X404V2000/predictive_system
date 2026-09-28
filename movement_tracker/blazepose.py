"""
BlazePose Landmark Mapping.

Converts the 33 landmarks produced by the OpenCV-Zoo BlazePose model into the
31-joint CMU skeleton the action classifier was trained on.

The two skeletons are NOT the same, and the differences are the reason this file
exists rather than a one-line index lookup:

  * **Different topology.** CMU has a spine chain (LowerBack, Spine, Spine1) and
    separate hip joints. BlazePose has neither: it goes straight from shoulders
    to hips and has one hip landmark per side. The missing joints are therefore
    interpolated along the line between their neighbours, which preserves overall
    body shape but does not recover real spinal curvature.
  * **Different root.** CMU's root is the Hips joint. BlazePose has separate left
    and right hips, so the root is their midpoint.
  * **Different fingers.** CMU models finger bases and thumbs; BlazePose stops at
    the wrist. Hand detail is replicated from the wrist position.
  * **Different units and axes.** BlazePose reports normalised image coordinates
    (x, y in [0, 1] of the frame, z relative to the hips) with y pointing DOWN.
    CMU data is metres with y pointing UP.

The consequence, stated plainly: features derived from this mapping are
approximate, and the classifier will not reach its motion-capture accuracy on
webcam input. tools/measure_pose_transfer.py quantifies the gap.

Landmark indices, verified empirically by running the model on the reference
skeleton diagram and reading the resulting geometry (y grows downward):

    face         0-10   (nose, eyes, ears, mouth)
    shoulders    11 left, 12 right
    elbows       13 left, 14 right
    wrists       15 left, 16 right
    hips         23 left, 24 right
    knees        25 left, 26 right
    ankles       27 left, 28 right
    feet         29-32  (heels and toe tips)
    33-38        PADDING: visibility is exactly 0.0 on every input tested,
                 both a real person and pure noise. They are not landmarks.

numpy and the standard library only.
"""

from dataclasses import dataclass

import numpy as np

from features import JOINT_ORDER

# BlazePose landmark indices.
BP_NOSE = 0
BP_LEFT_SHOULDER, BP_RIGHT_SHOULDER = 11, 12
BP_LEFT_ELBOW, BP_RIGHT_ELBOW = 13, 14
BP_LEFT_WRIST, BP_RIGHT_WRIST = 15, 16
BP_LEFT_HIP, BP_RIGHT_HIP = 23, 24
BP_LEFT_KNEE, BP_RIGHT_KNEE = 25, 26
BP_LEFT_ANKLE, BP_RIGHT_ANKLE = 27, 28
BP_LEFT_HEEL, BP_RIGHT_HEEL = 29, 30
BP_LEFT_FOOT, BP_RIGHT_FOOT = 31, 32

# Real landmarks end here; everything after is padding.
BLAZEPOSE_LANDMARKS: int = 33

# Median adult proportions (metres), used to convert normalised landmarks into an
# approximate metric scale. A single RGB camera cannot recover absolute size, so
# one reference length must be assumed. Shoulder width is used because it is the
# most stably observed segment and is measured directly by both skeletons.
ASSUMED_SHOULDER_WIDTH_M: float = 0.40

# Ratio of the shoulder-to-hip span to shoulder width in a standing adult. Used to
# place the ankle plane and scale depth, since BlazePose z is hip-relative.
_SPINE_TO_HIP_RATIO: float = 1.15

# Fingertip chains are absent from BlazePose. They are placed at the wrist rather
# than omitted so downstream code can rely on a fixed joint count, and so posture
# features that average limb positions are not skewed by missing entries.
_FINGER_JOINTS: tuple[int, ...] = (21, 22, 23, 28, 29, 30)


@dataclass(frozen=True)
class LandmarkResult:
    """Raw model output for one frame, in normalised image coordinates."""

    landmarks: np.ndarray  # (33, 3) x, y, z with y pointing DOWN
    visibility: np.ndarray  # (33,) in [0, 1] after sigmoid
    present: bool  # whether a person was detected above threshold


def sigmoid(values: np.ndarray) -> np.ndarray:
    """
    Recovers probabilities from the model's raw logits.

    The model emits visibility and presence as logits, not probabilities: raw
    values span roughly -18 to +4, which is not a valid probability. Applying a
    sigmoid is required; treating them directly as confidences would make almost
    every landmark look invisible.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


# The model reports coordinates in PIXELS of its own square input, not in [0, 1].
# Verified by inspecting the raw output on a 772x438 diagram: values span roughly
# 99 to 151 for a figure occupying the middle of the frame, which is only
# consistent with pixel units. Treating them as normalised coordinates makes the
# derived scale wrong by the input resolution.
MODEL_INPUT_PIXELS: float = 256.0


def landmark_scale(landmarks: np.ndarray) -> float:
    """
    Estimates metres per model pixel from shoulder width.

    A monocular camera cannot observe absolute scale, so one physical reference
    must be assumed. Shoulder width is chosen because it varies little across
    adults and is observed directly by the model.

    Returns 0.0 when the shoulders coincide, which means the estimate is
    degenerate and the frame should be rejected rather than scaled to infinity.
    """
    left = landmarks[BP_LEFT_SHOULDER]
    right = landmarks[BP_RIGHT_SHOULDER]
    span = float(np.linalg.norm(left[:2] - right[:2]))
    if span < 1e-6:
        return 0.0
    return ASSUMED_SHOULDER_WIDTH_M / span


def to_cmu_skeleton(
    landmarks: np.ndarray,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> np.ndarray:
    """
    Maps 33 BlazePose landmarks to the 31-joint CMU skeleton, in metres, Y-up.

    Args:
        landmarks     (33, 3) coordinates in MODEL INPUT PIXELS, y pointing DOWN
        frame_width   unused; retained so callers pass the source frame and a
                      future aspect-ratio correction has the information
        frame_height  unused, as above

    Returns (31, 3) joint positions in metres with Y pointing UP, in JOINT_ORDER.

    Joints absent from BlazePose (the spine chain, hip pivots, fingers) are
    interpolated between their nearest modelled neighbours. Interpolation keeps
    the silhouette correct but does NOT invent information the camera did not
    observe, which is the honest limit of this mapping.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] < 3:
        raise ValueError(f"expected (n, 3) landmarks, got {landmarks.shape}")
    if landmarks.shape[0] < BLAZEPOSE_LANDMARKS:
        raise ValueError(
            f"expected at least {BLAZEPOSE_LANDMARKS} landmarks, got "
            f"{landmarks.shape[0]}"
        )
    # This export emits 39 rows, but rows 33-38 are padding whose visibility is
    # exactly 0.0 for every input tried, including a real person and pure noise.
    # They are dropped rather than treated as landmarks.
    landmarks = landmarks[:BLAZEPOSE_LANDMARKS]

    scale = landmark_scale(landmarks)
    if scale <= 0.0:
        raise ValueError("degenerate landmark geometry: shoulder width is zero")

    # Convert to a metric, Y-up frame. The model's coordinates are already in
    # pixels of its own square input, so a single metres-per-pixel factor is all
    # that is required. Multiplying by the source frame size as well would scale
    # the skeleton by the frame resolution, which produced bodies hundreds of
    # metres tall.
    #
    # BlazePose y grows downward, so it is negated. Its z is hip-relative and on a
    # different scale from x and y, so it is adjusted by an assumed ratio rather
    # than used directly.
    points = np.empty((BLAZEPOSE_LANDMARKS, 3), dtype=np.float64)
    points[:, 0] = landmarks[:, 0] * scale
    points[:, 1] = -landmarks[:, 1] * scale
    points[:, 2] = landmarks[:, 2] * scale * _SPINE_TO_HIP_RATIO

    hips = 0.5 * (points[BP_LEFT_HIP] + points[BP_RIGHT_HIP])
    neck_base = 0.5 * (points[BP_LEFT_SHOULDER] + points[BP_RIGHT_SHOULDER])
    head = points[BP_NOSE]

    # Shift so the hip midpoint is the origin, matching how the CMU root relates
    # to its own body: a human's hip height above the floor is not observable from
    # one camera, and inventing an absolute height would corrupt the vertical
    # features that separate seated from standing.
    points = points - hips

    def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation; used only where the skeleton has no joint."""
        return a + (b - a) * t

    output = np.zeros((len(JOINT_ORDER), 3), dtype=np.float64)
    index = {name: i for i, name in enumerate(JOINT_ORDER)}

    def put(name: str, value: np.ndarray) -> None:
        output[index[name]] = value

    # --- root and legs ---------------------------------------------------
    put("Hips", hips - hips)  # origin by construction
    put("LHipJoint", points[BP_LEFT_HIP])
    put("RHipJoint", points[BP_RIGHT_HIP])
    put("LeftUpLeg", points[BP_LEFT_HIP])
    put("RightUpLeg", points[BP_RIGHT_HIP])
    # BlazePose gives knee and ankle directly; the CMU chain has an intermediate
    # leg joint with no counterpart, placed midway along the bone.
    put("LeftLeg", lerp(points[BP_LEFT_HIP], points[BP_LEFT_KNEE], 0.5))
    put("RightLeg", lerp(points[BP_RIGHT_HIP], points[BP_RIGHT_KNEE], 0.5))
    put("LeftFoot", points[BP_LEFT_ANKLE])
    put("RightFoot", points[BP_RIGHT_ANKLE])
    put("LeftToeBase", points[BP_LEFT_FOOT])
    put("RightToeBase", points[BP_RIGHT_FOOT])

    # --- spine (entirely interpolated) -----------------------------------
    # CMU has three spine joints between hips and neck. BlazePose models none of
    # them, so they are spaced along the hip-to-neck line. Real spinal curvature
    # is therefore not represented.
    put("LowerBack", lerp(hips, neck_base, 0.25) - hips)
    put("Spine", lerp(hips, neck_base, 0.50) - hips)
    put("Spine1", lerp(hips, neck_base, 0.75) - hips)
    put("Neck", lerp(neck_base, head, 0.30) - hips)
    put("Neck1", lerp(neck_base, head, 0.60) - hips)
    put("Head", head - hips)

    # --- arms ------------------------------------------------------------
    put("LeftShoulder", points[BP_LEFT_SHOULDER])
    put("RightShoulder", points[BP_RIGHT_SHOULDER])
    put("LeftArm", lerp(points[BP_LEFT_SHOULDER], points[BP_LEFT_ELBOW], 0.5))
    put("RightArm", lerp(points[BP_RIGHT_SHOULDER], points[BP_RIGHT_ELBOW], 0.5))
    put("LeftForeArm", points[BP_LEFT_ELBOW])
    put("RightForeArm", points[BP_RIGHT_ELBOW])
    put("LeftHand", points[BP_LEFT_WRIST])
    put("RightHand", points[BP_RIGHT_WRIST])

    # Fingers are not observed. They are placed at the wrist so the joint count is
    # complete; a caller needing real finger position must use a hand model.
    for name in ("LeftFingerBase", "LeftHandIndex1", "LThumb"):
        put(name, points[BP_LEFT_WRIST])
    for name in ("RightFingerBase", "RightHandIndex1", "RThumb"):
        put(name, points[BP_RIGHT_WRIST])

    if not np.all(np.isfinite(output)):
        raise ValueError("mapping produced non-finite joint positions")

    return output


def mean_visibility(visibility: np.ndarray, indices: tuple[int, ...]) -> float:
    """Average visibility over selected landmarks, for confidence gating."""
    return float(np.mean([visibility[i] for i in indices]))


# Landmarks whose visibility must be reasonable for the skeleton to be usable for
# action recognition: torso, arms and legs. Face points are excluded because they
# are frequently occluded or off-frame without affecting body motion.
CORE_LANDMARKS: tuple[int, ...] = (
    BP_LEFT_SHOULDER,
    BP_RIGHT_SHOULDER,
    BP_LEFT_ELBOW,
    BP_RIGHT_ELBOW,
    BP_LEFT_WRIST,
    BP_RIGHT_WRIST,
    BP_LEFT_HIP,
    BP_RIGHT_HIP,
    BP_LEFT_KNEE,
    BP_RIGHT_KNEE,
    BP_LEFT_ANKLE,
    BP_RIGHT_ANKLE,
)

__all__ = [
    "BLAZEPOSE_LANDMARKS",
    "CORE_LANDMARKS",
    "LandmarkResult",
    "to_cmu_skeleton",
    "landmark_scale",
    "mean_visibility",
    "sigmoid",
]
