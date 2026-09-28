"""
Tests for the BlazePose landmark mapping.

`blazepose.py` converts the 33 landmarks produced by the webcam pose model into
the 31-joint CMU skeleton the action classifier was trained on. Two skeletons
with different topology means this cannot be an index lookup, so the mapping is
worth testing carefully.

The landmark indices were established empirically by running the model on the
reference skeleton diagram and reading the resulting geometry, not from memory.
The first attempt used shoulders at 11/12 for the hips and would have mapped the
upper body onto the legs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import blazepose as bp
from features import (
    HEAD,
    JOINT_ORDER,
    LEFT_FOOT,
    LEFT_HAND,
    N_JOINTS,
    RIGHT_FOOT,
    ROOT_JOINT,
)


def _landmarks_from_pixel_coords() -> np.ndarray:
    """
    A hand-built front-facing figure in MODEL PIXEL coordinates, y downward.

    Shoulder span is exactly 20 px so the metres-per-pixel factor is exactly
    0.02 and every expected value can be computed by hand.
    """
    L = np.zeros((39, 3), dtype=np.float64)  # 39 rows, as the model emits
    L[0] = [110, 80, 0]  # nose
    L[11] = [100, 100, 0]  # left shoulder
    L[12] = [120, 100, 0]  # right shoulder
    L[13] = [92, 140, 0]  # left elbow
    L[14] = [128, 140, 0]  # right elbow
    L[15] = [86, 180, 0]  # left wrist
    L[16] = [134, 180, 0]  # right wrist
    L[23] = [104, 160, 0]  # left hip
    L[24] = [116, 160, 0]  # right hip
    L[25] = [104, 220, 0]  # left knee
    L[26] = [116, 220, 0]  # right knee
    L[27] = [104, 280, 0]  # left ankle
    L[28] = [116, 280, 0]  # right ankle
    L[29] = [106, 286, 0]  # left heel
    L[30] = [114, 286, 0]  # right heel
    L[31] = [104, 290, 0]  # left foot
    L[32] = [116, 290, 0]  # right foot
    return L


# --- scale and units -------------------------------------------------------


def test_scale_is_metres_per_model_pixel() -> None:
    """
    The model reports pixels of its own 256x256 input, not normalised [0, 1].

    Treating them as normalised and also multiplying by frame size scaled the body
    by the frame resolution, producing a skeleton hundreds of metres tall.
    """
    landmarks = _landmarks_from_pixel_coords()
    scale = bp.landmark_scale(landmarks)

    # Shoulders are 20 px apart and represent 0.40 m.
    assert scale == pytest.approx(0.40 / 20.0)


def test_degenerate_geometry_is_rejected() -> None:
    """Coincident shoulders give an infinite scale, so the frame must be refused."""
    landmarks = _landmarks_from_pixel_coords()
    landmarks[bp.BP_RIGHT_SHOULDER] = landmarks[bp.BP_LEFT_SHOULDER]
    assert bp.landmark_scale(landmarks) == 0.0

    with pytest.raises(ValueError, match="degenerate"):
        bp.to_cmu_skeleton(landmarks)


# --- geometry --------------------------------------------------------------


def test_mapping_produces_the_cmu_joint_count() -> None:
    skeleton = bp.to_cmu_skeleton(_landmarks_from_pixel_coords())
    assert skeleton.shape == (N_JOINTS, 3)
    assert np.all(np.isfinite(skeleton))


def test_hip_midpoint_is_the_origin() -> None:
    """
    The CMU root is the Hips joint, which BlazePose does not have.

    Absolute height above the floor is not observable from one camera, so the hip
    midpoint becomes the origin rather than an invented floor level. Inventing one
    would corrupt the vertical features that separate seated from standing.
    """
    skeleton = bp.to_cmu_skeleton(_landmarks_from_pixel_coords())
    assert np.allclose(skeleton[ROOT_JOINT], 0.0)


def test_known_geometry_is_exact() -> None:
    """
    Every landmark's mapped position must match hand-computed values.

    With a 20 px shoulder span the scale is exactly 0.02 m/px, so expected
    coordinates follow by inspection. This is the test that proves the axis flip
    and unit conversion are right rather than merely plausible.
    """
    skeleton = bp.to_cmu_skeleton(_landmarks_from_pixel_coords())

    shoulders = skeleton[[17, 24]]  # LeftShoulder, RightShoulder
    assert np.linalg.norm(shoulders[0][:2] - shoulders[1][:2]) == pytest.approx(0.40)

    # Nose is 80 px above the hip line (y=160), and y is flipped, so it is +1.6 m.
    assert skeleton[HEAD, 1] == pytest.approx((160 - 80) * 0.02)

    # Ankle is 120 px below the hip line, so -2.4 m before the flip, +2.4 after
    # negating y: the hip sits ABOVE the ankle, which is what "below hips" means.
    assert skeleton[ROOT_JOINT, 1] - skeleton[LEFT_FOOT, 1] == pytest.approx(
        (280 - 160) * 0.02
    )

    # Wrist at y=180 is 20 px below the hip line, so below the hips.
    assert skeleton[LEFT_HAND, 1] - skeleton[ROOT_JOINT, 1] == pytest.approx(
        (160 - 180) * 0.02
    )


def test_vertical_axis_points_up() -> None:
    """The head must map above the hips: BlazePose y grows downward and is negated."""
    skeleton = bp.to_cmu_skeleton(_landmarks_from_pixel_coords())
    assert skeleton[HEAD, 1] > skeleton[ROOT_JOINT, 1]
    assert skeleton[LEFT_FOOT, 1] < skeleton[ROOT_JOINT, 1]


def test_left_and_right_are_not_swapped() -> None:
    """Left-side joints must map to positive x, given the model's convention."""
    skeleton = bp.to_cmu_skeleton(_landmarks_from_pixel_coords())
    left_shoulder = skeleton[JOINT_ORDER.index("LeftShoulder")]
    right_shoulder = skeleton[JOINT_ORDER.index("RightShoulder")]

    # Landmark 11 (left) has the smaller pixel x; after mapping, left < right.
    assert left_shoulder[0] < right_shoulder[0]


def test_interpolated_joints_lie_between_their_neighbours() -> None:
    """
    The spine chain does not exist in BlazePose and is interpolated.

    Interpolation is honest only if the result actually lies on the segment
    between the modelled endpoints; if it drifted outside, the body shape would be
    wrong rather than merely simplified.
    """
    skeleton = bp.to_cmu_skeleton(_landmarks_from_pixel_coords())
    index = {name: i for i, name in enumerate(JOINT_ORDER)}

    lower_back = skeleton[index["LowerBack"]]
    spine = skeleton[index["Spine"]]
    spine1 = skeleton[index["Spine1"]]

    # Spine joints must be monotonically ordered up the torso.
    assert lower_back[1] < spine[1] < spine1[1]

    # And each must sit within the hip-to-shoulder span.
    hip_y = skeleton[ROOT_JOINT][1]
    neck_y = 0.5 * (skeleton[index["LeftShoulder"]][1] + skeleton[index["RightShoulder"]][1])
    for point in (lower_back, spine, spine1):
        assert min(hip_y, neck_y) <= point[1] <= max(hip_y, neck_y)


def test_finger_joints_are_placed_at_the_wrist() -> None:
    """
    BlazePose stops at the wrist, so finger joints replicate it.

    They are filled rather than left as zeros so the joint count is complete and
    posture features that average limb positions are not dragged toward the
    origin by missing entries.
    """
    skeleton = bp.to_cmu_skeleton(_landmarks_from_pixel_coords())
    index = {name: i for i, name in enumerate(JOINT_ORDER)}
    wrist = skeleton[index["LeftHand"]]

    for name in ("LeftFingerBase", "LeftHandIndex1", "LThumb"):
        assert np.allclose(skeleton[index[name]], wrist)


def test_nan_input_is_rejected() -> None:
    landmarks = _landmarks_from_pixel_coords()
    landmarks[11, 0] = np.nan
    with pytest.raises(ValueError):
        bp.to_cmu_skeleton(landmarks)


def test_too_few_landmarks_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        bp.to_cmu_skeleton(np.zeros((10, 3)))

    with pytest.raises(ValueError):
        bp.to_cmu_skeleton(np.zeros((33, 2)))


def test_padding_rows_are_ignored() -> None:
    """
    The export emits 39 rows but only the first 33 are landmarks.

    Rows 33-38 have visibility exactly 0.0 on every input tested, including a real
    person and pure noise. Garbage placed there must not affect the result.
    """
    clean = _landmarks_from_pixel_coords()
    polluted = clean.copy()
    polluted[33:] = 9999.0

    assert np.allclose(bp.to_cmu_skeleton(clean), bp.to_cmu_skeleton(polluted))


# --- confidence ------------------------------------------------------------


def test_sigmoid_recovers_probabilities() -> None:
    """Visibility is emitted as a logit and must be squashed into [0, 1]."""
    logits = np.array([-18.0, -2.0, 0.0, 2.0, 4.0])
    probabilities = bp.sigmoid(logits)

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[2] == pytest.approx(0.5)
    assert probabilities[-1] > probabilities[0]


def test_visibility_uses_body_landmarks_not_face() -> None:
    """
    Confidence gating must key on the body, not the face.

    Face landmarks are often occluded or cropped without affecting body motion, so
    including them would reject otherwise usable frames.
    """
    assert bp.BP_NOSE not in bp.CORE_LANDMARKS
    for landmark in (bp.BP_LEFT_SHOULDER, bp.BP_LEFT_HIP, bp.BP_LEFT_ANKLE):
        assert landmark in bp.CORE_LANDMARKS


def test_mean_visibility_averages_selected_landmarks() -> None:
    visibility = np.zeros(33)
    visibility[bp.BP_LEFT_SHOULDER] = 1.0
    assert bp.mean_visibility(visibility, (bp.BP_LEFT_SHOULDER,)) == pytest.approx(1.0)
    assert bp.mean_visibility(visibility, (bp.BP_LEFT_SHOULDER, bp.BP_RIGHT_SHOULDER)) == pytest.approx(0.5)
