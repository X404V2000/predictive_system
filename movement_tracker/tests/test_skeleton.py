"""
Tests for BVH skeleton parsing, forward kinematics and feature extraction.

These guard the foundation of the behavioural layer. Getting any of it wrong is
SILENT: features are still produced, they are simply describing the wrong part
of the body. Three format details had to be established empirically, and each is
pinned by a test here:

  1. The cgspeed files define 31 joints, and only the root has position channels
     (96 channels: 6 root, 3 per other joint). Joint positions therefore require
     forward kinematics, not column indexing.
  2. Rotations are in DEGREES, not radians. Treating them as radians misplaces
     every joint away from the root.
  3. Braces appear both alone and trailing a JOINT line, so parent links must be
     derived by brace counting rather than per-line matching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from features import (
    FEATURE_NAMES,
    HEAD,
    LEFT_FOOT,
    LEFT_HAND,
    N_JOINTS,
    RIGHT_FOOT,
    RIGHT_HAND,
    ROOT_JOINT,
    extract_features,
    load_joint_positions,
)
from mocap import DEFAULT_DATA_ROOT, iter_clip_paths, load_clip
from skeleton import parse_skeleton

_HAS_DATA = DEFAULT_DATA_ROOT.exists() and any(DEFAULT_DATA_ROOT.glob("*/*.bvh"))
requires_data = pytest.mark.skipif(
    not _HAS_DATA, reason="CMU MoCap data not downloaded"
)


def _first_clip():
    return load_clip(next(iter_clip_paths()))


# --- skeleton parsing ------------------------------------------------------


@requires_data
def test_skeleton_has_expected_topology() -> None:
    """The cgspeed conversion defines 31 joints and 96 channels."""
    skeleton = parse_skeleton(_first_clip().path)

    assert len(skeleton.joints) == N_JOINTS == 31
    assert skeleton.channel_count == 96  # 6 for the root, 3 for each other joint
    assert skeleton.joints[0].name == "Hips"
    assert skeleton.joints[0].parent == -1
    assert skeleton.joints[0].is_root


@requires_data
def test_root_has_six_channels_and_others_three() -> None:
    skeleton = parse_skeleton(_first_clip().path)
    widths = [j.channel_slice.stop - j.channel_slice.start for j in skeleton.joints]

    assert widths[0] == 6
    assert all(w == 3 for w in widths[1:])
    assert sum(widths) == skeleton.channel_count


@requires_data
def test_parent_links_form_a_tree_rooted_at_hips() -> None:
    """
    Every joint except the root must have a valid parent that appears earlier in
    the list, otherwise a single forward FK pass would be invalid.
    """
    skeleton = parse_skeleton(_first_clip().path)

    for index, joint in enumerate(skeleton.joints):
        if joint.parent < 0:
            assert index == 0, "only the root may be parentless"
        else:
            assert 0 <= joint.parent < index


@requires_data
def test_known_joint_chain_is_correct() -> None:
    """Spot-check the chains that posture features depend on."""
    skeleton = parse_skeleton(_first_clip().path)

    def parent_name(name: str) -> str:
        index = skeleton.index_of(name)
        return skeleton.joints[skeleton.joints[index].parent].name

    assert parent_name("LeftFoot") == "LeftLeg"
    assert parent_name("RightFoot") == "RightLeg"
    assert parent_name("Spine") == "LowerBack"
    assert parent_name("Head") == "Neck1"


# --- forward kinematics ----------------------------------------------------


@requires_data
def test_forward_kinematics_is_anatomically_plausible() -> None:
    """
    The head must be above the hips and the feet below them.

    This is the check that caught both the radian/degree error and the incorrect
    rotation accumulation, each of which placed the head BELOW the hips.
    """
    positions, skeleton = load_joint_positions(_first_clip().path)

    hip = positions[:, ROOT_JOINT, 1].mean()
    head = positions[:, HEAD, 1].mean()
    left_foot = positions[:, LEFT_FOOT, 1].mean()
    right_foot = positions[:, RIGHT_FOOT, 1].mean()

    assert 0.6 < hip < 1.2, f"hip height {hip:.3f} m is implausible"
    assert head > hip, "head must be above the hips"
    assert left_foot < hip, "left foot must be below the hips"
    assert right_foot < hip, "right foot must be below the hips"


@requires_data
def test_forward_kinematics_is_consistent_across_clips() -> None:
    """
    Body proportions must be stable across subjects and actions; a format error
    typically shows up as one clip behaving differently from the rest.
    """
    snippets = list(iter_clip_paths())[:6]
    for path in snippets:
        positions, skeleton = load_joint_positions(path)
        hip = positions[:, skeleton.index_of("Hips"), 1].mean()
        head = positions[:, skeleton.index_of("Head"), 1].mean()
        # The measured head-to-hip offset is 0.36-0.37 m across every clip. This
        # bound is set around the observed value rather than an assumed anatomy,
        # since the skeleton's spine chain is a specific (short) proxy for the
        # torso rather than a full head-to-pelvis measurement.
        assert 0.25 < (head - hip) < 0.55, (
            f"{path.stem}: head-hip offset {head - hip:.3f} m is implausible"
        )


@requires_data
def test_joint_positions_have_expected_shape() -> None:
    clip = _first_clip()
    positions, skeleton = load_joint_positions(clip.path)

    assert positions.shape == (clip.frame_count, N_JOINTS, 3)
    assert np.all(np.isfinite(positions))


# --- features --------------------------------------------------------------


@requires_data
def test_feature_vector_is_finite_and_named() -> None:
    features = extract_features(_first_clip(), label="walk")

    assert features.values.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features.values))
    assert set(features.as_dict()) == set(FEATURE_NAMES)


@requires_data
def test_running_clip_is_faster_than_a_seated_clip() -> None:
    """A basic sanity property the classifier depends on."""
    paths = {p.parent.name: p for p in iter_clip_paths()}

    run = extract_features(load_clip(paths["009"]), label="run").as_dict()
    sit = extract_features(load_clip(paths["013"]), label="sit").as_dict()

    assert run["speed_mean"] > 2.0
    assert sit["speed_mean"] < 1.0
    assert run["speed_mean"] > 3.0 * sit["speed_mean"]


@requires_data
def test_posture_features_distinguish_raised_hands() -> None:
    """
    Gesturing actions raise the hands relative to walking. If the hand joint
    index were wrong, this would collapse.
    """
    paths = {p.parent.name: p for p in iter_clip_paths()}

    walk = extract_features(load_clip(paths["007"]), label="walk").as_dict()
    boxing = extract_features(load_clip(paths["014"]), label="boxing").as_dict()

    assert boxing["hand_height_mean"] > walk["hand_height_mean"]
