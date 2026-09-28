"""
Tests for pose sources and the live pipeline glue.

The pose abstraction exists so the pipeline does not depend on a particular way
of obtaining skeletons. These tests pin the contract every source must satisfy,
because a source that silently emitted the wrong joint order or units would
produce plausible-looking but wrong predictions rather than an error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from features import (
    HEAD,
    JOINT_ORDER,
    LEFT_FOOT,
    N_JOINTS,
    RIGHT_FOOT,
    ROOT_JOINT,
    features_from_arrays,
)
from mocap import DEFAULT_DATA_ROOT
from pose import (
    MocapPlaybackSource,
    PoseFrame,
    PoseSourceError,
    SyntheticPoseSource,
    reorder_to_cmu,
    validate_frame,
)

_HAS_DATA = DEFAULT_DATA_ROOT.exists() and any(DEFAULT_DATA_ROOT.glob("*/*.bvh"))
requires_data = pytest.mark.skipif(not _HAS_DATA, reason="motion capture not present")


# --- frame contract --------------------------------------------------------


def test_validate_frame_accepts_a_correct_skeleton() -> None:
    frame = PoseFrame(np.zeros((N_JOINTS, 3)), timestamp=0.0, source="test")
    validate_frame(frame)


def test_validate_frame_rejects_wrong_joint_count() -> None:
    """A different skeleton must fail loudly, not silently mispredict."""
    frame = PoseFrame(np.zeros((17, 3)), timestamp=0.0, source="test")
    with pytest.raises(PoseSourceError, match=r"\(31, 3\)"):
        validate_frame(frame)


def test_validate_frame_rejects_non_finite_values() -> None:
    positions = np.zeros((N_JOINTS, 3))
    positions[3, 1] = np.inf
    frame = PoseFrame(positions, timestamp=0.0, source="test")
    with pytest.raises(PoseSourceError, match="non-finite"):
        validate_frame(frame)


# --- synthetic source ------------------------------------------------------


def test_synthetic_source_requires_start() -> None:
    with pytest.raises(PoseSourceError, match="start"):
        SyntheticPoseSource().read()


def test_synthetic_source_is_anatomically_plausible() -> None:
    """
    The synthetic walker must have a real skeleton: head above hips, feet below.

    If it did not, it would exercise the pipeline with nonsense and give false
    confidence that the pipeline works.
    """
    source = SyntheticPoseSource(fps=30.0, speed=1.2)
    source.start()
    frames = [source.read() for _ in range(60)]
    positions = np.stack([f.positions for f in frames])

    hip = positions[:, ROOT_JOINT, 1].mean()
    head = positions[:, HEAD, 1].mean()
    foot = positions[:, LEFT_FOOT, 1].mean()

    assert 0.7 < hip < 1.1
    assert head > hip
    assert foot < hip


def test_synthetic_source_travels_at_the_requested_speed() -> None:
    source = SyntheticPoseSource(fps=60.0, speed=1.5)
    source.start()
    frames = [source.read() for _ in range(120)]
    travelled = frames[-1].positions[ROOT_JOINT, 2] - frames[0].positions[ROOT_JOINT, 2]

    # 120 frames at 60 fps is 2 s, so 3 m at 1.5 m/s.
    assert travelled == pytest.approx(3.0, abs=0.15)


def test_synthetic_source_honours_duration() -> None:
    source = SyntheticPoseSource(fps=30.0, duration=1.0)
    source.start()
    frames = 0
    while source.read() is not None:
        frames += 1
        assert frames < 1000, "source did not terminate"
    assert frames == pytest.approx(31, abs=2)


def test_synthetic_walk_is_recognised_as_locomotion() -> None:
    """
    The whole chain must agree: pose -> features -> classifier says "locomote".

    This is the end-to-end check that the serving path is wired correctly. If the
    joint order or units drifted, this would fail even though every unit test
    passed.
    """
    from actions import train_default_model

    source = SyntheticPoseSource(fps=30.0, speed=1.2)
    source.start()
    frames = [source.read() for _ in range(60)]
    window = np.stack([f.positions for f in frames])

    values = features_from_arrays(window[:, ROOT_JOINT, :], window, 1.0 / 30.0)
    model = train_default_model()
    predicted, confidence = model.predict(values)

    assert predicted == "locomote"
    assert confidence > 0.4


def test_synthetic_source_returns_fresh_arrays() -> None:
    """
    Consecutive frames must not alias the same buffer.

    Frames are accumulated into a rolling window, so a source reusing one array
    would make the whole window identical and silently destroy the features.
    """
    source = SyntheticPoseSource(fps=30.0)
    source.start()
    first = source.read()
    second = source.read()

    assert not np.shares_memory(first.positions, second.positions)
    assert not np.allclose(first.positions, second.positions)


# --- mocap playback --------------------------------------------------------


@requires_data
def test_mocap_source_replays_ordered_frames() -> None:
    path = DEFAULT_DATA_ROOT / "013" / "13_05.bvh"
    source = MocapPlaybackSource(path=path, loop=False)
    source.start()

    frame = source.read()
    assert frame is not None
    validate_frame(frame)

    timestamps = []
    while True:
        nxt = source.read()
        if nxt is None:
            break
        timestamps.append(nxt.timestamp)
        if len(timestamps) > 5000:
            break

    assert timestamps == sorted(timestamps), "timestamps must advance"
    assert len(set(timestamps)) == len(timestamps), "timestamps must be unique"


@requires_data
def test_mocap_source_loops_when_asked() -> None:
    path = DEFAULT_DATA_ROOT / "013" / "13_05.bvh"
    source = MocapPlaybackSource(path=path, loop=True)
    source.start()

    # Read past the clip length; a looping source never returns None.
    for _ in range(6000):
        assert source.read() is not None


# --- joint reordering ------------------------------------------------------


def test_reorder_to_cmu_round_trips() -> None:
    rng = np.random.default_rng(0)
    original = rng.normal(size=(5, N_JOINTS, 3))

    shuffled_names = list(reversed(JOINT_ORDER))
    shuffled = original[:, ::-1, :]

    restored = reorder_to_cmu(shuffled, shuffled_names)
    assert np.allclose(restored, original)


def test_reorder_to_cmu_rejects_missing_joints() -> None:
    positions = np.zeros((5, 3, 3))
    with pytest.raises(PoseSourceError, match="missing"):
        reorder_to_cmu(positions, ["Hips", "Spine", "Head"])
