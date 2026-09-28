"""
Action Sequence Segmentation and Transition Modelling.

Models which action tends to FOLLOW which, i.e. P(next action | current action),
the capability the phrase "predict the next movement" actually refers to.

Data reality, established by measurement rather than assumed:

* Each clip is labelled with one or more comma-separated actions from the CMU
  catalogue, e.g. "sit on high stool, stand up" or "unscrew bottlecap, drink
  soda, screw on bottlecap". 34 of the downloaded clips describe sequences.
* Those labels do NOT come with timings, so segment boundaries must be inferred
  from the motion itself. A clip is therefore cut into fixed-duration windows and
  each window is classified independently; the sequence of window classifications
  is the observed action sequence.
* Transition evidence is consequently weaker than the labels suggest, because a
  misclassified window becomes a spurious transition.

MEASURED LIMITATION — this model is not yet useful, and that should not be
softened. Under leave-one-subject-out it scores 64.9% next-action accuracy, but
a constant "always predict manipulate" baseline scores 64.3%. On balanced
accuracy, which removes the class skew, both score exactly 51.4%: the model
adds ZERO lift over a constant predictor.

Three reasons, in order of importance:

  1. One class dominates the window stream. 64% of classified windows are
     "manipulate", so most rows of the transition matrix are near-degenerate and
     collapse to the majority action.
  2. Segment boundaries are inferred, not known. The catalogue labels carry no
     timings, so each misclassified window manufactures a transition that never
     happened.
  3. Only 34 sequence clips are available.

The infrastructure is kept because it is correct and will become useful with
better input: even window durations would help, and finer window granularity
plus a class-balanced classifier would help more. It should NOT be presented as
delivering "predict the next movement" until the lift is demonstrably positive.

numpy and the standard library only.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from actions import GaussianActionModel, coarse_action
from features import ClipFeatures, extract_features
from mocap import MotionClip, load_clip

# Duration of each analysis window. Long enough for the features to be stable
# (several gait cycles), short enough to resolve a transition within a clip.
WINDOW_SECONDS: float = 2.0

# A transition is only counted when a window is confidently classified; weak
# windows are treated as "no decision" rather than turned into spurious edges.
MIN_TRANSITION_CONFIDENCE: float = 0.5

# Pseudo-count added to every transition so that an unobserved pair retains a
# small probability, given the small number of clips available.
TRANSITION_SMOOTHING: float = 0.2

# Sentinel actions bounding every observed sequence.
START_ACTION: str = "<start>"
END_ACTION: str = "<end>"


@dataclass(frozen=True)
class SegmentedClip:
    """A clip decomposed into a sequence of classified windows."""

    clip_id: str
    label: str
    actions: tuple[str, ...]
    confidences: tuple[float, ...]

    @property
    def sequence(self) -> tuple[str, ...]:
        """Action sequence including the start and end sentinels."""
        return (START_ACTION,) + self.actions + (END_ACTION,)


def segment_clip(
    clip: MotionClip, label: str, model: GaussianActionModel
) -> SegmentedClip:
    """
    Classifies a clip in fixed-duration windows, producing an action sequence.

    Windows that the classifier is unsure about are recorded as "uncertain"
    rather than forced into a class, so a weak classification cannot masquerade
    as a real transition.
    """
    dt = clip.sample_dt
    window_frames = max(1, int(round(WINDOW_SECONDS / dt)))

    actions: list[str] = []
    confidences: list[float] = []

    for start in range(0, clip.frame_count, window_frames):
        end = min(start + window_frames, clip.frame_count)
        if end - start < window_frames // 2:
            break  # ignore a trailing fragment too short to characterise

        window = _subclip(clip, start, end)
        features = extract_features(window, label=label)
        predicted, confidence = model.predict(features.values)
        if confidence < MIN_TRANSITION_CONFIDENCE:
            predicted = "uncertain"
        actions.append(predicted)
        confidences.append(confidence)

    return SegmentedClip(
        clip_id=clip.subject,
        label=label,
        actions=tuple(actions),
        confidences=tuple(confidences),
    )


def _subclip(clip: MotionClip, start: int, end: int) -> MotionClip:
    """
    Builds a MotionClip covering frames [start, end).

    Only the fields needed for feature extraction and duration are populated;
    the joint block is re-read from the original file, so the subclip keeps a
    reference to the parent path plus an explicit frame range.
    """
    from physics import derive_motion

    positions = clip.positions[start:end]
    velocities, accelerations = derive_motion(positions, clip.sample_dt)
    return MotionClip(
        path=clip.path,
        subject=clip.subject,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        rotations=clip.rotations[start:end],
        sample_dt=clip.sample_dt,
        frame_range=(start, end),
    )


# Segmentation re-reads motion capture and runs forward kinematics per window,
# which is expensive. Results are cached because analysis and tests both call it.
#
# The key is the model's fitted classes rather than its identity: callers
# construct a fresh model object each time, so id() would never hit.
_SEQUENCE_CACHE: dict[tuple[str, ...], list[SegmentedClip]] = {}


def collect_sequences(
    model: GaussianActionModel,
    max_clips: int | None = None,
) -> list[SegmentedClip]:
    """Segments every labelled clip into an action sequence, with caching."""
    cache_key = tuple(model.classes)
    if max_clips is None and cache_key in _SEQUENCE_CACHE:
        return _SEQUENCE_CACHE[cache_key]

    from actions import DEFAULT_LABELS_PATH, coarse_action as _coarse

    descriptions: dict[str, str] = {}
    for line in DEFAULT_LABELS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        code, desc = line.split("\t", 1)
        descriptions[code.strip()] = desc.strip()

    from mocap import iter_clip_paths

    sequences: list[SegmentedClip] = []
    for path in iter_clip_paths():
        description = descriptions.get(path.stem)
        if description is None or _coarse(description) == "other":
            continue
        # Only clips that describe a genuine sequence are useful here.
        if "," not in description:
            continue
        sequences.append(segment_clip(load_clip(path), description, model))
        if max_clips is not None and len(sequences) >= max_clips:
            break

    if max_clips is None:
        _SEQUENCE_CACHE[cache_key] = sequences
    return sequences


@dataclass
class TransitionModel:
    """
    First-order Markov model over action sequences.

    Counts transitions from the observed sequences and normalises each row into
    a probability distribution over the next action. Smoothing keeps unobserved
    pairs possible, which matters at this sample size.
    """

    states: tuple[str, ...] = ()
    counts: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    observations: int = 0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, sequences: list[SegmentedClip]) -> "TransitionModel":
        """Estimates the transition matrix from observed action sequences."""
        if not sequences:
            raise ValueError("no sequences supplied")

        states = sorted(
            {START_ACTION, END_ACTION}
            | {a for seq in sequences for a in seq.sequence}
        )
        index = {s: i for i, s in enumerate(states)}
        counts = np.zeros((len(states), len(states)), dtype=np.float64)

        for sequence in sequences:
            for current, following in zip(sequence.sequence, sequence.sequence[1:]):
                counts[index[current], index[following]] += 1.0

        counts += TRANSITION_SMOOTHING
        # The end state is absorbing: nothing follows it.
        end_index = index[END_ACTION]
        counts[end_index, :] = 0.0
        counts[end_index, end_index] = 1.0

        probabilities = counts / counts.sum(axis=1, keepdims=True)

        return cls(
            states=tuple(states),
            counts=counts,
            probabilities=probabilities,
            observations=len(sequences),
            metadata={"windows_per_clip": [len(s.actions) for s in sequences]},
        )

    def next_action_distribution(self, current: str) -> dict[str, float]:
        """
        Returns P(next action | current action).

        An unknown current action yields the empirical action prior rather than
        an error, so the model degrades gracefully on unseen input.
        """
        if self.probabilities is None:
            raise ValueError("model has not been fitted")

        if current in self.states:
            row = self.probabilities[self.states.index(current)]
        elif START_ACTION in self.states:
            row = self.probabilities[self.states.index(START_ACTION)]
        else:
            raise ValueError("model has no states")

        return {
            state: float(row[i])
            for i, state in enumerate(self.states)
            if row[i] > 0.0
        }

    def most_likely_next(self, current: str) -> tuple[str, float]:
        """Returns the single most probable following action and its probability."""
        distribution = self.next_action_distribution(current)
        # The absorbing end state is not a useful prediction.
        candidates = {k: v for k, v in distribution.items() if k != END_ACTION}
        if not candidates:
            return END_ACTION, 1.0
        best = max(candidates, key=lambda k: candidates[k])
        return best, candidates[best]


__all__ = [
    "END_ACTION",
    "MIN_TRANSITION_CONFIDENCE",
    "START_ACTION",
    "SegmentedClip",
    "TransitionModel",
    "WINDOW_SECONDS",
    "collect_sequences",
    "segment_clip",
]
