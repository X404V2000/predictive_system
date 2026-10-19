"""
Simulated Perception and ReID Backend.

=============================================================================
THIS IS A SIMULATION. IT DOES NOT OBSERVE REAL PEOPLE.
=============================================================================

The integrated acceptance test exercises the whole platform flow:

    track a person -> calibrated precursor alert -> human review ->
    appearance-based ReID path reconstruction -> audit log

The production codebase does not yet contain multi-object tracking, a precursor
behaviour model, or an appearance-embedding ReID model. Rather than fake accuracy
figures, the perception layer is replaced here by a *deterministic scenario* driven
by a scripted ground truth, and every output carries `simulated=True`.

What this proves and what it does not:

  PROVES   the orchestration, the calibration contract, the human-review gate, the
           ReID path reconstruction and the audit trail all work end to end, and
           that the system refuses to release without review.

  DOES NOT prove that a real tracker, a real behaviour model, or a real ReID
           embedding produces usable results. Those remain unbuilt, and no metric
           from this run should be quoted as system accuracy.

The scenario is seeded, so the run is reproducible. Coordinates are a synthetic
2-D floor plan in metres; nowhere does this claim metric truth from a camera.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventType(str, Enum):
    TRACK_START = "track_start"
    TRACK_UPDATE = "track_update"
    TRACK_END = "track_end"
    REVIEW_SUBMITTED = "review_submitted"


@dataclass(frozen=True)
class TrackObservation:
    """One position sample for a tracked person, in simulated floor-plan metres."""

    track_id: str
    timestamp: float
    x: float
    y: float
    floor: int = 0


@dataclass(frozen=True)
class Track:
    """
    A person's track within one camera session.

    `simulated` is carried on the object itself so downstream code cannot
    accidentally treat scenario data as a real observation.
    """

    track_id: str
    camera_id: str
    start_time: float
    observations: tuple[TrackObservation, ...]
    simulated: bool = True

    @property
    def end_time(self) -> float:
        return self.observations[-1].timestamp if self.observations else self.start_time

    @property
    def path(self) -> tuple[tuple[float, float], ...]:
        return tuple((o.x, o.y) for o in self.observations)


@dataclass(frozen=True)
class AppearanceEmbedding:
    """
    A stand-in for an appearance descriptor.

    A real ReID system produces a learned vector from clothing and body shape. This
    is a deterministic hash-derived unit vector, stitched together across camera
    sessions to simulate a match. It is explicitly NOT a biometric template and is
    not derived from any image.
    """

    identity_token: str  # stands in for the real person; never exposed as identity
    vector: tuple[float, ...]
    simulated: bool = True


def _embedding_for(identity_token: str, dimensions: int = 16) -> tuple[float, ...]:
    """
    Deterministic pseudo-embedding.

    Derived from a hash so the same identity yields the same vector across camera
    sessions, which is what lets the scenario simulate a cross-camera match without
    any image or model.
    """
    digest = hashlib.sha256(identity_token.encode()).digest()
    raw = [b / 255.0 - 0.5 for b in digest[:dimensions]]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return tuple(v / norm for v in raw)


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity between two embeddings, in [-1, 1]."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


class PrecursorKind(str, Enum):
    CONCEALMENT_GESTURE = "concealment_gesture"
    LOITERING = "loitering"
    REPEATED_APPROACH = "repeated_approach"


@dataclass(frozen=True)
class PrecursorAlert:
    """
    A calibrated precursor alert.

    `probability` is a probability, not a verdict. The bundled platform constraint
    is that outputs are decision-support, so the field is named to force callers to
    treat it as such, and `requires_review` is always true for the kinds modelled
    here.
    """

    alert_id: str
    track_id: str
    kind: PrecursorKind
    probability: float
    calibrated: bool
    window_start: float
    window_end: float
    evidence: tuple[str, ...]
    requires_review: bool = True
    simulated: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not self.calibrated:
            raise ValueError(
                "an alert must carry a calibrated probability; an uncalibrated "
                "score must not be presented as one"
            )


class ReviewDecision(str, Enum):
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class HumanReview:
    """
    The operator's verdict on an alert.

    Every field exists because the audit trail needs it: who decided, when, on
    what evidence, and with what reasoning. A review without a reviewer is not a
    review.
    """

    alert_id: str
    reviewer_id: str
    decision: ReviewDecision
    rationale: str
    reviewed_at: str = field(default_factory=_now)
    simulated: bool = True


class SimulatedPerceptionBackend:
    """
    Produces the scripted scenario: tracks, alerts and cross-camera ReID matches.

    Everything is deterministic given the same seed, so the acceptance run is
    reproducible and a failure is debuggable.
    """

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    # -- tracking ---------------------------------------------------------

    def generate_tracks(self) -> tuple[Track, ...]:
        """
        Two camera sessions of the same simulated person, plus a decoy.

        The split across cameras is what makes cross-camera path reconstruction a
        real test rather than one contiguous walk.
        """
        identity = f"sim-identity-{self.seed}"
        decoy = f"sim-decoy-{self.seed}"

        camera_a = self._walk(
            track_id="track-A-0001",
            camera_id="cam-a",
            identity=identity,
            start=(0.0, 0.0),
            heading=(1.0, 0.0),
            steps=12,
            start_time=0.0,
        )
        camera_b = self._walk(
            track_id="track-B-0042",
            camera_id="cam-b",
            identity=identity,
            start=(9.0, 0.5),
            heading=(1.0, 0.0),
            steps=10,
            start_time=48.0,
        )
        decoy_track = self._walk(
            track_id="track-B-0099",
            camera_id="cam-b",
            identity=decoy,
            start=(9.0, 6.0),
            heading=(-1.0, 0.0),
            steps=8,
            start_time=49.0,
        )

        # The identity is attached out-of-band, as a real system would keep it in
        # the embedding store rather than on the track object.
        self._identities = {
            camera_a.track_id: identity,
            camera_b.track_id: identity,
            decoy_track.track_id: decoy,
        }
        return (camera_a, camera_b, decoy_track)

    def _walk(
        self,
        track_id: str,
        camera_id: str,
        identity: str,
        start: tuple[float, float],
        heading: tuple[float, float],
        steps: int,
        start_time: float,
        interval: float = 1.0,
    ) -> Track:
        observations = []
        x, y = start
        for step in range(steps):
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    timestamp=start_time + step * interval,
                    x=round(x, 3),
                    y=round(y, 3),
                )
            )
            x += heading[0] * 0.35
            y += heading[1] * 0.35
        return Track(
            track_id=track_id,
            camera_id=camera_id,
            start_time=start_time,
            observations=tuple(observations),
        )

    # -- appearance ReID --------------------------------------------------

    def embedding_for(self, track_id: str) -> AppearanceEmbedding:
        identity = self._identities[track_id]
        return AppearanceEmbedding(
            identity_token=identity, vector=_embedding_for(identity)
        )

    # -- precursor detection ----------------------------------------------

    def detect_precursors(self, track: Track) -> tuple[PrecursorAlert, ...]:
        """
        Emits a scripted precursor alert for the first camera session.

        The probability is a fixed calibration-demonstrating value, not the output
        of a behaviour model. A real implementation would run a pose-sequence
        classifier here; this exists to exercise the review gate.
        """
        alerts: list[PrecursorAlert] = []
        if track.track_id != "track-A-0001":
            return ()

        alerts.append(
            PrecursorAlert(
                alert_id=f"alert-{uuid.UUID(int=self.seed).hex[:8]}-001",
                track_id=track.track_id,
                kind=PrecursorKind.CONCEALMENT_GESTURE,
                probability=0.78,
                calibrated=True,
                window_start=track.start_time + 3.0,
                window_end=track.start_time + 7.0,
                evidence=(
                    "simulated: hand-to-torso gesture repeated 3x",
                    "simulated: torso rotation toward shelving",
                ),
            )
        )
        return tuple(alerts)

    # -- audit -------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "SimulatedPerceptionBackend",
            "simulated": True,
            "seed": self.seed,
            "provides": ["tracks", "precursor_alerts", "appearance_embeddings"],
            "does_not_provide": [
                "real multi-object tracking",
                "real behaviour model",
                "real ReID embedding model",
            ],
        }


__all__ = [
    "AppearanceEmbedding",
    "EventType",
    "HumanReview",
    "PrecursorAlert",
    "PrecursorKind",
    "ReviewDecision",
    "SimulatedPerceptionBackend",
    "Track",
    "TrackObservation",
    "cosine_similarity",
]
