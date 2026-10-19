"""
Approach Ledger.

The rule this enforces, mechanically:

    Three DISTINCT failed approaches against the same problem halts that problem
    and escalates it to the owner.

Why "distinct" and not "three attempts". Three retries of the same fix is the loop
you asked to avoid: it burns the budget while nothing changes. An attempt is only
counted once it is registered with a stated approach, the evidence it produced, and
a claim that differs from everything already tried.

Two guards make that real rather than nominal:

  * **Retry-in-disguise detection.** Re-registering an approach whose description is
    the same, or near-identical, is refused outright. It does not consume budget,
    because it is not a new approach.
  * **Evidence is required.** An attempt without evidence of what it produced is
    refused. "I tried something and it didn't work" is not a data point.

The ledger is append-only. Nothing is deleted when a problem is escalated, so the
record of what was tried survives for the owner to read.

ESCALATION HALTS THE RUN. It raises `Halted` rather than returning a value, because
a decision from the owner is required before any further work on that problem. A
caller cannot continue by ignoring a return value, and the escalation is written to
disk before the exception is raised, so the record exists even if the process dies
immediately afterwards.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# How many distinct failed approaches before a problem is halted and escalated.
MAX_FAILED_APPROACHES: int = 3

# Two approach descriptions closer than this are treated as the same approach.
# Uses normalised token overlap, so "add retry to the parser" and "retry the parser"
# are recognised as the same idea rather than counted as progress.
SIMILARITY_THRESHOLD: float = 0.6


class AttemptOutcome(str, Enum):
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class LedgerError(RuntimeError):
    """Raised when an attempt is refused registration."""


class RetryInDisguise(LedgerError):
    """Raised when an attempt is too similar to one already made."""


class Halted(RuntimeError):
    """
    Raised when work must stop because a problem exhausted its approach budget.

    A halt is not a result to be aggregated. It means a decision is required from
    the owner before anything further is attempted, so it is raised rather than
    returned: a caller cannot ignore it, and it cannot be mistaken for a normal
    failure to retry.
    """

    def __init__(self, escalation: "Escalation") -> None:
        self.escalation = escalation
        super().__init__(
            f"halted: {escalation.problem!r} exhausted "
            f"{MAX_FAILED_APPROACHES} distinct approaches"
        )


class _StopWords:
    """Words carrying no information about what approach was taken."""

    WORDS = frozenset(
        "the a an to of and or in on for with is it this that then again try tried "
        "trying just some more also fix fixed change changed make made do did".split()
    )


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _StopWords.WORDS and len(w) > 2}


def approach_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity between two approach descriptions, in [0, 1].

    Chosen because it is explainable: the score is the overlap of meaningful words.
    A reviewer can see exactly why two attempts were judged the same.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class Attempt:
    """One recorded attempt at solving a problem."""

    problem: str
    approach: str
    outcome: AttemptOutcome
    evidence: str
    recorded_at: str = field(default_factory=_now)
    attempt_number: int = 0

    @property
    def failed(self) -> bool:
        return self.outcome is AttemptOutcome.FAILED


@dataclass
class Escalation:
    """
    A problem that exhausted its approach budget.

    Carries what was tried, so the owner receives a diagnosis rather than "it
    failed three times".
    """

    problem: str
    attempts: tuple[Attempt, ...]
    raised_at: str = field(default_factory=_now)
    guidance: str = ""

    @property
    def approaches_tried(self) -> tuple[str, ...]:
        return tuple(a.approach for a in self.attempts)

    def report(self) -> str:
        lines = [
            "ESCALATION — A PROBLEM HAS EXHAUSTED ITS APPROACH BUDGET",
            "",
            f"Problem : {self.problem}",
            f"Raised  : {self.raised_at}",
            "",
            f"{MAX_FAILED_APPROACHES} distinct approaches have failed:",
        ]
        for attempt in self.attempts:
            lines.append(
                f"  {attempt.attempt_number}. {attempt.approach}"
            )
            lines.append(f"     evidence: {attempt.evidence}")
        lines.extend(
            [
                "",
                "STOP. Repeating any of the above is not an option.",
                self.guidance
                or "A different mechanism, a different layer, or a decision about "
                "scope is required. This needs the owner.",
            ]
        )
        return "\n".join(lines)


class ApproachLedger:
    """
    Append-only record of attempts, per problem, with escalation enforcement.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._attempts: list[Attempt] = []
        self._escalations: dict[str, Escalation] = {}
        self._resolutions: dict[str, dict[str, str]] = {}
        if self.path and self.path.exists():
            self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "attempt":
                self._attempts.append(
                    Attempt(
                        problem=record["problem"],
                        approach=record["approach"],
                        outcome=AttemptOutcome(record["outcome"]),
                        evidence=record["evidence"],
                        recorded_at=record["recorded_at"],
                        attempt_number=record["attempt_number"],
                    )
                )
            elif record.get("kind") == "resolution":
                self._resolutions[record["problem"]] = {
                    "resolution": record["resolution"],
                    "resolved_by": record.get("resolved_by", ""),
                    "resolved_at": record.get("resolved_at", ""),
                }
            elif record.get("kind") == "escalation":
                attempts = tuple(
                    Attempt(
                        problem=a["problem"],
                        approach=a["approach"],
                        outcome=AttemptOutcome(a["outcome"]),
                        evidence=a["evidence"],
                        recorded_at=a["recorded_at"],
                        attempt_number=a["attempt_number"],
                    )
                    for a in record["attempts"]
                )
                self._escalations[record["problem"]] = Escalation(
                    problem=record["problem"],
                    attempts=attempts,
                    raised_at=record["raised_at"],
                    guidance=record.get("guidance", ""),
                )

    def _append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    # -- recording --------------------------------------------------------

    def record(
        self,
        problem: str,
        approach: str,
        outcome: AttemptOutcome,
        evidence: str,
        guidance: str = "",
    ) -> Attempt | Escalation:
        """
        Records an attempt.

        Raises:
            LedgerError       if the problem is already escalated, or the fields are
                              empty or unevidenced
            RetryInDisguise   if the approach duplicates one already tried

        Returns the Attempt, or an Escalation when the budget is exhausted.
        """
        problem = problem.strip()
        approach = approach.strip()
        evidence = evidence.strip()

        if not problem:
            raise LedgerError("a problem statement is required")
        if not approach:
            raise LedgerError("an approach description is required")
        if not evidence:
            raise LedgerError(
                "evidence is required: record what the attempt produced, not just "
                "that it did not work"
            )

        if problem in self._escalations:
            raise LedgerError(
                f"problem {problem!r} is already escalated and halted. "
                f"Read the escalation before continuing."
            )

        prior = self._failed_for(problem)
        for previous in prior:
            similarity = approach_similarity(previous.approach, approach)
            if similarity >= SIMILARITY_THRESHOLD:
                raise RetryInDisguise(
                    f"approach is {similarity:.0%} similar to an earlier attempt "
                    f"({previous.approach!r}). Repeating it is not a new approach, "
                    f"and it does not count toward the budget. Try a different "
                    f"mechanism."
                )

        attempt = Attempt(
            problem=problem,
            approach=approach,
            outcome=outcome,
            evidence=evidence,
            attempt_number=len(prior) + 1 if outcome is AttemptOutcome.FAILED else 1,
        )

        # A success closes the problem; it never escalates.
        if outcome is AttemptOutcome.SUCCEEDED:
            self._attempts.append(attempt)
            self._append({"kind": "attempt", **asdict(attempt)})
            return attempt

        self._attempts.append(attempt)
        self._append({"kind": "attempt", **asdict(attempt)})

        failed = self._failed_for(problem)
        if len(failed) >= MAX_FAILED_APPROACHES:
            escalation = Escalation(
                problem=problem,
                attempts=tuple(failed[:MAX_FAILED_APPROACHES]),
                guidance=guidance,
            )
            self._escalations[problem] = escalation
            self._append(
                {
                    "kind": "escalation",
                    "problem": problem,
                    "raised_at": escalation.raised_at,
                    "guidance": escalation.guidance,
                    "attempts": [asdict(a) for a in escalation.attempts],
                }
            )
            # Escalation halts the run. Returning the escalation would let a caller
            # continue past the point where a decision was required, which is the
            # failure mode this rule exists to prevent.
            raise Halted(escalation)

        return attempt

    # -- queries ----------------------------------------------------------

    def resolve(self, problem: str, resolution: str, resolved_by: str = "owner") -> None:
        """
        Closes a problem that is no longer active.

        This does NOT delete anything: the attempts remain, so the record of what
        was tried survives. What changes is that the problem stops appearing as
        open, which matters because a list of problems that are already fixed
        trains the reader to ignore the list, and that is how a real escalation
        gets missed.

        A resolution requires a substantive reason, for the same reason an override
        does: "resolved" with no explanation is indistinguishable from "forgotten".
        """
        if len(resolution.strip()) < 20:
            raise LedgerError(
                "a resolution requires a substantive reason (at least 20 characters); "
                "'fixed' does not record what was actually done"
            )

        self._resolutions[problem] = {
            "resolution": resolution.strip(),
            "resolved_by": resolved_by,
            "resolved_at": _now(),
        }
        self._append(
            {
                "kind": "resolution",
                "problem": problem,
                "resolution": resolution.strip(),
                "resolved_by": resolved_by,
                "resolved_at": self._resolutions[problem]["resolved_at"],
            }
        )

    def is_resolved(self, problem: str) -> bool:
        return problem in self._resolutions

    def resolution_of(self, problem: str) -> str | None:
        record = self._resolutions.get(problem)
        return record["resolution"] if record else None

    def open_problems(self) -> tuple[str, ...]:
        """
        Problems with attempts recorded that have not been resolved or escalated.

        This is the list a reader should act on. Resolved and escalated problems are
        reported separately so neither is hidden nor mistaken for outstanding work.
        """
        seen = {a.problem for a in self._attempts}
        return tuple(
            sorted(
                p
                for p in seen
                if not self.is_resolved(p) and not self.is_escalated(p)
            )
        )

    def _failed_for(self, problem: str) -> list[Attempt]:
        return [a for a in self._attempts if a.problem == problem and a.failed]

    def failed_attempts(self, problem: str) -> tuple[Attempt, ...]:
        return tuple(self._failed_for(problem))

    def is_escalated(self, problem: str) -> bool:
        return problem in self._escalations

    def escalation(self, problem: str) -> Escalation | None:
        return self._escalations.get(problem)

    @property
    def escalations(self) -> tuple[Escalation, ...]:
        return tuple(self._escalations.values())

    def remaining_attempts(self, problem: str) -> int:
        """How many more distinct approaches are permitted before escalation."""
        if problem in self._escalations:
            return 0
        return max(0, MAX_FAILED_APPROACHES - len(self._failed_for(problem)))

    def summary(self) -> str:
        lines = [f"APPROACH LEDGER — {len(self._attempts)} attempt(s) recorded"]
        problems = sorted({a.problem for a in self._attempts})
        for problem in problems:
            if self.is_resolved(problem):
                state = "RESOLVED"
            elif self.is_escalated(problem):
                state = "ESCALATED"
            else:
                state = "OPEN"
            lines.append(f"  [{state}] {problem}")
            for attempt in self._failed_for(problem):
                lines.append(f"     {attempt.attempt_number}. {attempt.approach}")
            resolution = self.resolution_of(problem)
            if resolution:
                lines.append(f"     resolved: {resolution}")
        if not problems:
            lines.append("  nothing recorded yet")
        return "\n".join(lines)


__all__ = [
    "ApproachLedger",
    "Halted",
    "Attempt",
    "AttemptOutcome",
    "Escalation",
    "LedgerError",
    "MAX_FAILED_APPROACHES",
    "RetryInDisguise",
    "SIMILARITY_THRESHOLD",
    "approach_similarity",
]
