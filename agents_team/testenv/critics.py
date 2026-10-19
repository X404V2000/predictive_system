"""
Critic Agents.

Critics exist to find faults, not to confirm work. Each one has a single job and a
hard rule: it may pass, or it may BLOCK. It cannot be talked out of a block.

What "cannot be talked out of it" means concretely:

  * A critic's verdict is a function of EVIDENCE (test results, mutation results,
    measured invariants), not of a description of the work. There is no code path
    that lets prose change an outcome.
  * A reviewer who disagrees must file an OVERRIDE with a written reason. The
    override is recorded next to the block, permanently, and the report shows the
    count of overrides prominently.
  * An override does not erase the finding. The finding stays in the record.

This is deliberately uncomfortable. A system that logs how often its own checks were
bypassed is more trustworthy than one where disagreement leaves no trace.

Critics are not LLM calls. They are deterministic checks whose outcome is evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Verdict(str, Enum):
    PASS = "pass"
    BLOCK = "block"


class CriticRole(str, Enum):
    """Each critic attacks a different axis. One passing critic proves nothing."""

    CORRECTNESS = "correctness"
    ROBUSTNESS = "robustness"
    SECURITY = "security"
    HONESTY = "honesty"


@dataclass(frozen=True)
class Finding:
    """One defect a critic is prepared to block on."""

    code: str
    summary: str
    evidence: str
    severity: str = "high"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.severity}] {self.code}: {self.summary}"


@dataclass(frozen=True)
class CriticReport:
    """A critic's verdict on one target."""

    critic: CriticRole
    target: str
    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    checks_run: int = 0
    note: str = ""

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK


@dataclass
class Override:
    """
    A human (or the lead agent) disagreeing with a block.

    Recorded permanently with its reason. An override without a substantive reason
    is refused, because "I disagree" is not evidence.
    """

    critic: CriticRole
    target: str
    finding_codes: tuple[str, ...]
    overridden_by: str
    reason: str
    at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if len(self.reason.strip()) < 20:
            raise ValueError(
                "an override requires a substantive reason (at least 20 characters); "
                "a block must be answered with evidence or reasoning, not dismissal"
            )


class Critic:
    """
    A blocking critic.

    Subclasses implement `_check`, which returns findings. The verdict is derived
    from whether any findings exist. There is intentionally no method to set a
    verdict by hand.
    """

    role: CriticRole

    def evaluate(self, target: str, **evidence: Any) -> CriticReport:
        findings = tuple(self._check(**evidence))
        return CriticReport(
            critic=self.role,
            target=target,
            verdict=Verdict.BLOCK if findings else Verdict.PASS,
            findings=findings,
            checks_run=len(evidence.get("checks", []) or []) or 1,
            note=self.describe(),
        )

    def _check(self, **evidence: Any) -> Sequence[Finding]:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        return self.__class__.__doc__ or ""


class CorrectnessCritic(Critic):
    """
    Does the thing actually work as specified?

    Blocks on any failing test, and on a suspicious absence of tests.
    """

    role = CriticRole.CORRECTNESS

    def _check(self, **evidence: Any) -> Sequence[Finding]:
        findings: list[Finding] = []
        failures = evidence.get("failures", []) or []
        total = evidence.get("total", 0) or 0

        if failures:
            findings.append(
                Finding(
                    "COR-001",
                    f"{len(failures)} test(s) failed",
                    "; ".join(str(f)[:120] for f in failures[:5]),
                )
            )
        if total == 0:
            findings.append(
                Finding(
                    "COR-002",
                    "no tests were executed",
                    "an empty run looks like success and is not",
                )
            )
        return findings


class RobustnessCritic(Critic):
    """
    Does it survive hostile input?

    Blocks when generated inputs cause uncaught exceptions where a defined error was
    expected, or when the code silently accepts input it should reject.
    """

    role = CriticRole.ROBUSTNESS

    def _check(self, **evidence: Any) -> Sequence[Finding]:
        findings: list[Finding] = []
        crashes = evidence.get("uncaught_exceptions", []) or []
        silent_accepts = evidence.get("silent_accepts", []) or []

        if crashes:
            findings.append(
                Finding(
                    "ROB-001",
                    f"{len(crashes)} uncaught exception(s) on generated input",
                    "; ".join(str(c)[:140] for c in crashes[:5]),
                )
            )
        if silent_accepts:
            findings.append(
                Finding(
                    "ROB-002",
                    f"{len(silent_accepts)} invalid input(s) accepted silently",
                    "; ".join(str(s)[:140] for s in silent_accepts[:5]),
                )
            )
        return findings


class SecurityCritic(Critic):
    """
    Can authority be gained or leaked?

    Blocks when a privileged action succeeds for an agent that should not hold it,
    or when secret-shaped strings reach disk.
    """

    role = CriticRole.SECURITY

    def _check(self, **evidence: Any) -> Sequence[Finding]:
        findings: list[Finding] = []
        escalations = evidence.get("escalations", []) or []
        leaks = evidence.get("leaks", []) or []

        if escalations:
            findings.append(
                Finding(
                    "SEC-001",
                    f"{len(escalations)} privilege violation(s)",
                    "; ".join(str(e)[:140] for e in escalations[:5]),
                    severity="critical",
                )
            )
        if leaks:
            findings.append(
                Finding(
                    "SEC-002",
                    f"{len(leaks)} secret-shaped value(s) written to a file",
                    "; ".join(str(l)[:100] for l in leaks[:3]),
                    severity="critical",
                )
            )
        return findings


class HonestyCritic(Critic):
    """
    Does the report overstate what was proven?

    This is the critic against yes-men, including against me. It blocks when output
    claims accuracy, certainty, or completion that the evidence does not support,
    and when simulations are not labelled.
    """

    role = CriticRole.HONESTY

    # Words that assert system capability. Permitted only when paired with evidence.
    OVERCLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
        ("100% accurate", "HON-001"),
        ("fully validated", "HON-002"),
        ("production ready", "HON-003"),
        ("proven to work", "HON-004"),
        ("all requirements met", "HON-005"),
        ("no known issues", "HON-006"),
    )

    def _check(self, **evidence: Any) -> Sequence[Finding]:
        findings: list[Finding] = []
        reports = evidence.get("reports", []) or []
        unlabelled_outputs = evidence.get("unlabelled_simulated", []) or []

        for report in reports:
            text = str(report).lower()
            for phrase, code in self.OVERCLAIM_PATTERNS:
                if phrase in text:
                    findings.append(
                        Finding(
                            code,
                            f"unsubstantiated claim: {phrase!r}",
                            f"appears in output: {str(report)[:100]}",
                        )
                    )

        if unlabelled_outputs:
            findings.append(
                Finding(
                    "HON-007",
                    f"{len(unlabelled_outputs)} simulated output(s) not labelled",
                    "scenario data presented without a simulated marker can be "
                    "mistaken for real measurement",
                )
            )
        return findings


class CriticPanel:
    """
    All critics, applied together.

    A change passes only if EVERY critic passes. One critic's approval says nothing
    about the others: correctness does not imply robustness, and neither implies
    honesty.
    """

    def __init__(self) -> None:
        self.critics: tuple[Critic, ...] = (
            CorrectnessCritic(),
            RobustnessCritic(),
            SecurityCritic(),
            HonestyCritic(),
        )

    def review(self, target: str, **evidence: Any) -> tuple[CriticReport, ...]:
        return tuple(c.evaluate(target, **evidence) for c in self.critics)

    def blocked(self, reports: Sequence[CriticReport]) -> bool:
        return any(r.blocked for r in reports)

    def summary(self, reports: Sequence[CriticReport]) -> str:
        lines = [f"CRITIC REVIEW: {reports[0].target if reports else '?'}"]
        for report in reports:
            state = "BLOCK" if report.blocked else "PASS "
            lines.append(f"  [{state}] {report.critic.value}")
            for finding in report.findings:
                lines.append(f"           {finding}")
                lines.append(f"           evidence: {finding.evidence}")
        return "\n".join(lines)


def fingerprint(text: str) -> str:
    """Stable digest, used to prove two reports describe the same work."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


__all__ = [
    "CorrectnessCritic",
    "Critic",
    "CriticPanel",
    "CriticReport",
    "CriticRole",
    "Finding",
    "HonestyCritic",
    "Override",
    "RobustnessCritic",
    "SecurityCritic",
    "Verdict",
    "fingerprint",
]
