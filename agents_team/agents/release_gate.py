"""
Release Gate.

Step 4 of the brief: Marcus Reed gates releases, with `release_gate_block`.

The gate is not a checklist an agent can mark satisfied itself. It is enforced in
three ways:

  1. **Permission.** `release_gate_block` must appear in the agent's charter. A
     request from any other agent is refused and audited.
  2. **Evidence.** The gate cannot open while any required artifact is missing or
     any handoff contract remains blocked, so "everything passed" cannot be
     asserted without the board agreeing.
  3. **Open findings.** Any unresolved high or critical finding blocks the release
     regardless of paperwork. Marcus's charter says he may block on unresolved
     privacy or security risk, and this is where that takes effect.

The gate can only ever be opened by Marcus, and each decision is audited with its
reasoning, including refusals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agents.contracts import ContractEnforcer, HandoffOutcome
from agents.runtime import AuditLog, PermissionDenied, Team


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


BLOCKING_SEVERITIES: frozenset[Severity] = frozenset({Severity.HIGH, Severity.CRITICAL})

GATE_TOOL = "release_gate_block"


@dataclass(frozen=True)
class Finding:
    """An open review finding that may block a release."""

    finding_id: str
    severity: Severity
    summary: str
    raised_by: str
    resolved: bool = False

    @property
    def blocks(self) -> bool:
        return not self.resolved and self.severity in BLOCKING_SEVERITIES


@dataclass
class GateDecision:
    """Outcome of a release gate evaluation."""

    released: bool
    reason: str
    open_findings: tuple[Finding, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    blocked_handoffs: tuple[str, ...] = ()
    evaluated_by: str = ""
    correlation_id: str = ""

    @property
    def blocked(self) -> bool:
        return not self.released


class ReleaseGate:
    """
    Marcus Reed's release gate.

    Holds the open findings and evaluates whether a release candidate may ship.
    """

    GATEKEEPER_ID = "marcus_reed"

    def __init__(self, team: Team, required_artifacts: tuple[str, ...] = ()) -> None:
        self.team = team
        self.audit: AuditLog = team.audit
        self.enforcer = ContractEnforcer(team)
        self.required_artifacts = required_artifacts
        self._findings: list[Finding] = []

    # -- findings ---------------------------------------------------------

    def raise_finding(
        self,
        finding_id: str,
        severity: Severity,
        summary: str,
        raised_by: str,
        correlation_id: str,
    ) -> Finding:
        finding = Finding(
            finding_id=finding_id,
            severity=severity,
            summary=summary,
            raised_by=raised_by,
            resolved=False,
        )
        self._findings.append(finding)
        self.audit.record(
            "finding_raised",
            raised_by,
            correlation_id,
            "ok",
            finding_id=finding_id,
            severity=severity.value,
        )
        return finding

    def resolve_finding(self, finding_id: str, resolved_by: str, correlation_id: str) -> None:
        """
        Resolves a finding.

        Resolution is recorded rather than deleting the finding, so the history of
        what was found and when it was cleared remains auditable.
        """
        for index, finding in enumerate(self._findings):
            if finding.finding_id == finding_id:
                replacement = Finding(
                    finding_id=finding.finding_id,
                    severity=finding.severity,
                    summary=finding.summary,
                    raised_by=finding.raised_by,
                    resolved=True,
                )
                self._findings[index] = replacement
                self.audit.record(
                    "finding_resolved",
                    resolved_by,
                    correlation_id,
                    "ok",
                    finding_id=finding_id,
                )
                return
        raise KeyError(f"unknown finding {finding_id!r}")

    @property
    def open_findings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self._findings if not f.resolved)

    @property
    def blocking_findings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self._findings if f.blocks)

    # -- gate -------------------------------------------------------------

    def evaluate(self, correlation_id: str) -> GateDecision:
        """
        Evaluates the gate WITHOUT deciding to release.

        Separated from `release` so the evidence can be inspected before a decision
        is recorded against an agent's name.
        """
        missing = tuple(
            name for name in self.required_artifacts if not self.team.board.has(name)
        )
        # Record each contract evaluation, not just the aggregate verdict. A
        # release decision must be reconstructable: "all gates satisfied" with no
        # trail of what was checked is an assertion, not evidence.
        handoffs: list[HandoffOutcome] = []
        for contract in self.team.spec.handoff_contracts:
            handoffs.append(self.enforcer.execute(contract.source, contract.target, correlation_id))
        blocked = tuple(
            f"{o.contract.source}->{o.contract.target}" for o in handoffs if not o.allowed
        )
        blocking = self.blocking_findings

        reasons: list[str] = []
        if missing:
            reasons.append(f"{len(missing)} required artifact(s) missing")
        if blocked:
            reasons.append(f"{len(blocked)} handoff contract(s) blocked")
        if blocking:
            reasons.append(
                f"{len(blocking)} unresolved high/critical finding(s): "
                + ", ".join(f.finding_id for f in blocking)
            )

        released = not reasons
        reason = "all gates satisfied" if released else "; ".join(reasons)

        return GateDecision(
            released=released,
            reason=reason,
            open_findings=self.open_findings,
            missing_artifacts=missing,
            blocked_handoffs=blocked,
            evaluated_by=self.GATEKEEPER_ID,
            correlation_id=correlation_id,
        )

    def release(self, agent_id: str, correlation_id: str) -> GateDecision:
        """
        Attempts to release, on behalf of `agent_id`.

        Only the gatekeeper may open the gate, and only if it holds the
        `release_gate_block` permission in its charter. Both checks are audited on
        failure.
        """
        agent = self.team.agent(agent_id)

        if agent_id != self.GATEKEEPER_ID:
            self.audit.record(
                "release_denied",
                agent_id,
                correlation_id,
                "denied",
                reason="only the gatekeeper may open the release gate",
            )
            raise PermissionDenied(
                f"only {self.GATEKEEPER_ID} may open the release gate; "
                f"{agent_id} attempted it"
            )

        agent.require_tool(GATE_TOOL, correlation_id)

        decision = self.evaluate(correlation_id)
        self.audit.record(
            "release_decision",
            agent_id,
            correlation_id,
            "released" if decision.released else "blocked",
            reason=decision.reason,
            missing_artifacts=list(decision.missing_artifacts),
            blocked_handoffs=list(decision.blocked_handoffs),
            blocking_findings=[f.finding_id for f in decision.open_findings if f.blocks],
        )
        return decision


__all__ = [
    "BLOCKING_SEVERITIES",
    "Finding",
    "GATE_TOOL",
    "GateDecision",
    "ReleaseGate",
    "Severity",
]
