"""
Handoff Contract Enforcement.

Step 3 of the brief: an agent may not hand off until (a) the required artifact
exists and (b) the trigger condition is met. Both are checked, and both must pass.

Why both, rather than either:

  * **Artifact only** would let work proceed the moment a document appears, even
    when the gate that proves it is ready has not been passed. A spec written
    before the model is benchmarked is not the same deliverable.
  * **Trigger only** would let an agent claim readiness while pointing at nothing,
    which is exactly the failure an acceptance test exists to catch.

A blocked handoff is not silently dropped. It is returned with the reason, and it
is audited, so the orchestrator can report why work stalled.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.roles import HandoffContract
from agents.runtime import ArtifactBoard, AuditLog, Team


@dataclass(frozen=True)
class HandoffOutcome:
    """The result of attempting a handoff."""

    contract: HandoffContract
    allowed: bool
    reason: str
    artifact_present: bool
    trigger_met: bool

    def __str__(self) -> str:  # pragma: no cover - display only
        state = "ALLOWED" if self.allowed else "BLOCKED"
        return (
            f"{self.contract.source} -> {self.contract.target}: {state} "
            f"({self.reason})"
        )


class ContractEnforcer:
    """
    Checks handoff contracts against the shared board.

    Every attempt, allowed or blocked, is written to the audit log with the state
    of both requirements, so a failed handoff can be diagnosed afterwards rather
    than guessed at.
    """

    def __init__(self, team: Team) -> None:
        self.team = team
        self.board: ArtifactBoard = team.board
        self.audit: AuditLog = team.audit

    def check(
        self, contract: HandoffContract, correlation_id: str
    ) -> HandoffOutcome:
        """Evaluates one contract without changing any state."""
        artifact_present = self.board.has(contract.artifact)
        trigger_met = self.board.trigger_met(contract.trigger)

        if artifact_present and trigger_met:
            allowed, reason = True, "artifact present and trigger met"
        elif not artifact_present and not trigger_met:
            allowed, reason = False, (
                f"missing artifact {contract.artifact!r} and trigger "
                f"{contract.trigger!r} not met"
            )
        elif not artifact_present:
            allowed, reason = False, (
                f"missing artifact {contract.artifact!r} (trigger met)"
            )
        else:
            allowed, reason = False, (
                f"artifact {contract.artifact!r} exists but trigger "
                f"{contract.trigger!r} not met"
            )

        return HandoffOutcome(
            contract=contract,
            allowed=allowed,
            reason=reason,
            artifact_present=artifact_present,
            trigger_met=trigger_met,
        )

    def execute(
        self,
        source_id: str,
        target_id: str,
        correlation_id: str,
    ) -> HandoffOutcome:
        """
        Attempts the handoff between two agents.

        Raises nothing for a blocked handoff: the caller receives an outcome and
        decides how to proceed. An unknown pair is a programming error, however,
        so that raises.
        """
        matching = [
            c
            for c in self.team.spec.handoff_contracts
            if c.source == source_id and c.target == target_id
        ]
        if not matching:
            raise KeyError(
                f"no handoff contract from {source_id!r} to {target_id!r}"
            )
        contract = matching[0]

        outcome = self.check(contract, correlation_id)
        self.audit.record(
            "handoff_attempt",
            source_id,
            correlation_id,
            "allowed" if outcome.allowed else "blocked",
            target=target_id,
            artifact=contract.artifact,
            trigger=contract.trigger,
            artifact_present=outcome.artifact_present,
            trigger_met=outcome.trigger_met,
            reason=outcome.reason,
        )
        return outcome

    def check_all(self, correlation_id: str) -> list[HandoffOutcome]:
        """Evaluates every contract in the specification."""
        return [self.check(c, correlation_id) for c in self.team.spec.handoff_contracts]


__all__ = ["ContractEnforcer", "HandoffOutcome"]
