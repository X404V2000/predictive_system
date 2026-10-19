"""
Agent Runtime.

Instantiates one agent per role from the team specification, and provides the
shared machinery the orchestration needs:

  * **Artifact board** — the record of what has actually been produced. Handoff
    contracts are checked against this, so an agent cannot claim to hand off a
    deliverable that does not exist.
  * **Trigger registry** — the conditions that permit a handoff. A contract fires
    only when its trigger has been explicitly met, which prevents "the artifact
    exists" from being treated as "the work is ready".
  * **Audit log** — append-only record of every action, with the agent, the tool,
    the outcome and a correlation id. Nothing happens here without being logged.
  * **Tool permissions** — an agent may only use tools listed in its charter.

This module deliberately does NOT call a language model. The agents here are
deterministic actors: each method corresponds to a permitted action, and the
system_prompt is available for a model-backed agent to consume later. Running the
orchestration without a model makes the handoff and auditing logic testable and
reproducible, which is what the acceptance test needs to verify.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agents.roles import AgentRole, SpecError, TeamSpec


class PermissionDenied(RuntimeError):
    """Raised when an agent attempts an action outside its charter."""


class ContractViolation(RuntimeError):
    """Raised when a handoff is attempted without its artifact or trigger."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --- audit log -------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """One recorded event. Immutable once written."""

    sequence: int
    timestamp: str
    action: str
    agent_id: str | None
    correlation_id: str
    outcome: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "action": self.action,
            "agent_id": self.agent_id,
            "correlation_id": self.correlation_id,
            "outcome": self.outcome,
            "detail": self.detail,
        }


class AuditLog:
    """
    Append-only audit trail.

    Entries are never mutated or removed once written, and each carries a sequence
    number so gaps would be detectable. Every privileged action in this system goes
    through `record`, including refusals: a denied action is itself auditable.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(
        self,
        action: str,
        agent_id: str | None,
        correlation_id: str,
        outcome: str,
        **detail: Any,
    ) -> AuditEntry:
        entry = AuditEntry(
            sequence=len(self._entries) + 1,
            timestamp=_now(),
            action=action,
            agent_id=agent_id,
            correlation_id=correlation_id,
            outcome=outcome,
            detail=detail,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def for_correlation(self, correlation_id: str) -> tuple[AuditEntry, ...]:
        return tuple(e for e in self._entries if e.correlation_id == correlation_id)

    def actions_by(self, agent_id: str) -> tuple[AuditEntry, ...]:
        return tuple(e for e in self._entries if e.agent_id == agent_id)

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self._entries], indent=2)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    def __len__(self) -> int:
        return len(self._entries)


# --- shared board ----------------------------------------------------------


class ArtifactBoard:
    """
    The record of produced deliverables and met triggers.

    Kept separate from the agents so that no agent can assert its own compliance.
    An artifact exists only once something has been produced under that name, and a
    trigger is met only once the thing that satisfies it has happened.
    """

    def __init__(self, audit: AuditLog) -> None:
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._triggers: dict[str, str] = {}
        self._audit = audit

    def produce(
        self,
        name: str,
        owner: str,
        correlation_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Records that an artifact now exists."""
        self._artifacts[name] = {
            "owner": owner,
            "produced_at": _now(),
            "payload": payload or {},
        }
        self._audit.record(
            "artifact_produced", owner, correlation_id, "ok", artifact=name
        )

    def has(self, name: str) -> bool:
        return name in self._artifacts

    def get(self, name: str) -> dict[str, Any] | None:
        return self._artifacts.get(name)

    def fire_trigger(self, trigger: str, agent_id: str, correlation_id: str) -> None:
        """Records that a handoff trigger condition has been met."""
        self._triggers[trigger] = agent_id
        self._audit.record(
            "trigger_fired", agent_id, correlation_id, "ok", trigger=trigger
        )

    def trigger_met(self, trigger: str) -> bool:
        return trigger in self._triggers

    @property
    def artifacts(self) -> tuple[str, ...]:
        return tuple(self._artifacts)

    @property
    def triggers(self) -> tuple[str, ...]:
        return tuple(self._triggers)


# --- agent -----------------------------------------------------------------


@dataclass
class Agent:
    """
    One instantiated team member.

    Constructed entirely from its charter, so the runtime behaviour cannot drift
    from the specification. `system_prompt` is exposed for a model-backed agent to
    consume; this class does not call a model.
    """

    role: AgentRole
    board: ArtifactBoard
    audit: AuditLog

    @property
    def id(self) -> str:
        return self.role.id

    @property
    def system_prompt(self) -> str:
        return self.role.system_prompt()

    def require_tool(self, tool: str, correlation_id: str) -> None:
        """
        Enforces the charter's tool permissions.

        A refused attempt is audited as well as raised: an agent repeatedly trying
        actions outside its charter is itself a signal worth keeping.
        """
        if not self.role.may_use(tool):
            self.audit.record(
                "permission_denied",
                self.id,
                correlation_id,
                "denied",
                tool=tool,
                permitted=list(self.role.tool_permissions),
            )
            raise PermissionDenied(
                f"{self.id} is not permitted to use {tool!r}; permitted tools are "
                f"{list(self.role.tool_permissions)}"
            )

    def produce_artifact(
        self,
        name: str,
        correlation_id: str,
        tool: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Produces a deliverable, subject to the charter's tool permissions."""
        self.require_tool(tool, correlation_id)
        self.board.produce(name, self.id, correlation_id, payload)
        self.audit.record(
            "artifact_claimed", self.id, correlation_id, "ok", artifact=name, tool=tool
        )

    def declare_decision(
        self, decision: str, correlation_id: str, **detail: Any
    ) -> None:
        """Records a decision this agent is entitled to make."""
        if not self.role.decision_rights:
            self.audit.record(
                "decision_denied", self.id, correlation_id, "denied", decision=decision
            )
            raise PermissionDenied(f"{self.id} holds no decision rights")

        self.audit.record(
            "decision", self.id, correlation_id, "ok", decision=decision, **detail
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<Agent {self.id} ({self.role.title})>"


# --- team ------------------------------------------------------------------


class Team:
    """
    All instantiated agents plus the shared board and audit log.

    The Team is the only holder of the artifact board, so agents cannot produce
    artifacts through a private channel that the orchestrator would not see.
    """

    ORCHESTRATOR_ID = "aisha_rahman"

    def __init__(self, spec: TeamSpec) -> None:
        self.spec = spec
        self.audit = AuditLog()
        self.board = ArtifactBoard(self.audit)
        self.agents: dict[str, Agent] = {
            role_id: Agent(role=role, board=self.board, audit=self.audit)
            for role_id, role in spec.roles.items()
        }

        if self.ORCHESTRATOR_ID not in self.agents:
            raise SpecError(
                f"orchestrator {self.ORCHESTRATOR_ID!r} is not in the specification"
            )

    def agent(self, agent_id: str) -> Agent:
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            raise SpecError(f"unknown agent {agent_id!r}") from exc

    @property
    def orchestrator(self) -> Agent:
        return self.agents[self.ORCHESTRATOR_ID]

    def new_correlation_id(self, purpose: str) -> str:
        return f"{purpose}:{uuid.uuid4().hex[:12]}"


__all__ = [
    "Agent",
    "ArtifactBoard",
    "AuditEntry",
    "AuditLog",
    "ContractViolation",
    "PermissionDenied",
    "Team",
]
