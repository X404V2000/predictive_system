"""
Agent Role Definitions.

Loads and validates the team specification in `new_team.md` and turns each entry
into a typed `AgentRole`.

Everything here is derived from the spec file rather than transcribed, so a change
to `new_team.md` flows straight through to the agents. Fields that the orchestration
logic depends on are validated at load time: a role missing `system_prompt_seed` or
`tool_permissions` would otherwise produce an agent that silently behaves wrong.

The personas are fictional and exist only for local simulation, as the spec states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The specification is private data, not code: it lives in hiring_dir/ rather than
# alongside this package, so it is never tracked. Resolved through agents.config so
# the location is defined in one place and can be overridden on another machine.
from agents.config import spec_path as _spec_path

SPEC_PATH: Path = _spec_path()

# Fields every role must define. Anything the runtime reads is listed here, so a
# malformed spec fails loudly at load rather than mid-run.
REQUIRED_ROLE_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "title",
    "reports_to",
    "responsibilities",
    "decision_rights",
    "system_prompt_seed",
    "tool_permissions",
    "memory_scope",
)

# Tools that change what an agent is permitted to DO at runtime. Actions are
# checked against this set, so an agent cannot perform an action it was not hired
# for even if it is asked to.
SENSITIVE_TOOLS: frozenset[str] = frozenset(
    {"release_gate_block", "api_deploy", "dashboard_deploy", "model_registry_write"}
)


class SpecError(ValueError):
    """Raised when the team specification is missing or inconsistent."""


@dataclass(frozen=True)
class HandoffContract:
    """A required artifact moving from one agent to another."""

    source: str
    target: str
    artifact: str
    trigger: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.target)


@dataclass(frozen=True)
class AgentRole:
    """One agent's charter, as defined in the team specification."""

    id: str
    name: str
    title: str
    reports_to: str
    seniority: str
    agent_handle: str
    responsibilities: tuple[str, ...]
    decision_rights: tuple[str, ...]
    system_prompt_seed: str
    tool_permissions: tuple[str, ...]
    memory_scope: str
    deliverables: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()
    kpis: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()

    def may_use(self, tool: str) -> bool:
        """Whether this agent is permitted to use a tool."""
        return tool in self.tool_permissions

    def has_decision_right(self, needle: str) -> bool:
        """Whether any decision right contains the given phrase, case-insensitively."""
        lowered = needle.lower()
        return any(lowered in right.lower() for right in self.decision_rights)

    def system_prompt(self) -> str:
        """
        Builds the full system prompt for this agent.

        The seed from the spec is the core identity. Responsibilities, decision
        rights and memory scope are appended explicitly so the agent's operational
        boundaries are stated rather than merely implied by the persona text.
        """
        lines = [self.system_prompt_seed.strip(), ""]
        lines.append(f"You report to: {self.reports_to}.")
        lines.append("")
        lines.append("Your responsibilities:")
        lines.extend(f"- {item}" for item in self.responsibilities)
        lines.append("")
        lines.append("Your decision rights:")
        lines.extend(f"- {item}" for item in self.decision_rights)
        lines.append("")
        lines.append(f"Your memory scope (do not rely on information outside it): {self.memory_scope}")
        lines.append("")
        lines.append("Your permitted tools:")
        lines.extend(f"- {item}" for item in self.tool_permissions)
        return "\n".join(lines)


@dataclass(frozen=True)
class TeamSpec:
    """The whole team: roles, handoff contracts and acceptance artifacts."""

    team_id: str
    project_name: str
    project_goal: str
    constraints: tuple[str, ...]
    roles: dict[str, AgentRole]
    handoff_contracts: tuple[HandoffContract, ...]
    acceptance_artifacts: tuple[str, ...]
    disclaimer: str = ""
    synthetic: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def role(self, agent_id: str) -> AgentRole:
        try:
            return self.roles[agent_id]
        except KeyError as exc:
            raise SpecError(f"unknown agent id {agent_id!r}") from exc

    def contracts_from(self, agent_id: str) -> tuple[HandoffContract, ...]:
        return tuple(c for c in self.handoff_contracts if c.source == agent_id)

    def sponsors(self) -> tuple[str, ...]:
        """Agents that report outside the team."""
        return tuple(r.id for r in self.roles.values() if r.reports_to not in self.roles)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_spec(path: Path | None = None) -> TeamSpec:
    """
    Parses and validates the team specification.

    Raises SpecError for anything the runtime depends on: malformed JSON, missing
    role fields, dangling references between agents, or handoff contracts naming
    agents that do not exist.
    """
    spec_path = path or SPEC_PATH
    if not spec_path.exists():
        raise SpecError(f"team specification not found: {spec_path}")

    try:
        raw = json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:
        raise SpecError(f"{spec_path.name} is not valid JSON: {exc}") from exc

    # Validate the shape before reading any field. Without this, a specification
    # whose top level is a list, a string or null raises AttributeError from deep
    # inside the parser instead of being rejected as malformed.
    if not isinstance(raw, dict):
        raise SpecError(
            f"specification must be a JSON object, got {type(raw).__name__}"
        )
    if not isinstance(raw.get("roles"), list) or not raw["roles"]:
        raise SpecError("specification defines no roles")
    if not all(isinstance(entry, dict) for entry in raw["roles"]):
        raise SpecError("every role must be a JSON object")

    roles: dict[str, AgentRole] = {}
    for entry in raw["roles"]:
        missing = [f for f in REQUIRED_ROLE_FIELDS if not entry.get(f)]
        if missing:
            raise SpecError(
                f"role {entry.get('id', '<unknown>')!r} is missing required "
                f"field(s): {', '.join(missing)}"
            )
        role = AgentRole(
            id=entry["id"],
            name=entry["name"],
            title=entry["title"],
            reports_to=entry["reports_to"],
            seniority=str(entry.get("seniority", "")),
            agent_handle=str(entry.get("agent_handle", "")),
            responsibilities=_as_tuple(entry["responsibilities"]),
            decision_rights=_as_tuple(entry["decision_rights"]),
            system_prompt_seed=str(entry["system_prompt_seed"]),
            tool_permissions=_as_tuple(entry["tool_permissions"]),
            memory_scope=str(entry["memory_scope"]),
            deliverables=_as_tuple(entry.get("deliverables")),
            interfaces=_as_tuple(entry.get("interfaces")),
            handoffs=_as_tuple(entry.get("handoffs")),
            kpis=_as_tuple(entry.get("kpis")),
            skills=_as_tuple(entry.get("skills")),
        )
        if role.id in roles:
            raise SpecError(f"duplicate agent id {role.id!r}")
        roles[role.id] = role

    # Every reports_to must resolve, except the sponsor, who sits outside the team.
    for role in roles.values():
        if role.reports_to != "project_sponsor" and role.reports_to not in roles:
            raise SpecError(
                f"{role.id!r} reports to unknown agent {role.reports_to!r}"
            )

    contracts: list[HandoffContract] = []
    raw_contracts = raw.get("team_level_handoff_contracts", [])
    if not isinstance(raw_contracts, list):
        raise SpecError("team_level_handoff_contracts must be a list")
    for entry in raw_contracts:
        if not isinstance(entry, dict):
            raise SpecError(f"each handoff contract must be an object, got {entry!r}")
        for key in ("from", "to", "artifact", "trigger"):
            if not entry.get(key):
                raise SpecError(f"handoff contract missing {key!r}: {entry}")
        if entry["from"] not in roles:
            raise SpecError(f"handoff source {entry['from']!r} is not a team member")
        if entry["to"] not in roles:
            raise SpecError(f"handoff target {entry['to']!r} is not a team member")
        contracts.append(
            HandoffContract(
                source=entry["from"],
                target=entry["to"],
                artifact=str(entry["artifact"]),
                trigger=str(entry["trigger"]),
            )
        )

    if not contracts:
        raise SpecError("specification defines no handoff contracts")

    project = raw.get("project_context", {})
    if not isinstance(project, dict):
        project = {}
    return TeamSpec(
        team_id=str(raw.get("team_id", "")),
        project_name=str(project.get("name", "")),
        project_goal=str(project.get("goal", "")),
        constraints=_as_tuple(project.get("constraints")),
        roles=roles,
        handoff_contracts=tuple(contracts),
        acceptance_artifacts=_as_tuple(raw.get("acceptance_artifacts")),
        disclaimer=str(raw.get("disclaimer", "")),
        synthetic=bool(raw.get("synthetic", True)),
        raw=raw,
    )


__all__ = [
    "AgentRole",
    "HandoffContract",
    "REQUIRED_ROLE_FIELDS",
    "SENSITIVE_TOOLS",
    "SpecError",
    "TeamSpec",
    "load_spec",
]
