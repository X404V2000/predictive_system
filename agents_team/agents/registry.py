"""
Agent Registry.

Separates *registering* an agent from *instantiating* it.

Instantiating builds an agent in memory for one process. Registering records that
the agent exists as part of the team: when it was hired, under which charter, and
whether it is currently active. Without that record there is no way to answer
"when did Marcus gain the release-block permission?" or "is this the same Maya who
signed off last month?", and the audit trail from one run cannot be tied to the
agents of another.

Storage is deliberately plain:

  * **roster.json** — the current state of every agent, human-readable and
    diffable, so a changed charter is visible rather than buried in a binary.
  * **audit.jsonl** — append-only audit records, one JSON object per line. Newline
    delimited because appending is atomic and a corrupt line loses only that line.

Every registration also records a CHARTER HASH: a digest of the fields that define
an agent's authority (permissions, decision rights, responsibilities, prompt seed).
If the specification changes, the hash changes, and the registry reports drift
instead of silently re-instantiating an agent with different powers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from agents.roles import SENSITIVE_TOOLS, AgentRole, TeamSpec


class AgentStatus(str, Enum):
    """Lifecycle state of a registered agent."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class RegistryError(RuntimeError):
    """Raised when the registry cannot be read or written safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def charter_digest(role: AgentRole) -> str:
    """
    Digest of the fields that define an agent's authority.

    Title, location and email are deliberately excluded: changing a job title or an
    address does not change what the agent is allowed to do, and treating it as
    drift would bury the changes that matter. Permissions, decision rights,
    responsibilities and the prompt seed are included, because each of those
    changes the agent's behaviour.
    """
    material = {
        "id": role.id,
        "reports_to": role.reports_to,
        "tool_permissions": sorted(role.tool_permissions),
        "decision_rights": list(role.decision_rights),
        "responsibilities": list(role.responsibilities),
        "system_prompt_seed": role.system_prompt_seed,
        "memory_scope": role.memory_scope,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class RegisteredAgent:
    """One agent's persisted record."""

    agent_id: str
    name: str
    title: str
    agent_handle: str
    reports_to: str
    charter_sha256: str
    registered_at: str
    status: str = AgentStatus.ACTIVE.value
    updated_at: str = ""
    version: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == AgentStatus.ACTIVE.value


class AgentRegistry:
    """
    Persisted roster of agents.

    The registry owns the authoritative record of who is on the team. Team
    instantiation still comes from the specification, but it can now be checked
    against the roster so a charter change is detected rather than assumed benign.
    """

    ROSTER_FILENAME = "roster.json"
    AUDIT_FILENAME = "audit.jsonl"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.roster_path = self.root / self.ROSTER_FILENAME
        self.audit_path = self.root / self.AUDIT_FILENAME

    # -- storage ----------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self.roster_path.exists():
            return {"schema_version": "1.0", "agents": {}}
        try:
            data = json.loads(self.roster_path.read_text())
        except json.JSONDecodeError as exc:
            raise RegistryError(f"roster is not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or "agents" not in data:
            raise RegistryError("roster has no 'agents' key")
        # The agents block must be a mapping of id -> record. Anything else would
        # reach .values() and raise AttributeError from inside a query, which reads
        # as a crash rather than as a corrupt file.
        if not isinstance(data["agents"], dict):
            raise RegistryError(
                f"roster 'agents' must be a mapping, got {type(data['agents']).__name__}"
            )
        for agent_id, record in data["agents"].items():
            if not isinstance(record, dict):
                raise RegistryError(
                    f"roster entry {agent_id!r} must be an object, "
                    f"got {type(record).__name__}"
                )
        return data

    def _save(self, data: dict[str, Any]) -> None:
        """
        Writes the roster atomically.

        A partial write would corrupt the authoritative record of who is on the
        team, so the new content goes to a temporary file and is moved into place.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.roster_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True))
        temporary.replace(self.roster_path)

    def _append_audit(self, event: dict[str, Any]) -> None:
        """
        Appends one event to the audit log.

        Line-delimited and opened in append mode, so the previous contents are
        never rewritten. A crash mid-write can damage only the last line.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    # -- registration -----------------------------------------------------

    def register(self, role: AgentRole, actor: str = "system") -> RegisteredAgent:
        """
        Registers a role, or updates an existing registration.

        Re-registering the same agent with an unchanged charter is a no-op that
        only refreshes `updated_at`. A changed charter is recorded as a new version
        with the previous hash kept in history, so authority changes are auditable
        in their own right.
        """
        data = self._load()
        agents = data["agents"]
        digest = charter_digest(role)
        existing = agents.get(role.id)

        if existing is None:
            record = RegisteredAgent(
                agent_id=role.id,
                name=role.name,
                title=role.title,
                agent_handle=role.agent_handle,
                reports_to=role.reports_to,
                charter_sha256=digest,
                registered_at=_now(),
                updated_at=_now(),
            )
            agents[role.id] = asdict(record)
            self._save(data)
            self._append_audit(
                {
                    "event": "agent_registered",
                    "agent_id": role.id,
                    "actor": actor,
                    "at": record.registered_at,
                    "charter_sha256": digest,
                    "tool_permissions": list(role.tool_permissions),
                }
            )
            return record

        if existing["charter_sha256"] == digest:
            existing["updated_at"] = _now()
            existing["name"] = role.name
            existing["title"] = role.title
            existing["agent_handle"] = role.agent_handle
            existing["reports_to"] = role.reports_to
            self._save(data)
            self._append_audit(
                {
                    "event": "agent_reconfirmed",
                    "agent_id": role.id,
                    "actor": actor,
                    "at": _now(),
                    "charter_sha256": digest,
                }
            )
            return RegisteredAgent(**existing)

        existing.setdefault("history", []).append(
            {
                "charter_sha256": existing["charter_sha256"],
                "tool_permissions": None,
                "replaced_at": _now(),
                "version": existing.get("version", 1),
            }
        )
        previous_hash = existing["charter_sha256"]
        existing.update(
            {
                "name": role.name,
                "title": role.title,
                "agent_handle": role.agent_handle,
                "reports_to": role.reports_to,
                "charter_sha256": digest,
                "updated_at": _now(),
                "version": existing.get("version", 1) + 1,
            }
        )
        self._save(data)
        self._append_audit(
            {
                "event": "agent_charter_changed",
                "agent_id": role.id,
                "actor": actor,
                "at": _now(),
                "previous_charter_sha256": previous_hash,
                "charter_sha256": digest,
                "tool_permissions": list(role.tool_permissions),
                "version": existing["version"],
            }
        )
        return RegisteredAgent(**existing)

    def register_team(self, spec: TeamSpec, actor: str = "system") -> list[RegisteredAgent]:
        """Registers every role in a specification."""
        return [self.register(role, actor=actor) for role in spec.roles.values()]

    # -- lifecycle --------------------------------------------------------

    def set_status(
        self, agent_id: str, status: AgentStatus, actor: str = "system", reason: str = ""
    ) -> RegisteredAgent:
        """
        Changes an agent's status without deleting the record.

        Retirement is a status change rather than a deletion, because removing the
        record would break every past audit entry that refers to it.
        """
        data = self._load()
        existing = data["agents"].get(agent_id)
        if existing is None:
            raise RegistryError(f"agent {agent_id!r} is not registered")

        previous = existing.get("status", AgentStatus.ACTIVE.value)
        existing["status"] = status.value
        existing["updated_at"] = _now()
        self._save(data)
        self._append_audit(
            {
                "event": "agent_status_changed",
                "agent_id": agent_id,
                "actor": actor,
                "at": _now(),
                "from": previous,
                "to": status.value,
                "reason": reason,
            }
        )
        return RegisteredAgent(**existing)

    # -- queries ----------------------------------------------------------

    def get(self, agent_id: str) -> RegisteredAgent | None:
        record = self._load()["agents"].get(agent_id)
        return RegisteredAgent(**record) if record else None

    def by_handle(self, handle: str) -> RegisteredAgent | None:
        """Resolves an agent from its handle, e.g. '@marcus.trust'."""
        for record in self._load()["agents"].values():
            if record.get("agent_handle") == handle:
                return RegisteredAgent(**record)
        return None

    def all(self) -> list[RegisteredAgent]:
        return [RegisteredAgent(**r) for r in self._load()["agents"].values()]

    def active(self) -> list[RegisteredAgent]:
        return [r for r in self.all() if r.is_active]

    def audit_entries(self) -> list[dict[str, Any]]:
        """Reads the append-only audit log, skipping any damaged trailing line."""
        if not self.audit_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial write can only affect the final line; skip it rather
                # than discarding the whole history.
                continue
        return entries

    # -- drift ------------------------------------------------------------

    def check_privilege_conflicts(self, spec: TeamSpec) -> dict[str, list[str]]:
        """
        Reports sensitive permissions held by more than one agent.

        The registry records charter changes; it does not silently accept a change
        that splits authority. Separation of duties matters most for the release
        gate: if two agents can open it, the gate is no longer an independent
        check, and a compromise of one is enough to ship.

        This is a check rather than a hard failure, because widening a permission
        can be legitimate. It is intended to be surfaced and accepted explicitly
        by a human, not discovered later.
        """
        holders: dict[str, list[str]] = {}
        for role in spec.roles.values():
            for tool in role.tool_permissions:
                if tool in SENSITIVE_TOOLS:
                    holders.setdefault(tool, []).append(role.id)

        return {
            tool: sorted(agents)
            for tool, agents in holders.items()
            if len(agents) > 1
        }

    def detect_drift(
        self, spec: TeamSpec
    ) -> dict[str, list[str]]:
        """
        Compares a specification against the roster.

        Returns three groups: agents in the spec but not registered, agents
        registered but absent from the spec, and agents whose charter hash differs.
        This is what turns "the org chart changed" from something a human must
        notice into something the system reports.
        """
        registered = {r.agent_id: r for r in self.all()}
        specified = set(spec.roles)

        unregistered = sorted(specified - set(registered))
        removed = sorted(set(registered) - specified)
        changed = sorted(
            agent_id
            for agent_id in specified & set(registered)
            if registered[agent_id].charter_sha256 != charter_digest(spec.roles[agent_id])
        )
        return {"unregistered": unregistered, "removed": removed, "charter_changed": changed}


__all__ = [
    "AgentRegistry",
    "AgentStatus",
    "RegisteredAgent",
    "RegistryError",
    "charter_digest",
]
