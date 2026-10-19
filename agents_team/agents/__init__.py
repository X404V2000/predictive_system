"""
Agent team package.

Builds the behavioural tracking team from `new_team.md` and provides the
orchestration used by the integrated acceptance run.

This package is local simulation tooling. It is not imported by the prediction
service and is excluded from git by the `hiring_dir/` ignore rule.
"""

from agents.contracts import ContractEnforcer, HandoffOutcome
from agents.release_gate import Finding, GateDecision, ReleaseGate, Severity
from agents.roles import AgentRole, HandoffContract, SpecError, TeamSpec, load_spec
from agents.runtime import (
    Agent,
    ArtifactBoard,
    AuditEntry,
    AuditLog,
    ContractViolation,
    PermissionDenied,
    Team,
)

__all__ = [
    "Agent",
    "AgentRole",
    "ArtifactBoard",
    "AuditEntry",
    "AuditLog",
    "ContractEnforcer",
    "ContractViolation",
    "Finding",
    "GateDecision",
    "HandoffContract",
    "HandoffOutcome",
    "PermissionDenied",
    "ReleaseGate",
    "Severity",
    "SpecError",
    "Team",
    "TeamSpec",
    "load_spec",
]
