"""
Tests for the agent registry.

Registration is what separates "an agent existed in one process" from "this agent
is part of the team, under this charter, since this date". The tests here focus on
the properties that make that distinction meaningful:

  * the roster survives a restart
  * a charter change is detected rather than silently accepted
  * authority changes are versioned and audited
  * retirement preserves history instead of deleting records
  * a sensitive permission spreading to a second agent is reported
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agents.registry import (
    AgentRegistry,
    AgentStatus,
    RegistryError,
    charter_digest,
)
from agents.roles import SPEC_PATH, load_spec


def _spec_with_change(agent_id: str, mutate) -> Path:
    """
    Writes a copy of the specification with one role modified.

    Returned as a path in a temporary directory so the real spec is untouched.
    """
    raw = json.loads(SPEC_PATH.read_text())
    for role in raw["roles"]:
        if role["id"] == agent_id:
            mutate(role)
    target = SPEC_PATH.parent / f"_test_spec_{agent_id}.md"
    target.write_text(json.dumps(raw))
    return target


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    return AgentRegistry(tmp_path / "store")


# --- registration ----------------------------------------------------------


def test_every_role_registers(registry: AgentRegistry) -> None:
    spec = load_spec()
    records = registry.register_team(spec)

    assert len(records) == len(spec.roles) == 7
    assert {r.agent_id for r in records} == set(spec.roles)
    assert all(r.status == AgentStatus.ACTIVE.value for r in records)


def test_registration_persists_to_disk(registry: AgentRegistry) -> None:
    """A new registry object over the same directory must see the same roster."""
    spec = load_spec()
    registry.register_team(spec)

    reopened = AgentRegistry(registry.root)
    assert len(reopened.all()) == 7
    assert reopened.get("marcus_reed") is not None


def test_lookup_by_handle(registry: AgentRegistry) -> None:
    registry.register_team(load_spec())

    assert registry.by_handle("@marcus.trust").agent_id == "marcus_reed"
    assert registry.by_handle("@nobody.here") is None


def test_registration_is_recorded_in_the_audit_log(registry: AgentRegistry) -> None:
    registry.register_team(load_spec(), actor="owner")
    events = registry.audit_entries()

    assert len(events) == 7
    assert all(e["event"] == "agent_registered" for e in events)
    assert all(e["actor"] == "owner" for e in events)


def test_reregistering_an_unchanged_charter_is_not_a_new_version(
    registry: AgentRegistry,
) -> None:
    """
    Re-running registration must be idempotent for the version.

    Otherwise every startup would look like an authority change in the audit log,
    and real changes would be impossible to spot among the noise.
    """
    spec = load_spec()
    registry.register_team(spec)
    first = registry.get("priya_nair")

    registry.register_team(spec)
    second = registry.get("priya_nair")

    assert second.version == first.version == 1
    assert second.charter_sha256 == first.charter_sha256
    assert second.registered_at == first.registered_at

    events = [e["event"] for e in registry.audit_entries()]
    assert "agent_charter_changed" not in events


# --- charter drift ---------------------------------------------------------


def test_charter_digest_ignores_cosmetic_fields() -> None:
    """
    A title change must not count as a change of authority.

    Hashing title or email would make the digest fail on changes that confer no new
    power, burying the changes that do.
    """
    spec = load_spec()
    role = spec.role("maya_chen")
    before = charter_digest(role)

    renamed = type(role)(
        id=role.id,
        name=role.name,
        title="Distinguished Vision Lead",
        reports_to=role.reports_to,
        seniority=role.seniority,
        agent_handle=role.agent_handle,
        responsibilities=role.responsibilities,
        decision_rights=role.decision_rights,
        system_prompt_seed=role.system_prompt_seed,
        tool_permissions=role.tool_permissions,
        memory_scope=role.memory_scope,
    )
    assert charter_digest(renamed) == before


def test_charter_digest_changes_when_permissions_change() -> None:
    spec = load_spec()
    role = spec.role("priya_nair")

    widened = type(role)(
        id=role.id,
        name=role.name,
        title=role.title,
        reports_to=role.reports_to,
        seniority=role.seniority,
        agent_handle=role.agent_handle,
        responsibilities=role.responsibilities,
        decision_rights=role.decision_rights,
        system_prompt_seed=role.system_prompt_seed,
        tool_permissions=role.tool_permissions + ("release_gate_block",),
        memory_scope=role.memory_scope,
    )
    assert charter_digest(widened) != charter_digest(role)


def test_drift_reports_a_changed_charter(registry: AgentRegistry, tmp_path: Path) -> None:
    spec = load_spec()
    registry.register_team(spec)

    def grant_release_block(role: dict) -> None:
        role["tool_permissions"].append("release_gate_block")

    mutated_path = _spec_with_change("priya_nair", grant_release_block)
    try:
        mutated = load_spec(mutated_path)
        drift = registry.detect_drift(mutated)
        assert drift["charter_changed"] == ["priya_nair"]
        assert drift["removed"] == []
        assert drift["unregistered"] == []
    finally:
        mutated_path.unlink(missing_ok=True)


def test_drift_reports_unregistered_and_removed(
    registry: AgentRegistry,
) -> None:
    spec = load_spec()
    # Register everyone except one.
    for role_id, role in spec.roles.items():
        if role_id != "aisha_rahman":
            registry.register(role)

    drift = registry.detect_drift(spec)
    assert drift["unregistered"] == ["aisha_rahman"]

    # Now shrink the spec so a registered agent is no longer specified.
    #
    # The role must be removed together with every contract that references it:
    # load_spec rejects dangling references, which is the validator working as
    # intended. Editing only the role list would fail for the right reason, so the
    # contracts are pruned in step.
    raw = json.loads(SPEC_PATH.read_text())
    raw["roles"] = [r for r in raw["roles"] if r["id"] != "marcus_reed"]
    raw["team_level_handoff_contracts"] = [
        c
        for c in raw["team_level_handoff_contracts"]
        if c["from"] != "marcus_reed" and c["to"] != "marcus_reed"
    ]
    reduced_path = SPEC_PATH.parent / "_test_spec_reduced.md"
    reduced_path.write_text(json.dumps(raw))
    try:
        reduced = load_spec(reduced_path)
        assert "marcus_reed" in registry.detect_drift(reduced)["removed"]
    finally:
        reduced_path.unlink(missing_ok=True)


def test_spec_rejects_a_contract_referencing_an_absent_agent() -> None:
    """
    A handoff contract naming someone who is not on the team is a specification
    error, not something to be tolerated at runtime.
    """
    raw = json.loads(SPEC_PATH.read_text())
    raw["roles"] = [r for r in raw["roles"] if r["id"] != "marcus_reed"]
    # Contracts are deliberately NOT pruned here.
    broken_path = SPEC_PATH.parent / "_test_spec_broken.md"
    broken_path.write_text(json.dumps(raw))
    try:
        with pytest.raises(Exception, match="not a team member"):
            load_spec(broken_path)
    finally:
        broken_path.unlink(missing_ok=True)


def test_rechartering_bumps_version_and_keeps_history(
    registry: AgentRegistry,
) -> None:
    spec = load_spec()
    registry.register_team(spec)
    original_hash = registry.get("priya_nair").charter_sha256

    def grant_release_block(role: dict) -> None:
        role["tool_permissions"].append("release_gate_block")

    mutated_path = _spec_with_change("priya_nair", grant_release_block)
    try:
        mutated = load_spec(mutated_path)
        updated = registry.register(mutated.roles["priya_nair"], actor="owner")
    finally:
        mutated_path.unlink(missing_ok=True)

    assert updated.version == 2
    assert updated.charter_sha256 != original_hash
    assert len(updated.history) == 1
    assert updated.history[0]["charter_sha256"] == original_hash

    events = [e for e in registry.audit_entries() if e["event"] == "agent_charter_changed"]
    assert len(events) == 1
    assert events[0]["previous_charter_sha256"] == original_hash


# --- privilege separation --------------------------------------------------


def test_no_single_agent_holds_release_block_in_the_spec() -> None:
    spec = load_spec()
    holders = [r.id for r in spec.roles.values() if "release_gate_block" in r.tool_permissions]
    assert holders == ["marcus_reed"]


def test_privilege_conflict_detects_a_second_gate_keeper(
    registry: AgentRegistry,
) -> None:
    """
    If two agents can open the release gate, it is no longer an independent check.
    The registry must report that, even though widening a permission can be a
    deliberate decision.
    """
    spec = load_spec()
    assert "release_gate_block" not in registry.check_privilege_conflicts(spec)

    def grant_release_block(role: dict) -> None:
        role["tool_permissions"].append("release_gate_block")

    mutated_path = _spec_with_change("priya_nair", grant_release_block)
    try:
        mutated = load_spec(mutated_path)
        conflicts = registry.check_privilege_conflicts(mutated)
        assert "release_gate_block" in conflicts
        assert conflicts["release_gate_block"] == ["marcus_reed", "priya_nair"]
    finally:
        mutated_path.unlink(missing_ok=True)


# --- lifecycle -------------------------------------------------------------


def test_retirement_preserves_the_record(registry: AgentRegistry) -> None:
    """
    Retiring must not delete the record: past audit entries refer to the agent, and
    removing it would orphan them.
    """
    registry.register_team(load_spec())
    retired = registry.set_status(
        "marcus_reed", AgentStatus.RETIRED, actor="owner", reason="left the team"
    )

    assert retired.status == AgentStatus.RETIRED.value
    assert not retired.is_active
    assert registry.get("marcus_reed") is not None
    assert "marcus_reed" not in {r.agent_id for r in registry.active()}

    events = [e for e in registry.audit_entries() if e["event"] == "agent_status_changed"]
    assert events[0]["to"] == "retired"
    assert events[0]["reason"] == "left the team"


def test_status_change_on_unknown_agent_raises(registry: AgentRegistry) -> None:
    with pytest.raises(RegistryError, match="not registered"):
        registry.set_status("ghost_agent", AgentStatus.SUSPENDED)


def test_suspension_is_distinct_from_retirement(registry: AgentRegistry) -> None:
    registry.register_team(load_spec())
    registry.set_status("jonah_weiss", AgentStatus.SUSPENDED)

    record = registry.get("jonah_weiss")
    assert record.status == AgentStatus.SUSPENDED.value
    assert not record.is_active


# --- durability ------------------------------------------------------------


def test_corrupt_roster_is_reported(registry: AgentRegistry) -> None:
    """A damaged roster must raise clearly rather than fall back to empty."""
    registry.root.mkdir(parents=True, exist_ok=True)
    registry.roster_path.write_text("{ not json")
    with pytest.raises(RegistryError, match="not valid JSON"):
        registry.all()


def test_damaged_audit_line_does_not_discard_history(registry: AgentRegistry) -> None:
    """
    A partial final write must cost only the damaged line.

    The audit log is the record of what happened; silently dropping it because one
    append was interrupted would undermine the whole trail.
    """
    registry.register_team(load_spec())
    good = len(registry.audit_entries())

    with registry.audit_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "truncated"')

    entries = registry.audit_entries()
    assert len(entries) == good
    assert all("event" in e for e in entries)


def test_roster_file_is_human_readable(registry: AgentRegistry) -> None:
    """The roster is meant to be inspected and diffed by a person."""
    registry.register_team(load_spec())
    text = registry.roster_path.read_text()

    assert "\n" in text
    assert "marcus_reed" in text
    parsed = json.loads(text)
    assert parsed["agents"]["marcus_reed"]["charter_sha256"]
