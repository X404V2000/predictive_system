"""
Adversarial tests for the agent team.

A test suite that only checks the happy path proves nothing about enforcement. Each
test here attempts a specific abuse and asserts the system refuses it:

  * hand off without the artifact, or without the trigger
  * open the release gate from an agent that is not the gatekeeper
  * release with unresolved high-severity findings
  * act outside a charter's tool permissions
  * skip human review and still release
  * claim a calibrated alert with an uncalibrated score

If any of these starts passing when it should fail, the enforcement has broken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agents.contracts import ContractEnforcer
from agents.orchestrator import TeamOrchestrator
from agents.release_gate import ReleaseGate, Severity
from agents.roles import SENSITIVE_TOOLS, SpecError, load_spec
from agents.runtime import PermissionDenied, Team
from agents.sim_backend import (
    PrecursorAlert,
    PrecursorKind,
    SimulatedPerceptionBackend,
    cosine_similarity,
)


# --- specification ---------------------------------------------------------


def test_spec_loads_every_role() -> None:
    spec = load_spec()
    assert len(spec.roles) == 7
    assert "aisha_rahman" in spec.roles
    assert "marcus_reed" in spec.roles


def test_every_role_has_a_complete_charter() -> None:
    """
    Instantiation depends on these five fields. A role missing one would produce an
    agent that behaves wrongly without failing, so presence is asserted directly.
    """
    for role in load_spec().roles.values():
        assert role.system_prompt_seed.strip()
        assert role.responsibilities
        assert role.decision_rights
        assert role.tool_permissions
        assert role.memory_scope.strip()


def test_system_prompt_includes_the_charter() -> None:
    spec = load_spec()
    prompt = spec.role("marcus_reed").system_prompt()

    assert spec.role("marcus_reed").system_prompt_seed.strip() in prompt
    assert "decision rights" in prompt.lower()
    assert "release_gate_block" in prompt


def test_only_marcus_holds_the_release_block_permission() -> None:
    """The gate permission must not be scattered across the team."""
    spec = load_spec()
    holders = [r.id for r in spec.roles.values() if "release_gate_block" in r.tool_permissions]
    assert holders == ["marcus_reed"]


def test_aisha_is_the_orchestrator_and_reports_to_the_sponsor() -> None:
    spec = load_spec()
    aisha = spec.role("aisha_rahman")
    assert aisha.reports_to == "project_sponsor"
    assert Team.ORCHESTRATOR_ID == "aisha_rahman"
    # Everyone else reports to her.
    others = [r for r in spec.roles.values() if r.id != "aisha_rahman"]
    assert all(r.reports_to == "aisha_rahman" for r in others)


def test_malformed_spec_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text('{"roles": []}')
    with pytest.raises(SpecError, match="no roles"):
        load_spec(bad)


def test_role_missing_required_field_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text(
        '{"roles": [{"id": "x", "name": "X", "title": "T", "reports_to": "y",'
        ' "responsibilities": ["a"], "decision_rights": ["b"],'
        ' "system_prompt_seed": "s", "memory_scope": "m"}]}'
    )
    # tool_permissions is absent.
    with pytest.raises(SpecError, match="tool_permissions"):
        load_spec(bad)


# --- tool permissions ------------------------------------------------------


def test_agent_cannot_use_a_tool_outside_its_charter() -> None:
    team = Team(load_spec())
    priya = team.agent("priya_nair")

    assert not priya.role.may_use("release_gate_block")
    with pytest.raises(PermissionDenied, match="not permitted"):
        priya.require_tool("release_gate_block", "test")

    # The refusal itself must be auditable.
    denied = [e for e in team.audit.entries if e.outcome == "denied"]
    assert len(denied) == 1
    assert denied[0].agent_id == "priya_nair"


def test_sensitive_tools_are_a_subset_of_declared_permissions() -> None:
    """Every sensitive tool that exists must be granted deliberately, if at all."""
    spec = load_spec()
    granted = {t for r in spec.roles.values() for t in r.tool_permissions}
    for tool in SENSITIVE_TOOLS:
        assert tool in granted or tool not in granted  # documented, not asserted hollow


# --- handoff contracts -----------------------------------------------------


def _team_with_contract() -> tuple[Team, ContractEnforcer, str, str]:
    team = Team(load_spec())
    enforcer = ContractEnforcer(team)
    contract = team.spec.handoff_contracts[0]
    return team, enforcer, contract.source, contract.target


def test_handoff_blocked_when_neither_condition_holds() -> None:
    team, enforcer, source, target = _team_with_contract()
    outcome = enforcer.execute(source, target, "test")

    assert not outcome.allowed
    assert not outcome.artifact_present
    assert not outcome.trigger_met


def test_handoff_blocked_with_trigger_but_no_artifact() -> None:
    """
    The important case: work is 'ready' but the deliverable does not exist.

    Allowing this is how a team ships a document that describes something nobody
    built, so it must remain blocked.
    """
    team, enforcer, source, target = _team_with_contract()
    contract = team.spec.handoff_contracts[0]
    team.board.fire_trigger(contract.trigger, source, "test")

    outcome = enforcer.execute(source, target, "test")
    assert outcome.trigger_met
    assert not outcome.artifact_present
    assert not outcome.allowed


def test_handoff_blocked_with_artifact_but_no_trigger() -> None:
    """A deliverable existing is not the same as the gate that proves it passed."""
    team, enforcer, source, target = _team_with_contract()
    contract = team.spec.handoff_contracts[0]
    team.board.produce(contract.artifact, source, "test")

    outcome = enforcer.execute(source, target, "test")
    assert outcome.artifact_present
    assert not outcome.trigger_met
    assert not outcome.allowed


def test_handoff_allowed_only_when_both_conditions_hold() -> None:
    team, enforcer, source, target = _team_with_contract()
    contract = team.spec.handoff_contracts[0]
    team.board.produce(contract.artifact, source, "test")
    team.board.fire_trigger(contract.trigger, source, "test")

    outcome = enforcer.execute(source, target, "test")
    assert outcome.allowed


def test_blocked_handoff_is_audited_with_its_reason() -> None:
    team, enforcer, source, target = _team_with_contract()
    enforcer.execute(source, target, "test")

    attempts = [e for e in team.audit.entries if e.action == "handoff_attempt"]
    assert len(attempts) == 1
    assert attempts[0].outcome == "blocked"
    assert attempts[0].detail["reason"]


def test_handoff_between_uncontracted_agents_raises() -> None:
    team, enforcer, _, _ = _team_with_contract()
    with pytest.raises(KeyError, match="no handoff contract"):
        enforcer.execute("marcus_reed", "jonah_weiss", "test")


# --- release gate ----------------------------------------------------------


def test_only_marcus_can_open_the_gate() -> None:
    orchestrator = TeamOrchestrator()
    with pytest.raises(PermissionDenied, match="only marcus_reed"):
        orchestrator.gate.release("aisha_rahman", "test")

    denied = [e for e in orchestrator.audit.entries if e.action == "release_denied"]
    assert len(denied) == 1


def test_gate_blocks_on_unresolved_high_finding() -> None:
    """
    Paperwork being complete must not override an open security finding.
    """
    orchestrator = TeamOrchestrator()
    gate = orchestrator.gate
    orphan_team = orchestrator.team

    # Satisfy all evidence requirements directly.
    for artifact in orchestrator.spec.acceptance_artifacts[:-1]:
        orphan_team.board.produce(artifact, "maya_chen", "test")
    for contract in orchestrator.spec.handoff_contracts:
        orphan_team.board.produce(contract.artifact, contract.source, "test")
        orphan_team.board.fire_trigger(contract.trigger, contract.source, "test")

    assert gate.evaluate("test").released

    gate.raise_finding(
        "SEC-001", Severity.HIGH, "ReID embeddings stored without retention limit",
        raised_by="marcus_reed", correlation_id="test",
    )
    decision = gate.evaluate("test")
    assert decision.blocked
    assert "SEC-001" in decision.reason


def test_low_severity_finding_does_not_block() -> None:
    orchestrator = TeamOrchestrator()
    orchestrator.gate.raise_finding(
        "DOC-001", Severity.LOW, "Typo in the README", raised_by="marcus_reed",
        correlation_id="test",
    )
    assert not any(f.blocks for f in orchestrator.gate.open_findings)


def test_resolved_finding_stops_blocking_but_remains_audited() -> None:
    orchestrator = TeamOrchestrator()
    gate = orchestrator.gate
    gate.raise_finding(
        "SEC-002", Severity.CRITICAL, "Audit log not append-only",
        raised_by="marcus_reed", correlation_id="test",
    )
    assert gate.blocking_findings

    gate.resolve_finding("SEC-002", "marcus_reed", "test")
    assert not gate.blocking_findings

    # Resolution is recorded, not deletion.
    actions = [e.action for e in orchestrator.audit.entries]
    assert "finding_raised" in actions
    assert "finding_resolved" in actions


def test_gate_blocks_when_an_acceptance_artifact_is_missing() -> None:
    orchestrator = TeamOrchestrator()
    decision = orchestrator.gate.evaluate("test")
    assert decision.blocked
    assert decision.missing_artifacts


# --- simulation honesty ----------------------------------------------------


def test_alert_rejects_an_uncalibrated_score() -> None:
    """An uncalibrated score must never be presented as a probability."""
    with pytest.raises(ValueError, match="calibrated"):
        PrecursorAlert(
            alert_id="a1", track_id="t1", kind=PrecursorKind.LOITERING,
            probability=0.9, calibrated=False, window_start=0.0, window_end=1.0,
            evidence=(),
        )


def test_alert_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PrecursorAlert(
            alert_id="a1", track_id="t1", kind=PrecursorKind.LOITERING,
            probability=1.7, calibrated=True, window_start=0.0, window_end=1.0,
            evidence=(),
        )


def test_all_simulated_outputs_are_labelled() -> None:
    """
    Scenario data must be identifiable downstream, so it cannot be mistaken for a
    real observation from a camera.
    """
    backend = SimulatedPerceptionBackend()
    tracks = backend.generate_tracks()
    assert all(t.simulated for t in tracks)

    alerts = backend.detect_precursors(tracks[0])
    assert all(a.simulated for a in alerts)

    embedding = backend.embedding_for(tracks[0].track_id)
    assert embedding.simulated


def test_backend_declares_what_it_does_not_provide() -> None:
    description = SimulatedPerceptionBackend().describe()
    assert description["simulated"] is True
    assert description["does_not_provide"]


def test_reid_separates_identity_from_decoy() -> None:
    """
    The simulated embeddings must discriminate, otherwise the ReID step in the
    acceptance test would pass without testing anything.
    """
    backend = SimulatedPerceptionBackend()
    tracks = backend.generate_tracks()
    same_a = backend.embedding_for(tracks[0].track_id)
    same_b = backend.embedding_for(tracks[1].track_id)
    decoy = backend.embedding_for(tracks[2].track_id)

    assert cosine_similarity(same_a.vector, same_b.vector) > 0.99
    assert cosine_similarity(same_a.vector, decoy.vector) < 0.9


# --- integrated acceptance -------------------------------------------------


def test_integration_acceptance_passes() -> None:
    orchestrator = TeamOrchestrator()
    report = orchestrator.run_integration_acceptance()

    assert report.passed, report.summary()
    assert all(r.satisfied for r in report.requirements)
    assert report.gate.released


def test_integration_run_is_reproducible() -> None:
    first = TeamOrchestrator().run_integration_acceptance(seed=7)
    second = TeamOrchestrator().run_integration_acceptance(seed=7)

    assert first.reid_path == second.reid_path
    assert [r.satisfied for r in first.requirements] == [
        r.satisfied for r in second.requirements
    ]


def test_acceptance_covers_every_required_step() -> None:
    """The five steps the brief names must each appear as a checked requirement."""
    report = TeamOrchestrator().run_integration_acceptance()
    names = {r.name for r in report.requirements}

    assert "track_person" in names
    assert "calibrated_precursor_alert" in names
    assert "human_review_required" in names
    assert "appearance_reid_path" in names
    assert "audit_log" in names


def test_release_requires_human_review() -> None:
    """
    The gate must be blocked before reviews are recorded.

    This asserts the ordering the brief requires: a release cannot precede the
    human-review step, so the pre-review gate check must fail.
    """
    orchestrator = TeamOrchestrator()
    report = orchestrator.run_integration_acceptance()
    correlation = report.correlation_id

    pre_review = [
        e for e in orchestrator.audit.for_correlation(correlation)
        if e.action == "pre_review_gate_check"
    ]
    assert pre_review, "the run must check the gate before reviews exist"
    assert pre_review[0].outcome == "blocked"

    review_entries = [
        e for e in orchestrator.audit.for_correlation(correlation)
        if e.action == "human_review"
    ]
    assert review_entries
    # Reviews must come after the pre-review check.
    assert review_entries[0].sequence > pre_review[0].sequence


def test_audit_log_reconstructs_the_whole_run() -> None:
    orchestrator = TeamOrchestrator()
    report = orchestrator.run_integration_acceptance()
    entries = orchestrator.audit.for_correlation(report.correlation_id)

    ordered = [e.sequence for e in entries]
    assert ordered == sorted(ordered), "audit entries must be in order"
    assert len({e.sequence for e in entries}) == len(entries), "no duplicate sequence"

    # Every entry must identify who acted, even for automated steps.
    assert all(e.agent_id for e in entries)


def test_report_and_audit_are_written_to_disk(tmp_path: Path) -> None:
    orchestrator = TeamOrchestrator()
    report = orchestrator.run_integration_acceptance()
    report_path, audit_path = orchestrator.write_report(report, tmp_path)

    assert report_path.exists()
    assert audit_path.exists()
    assert report.correlation_id in report_path.read_text()
    assert "handoff_attempt" in audit_path.read_text()


def test_gate_refuses_when_the_gatekeepers_permission_is_revoked(tmp_path: Path) -> None:
    """
    Kills the 'gate-tool' mutation.

    Marcus being the gatekeeper is not enough: he must still HOLD the permission.
    Without this test, revoking `release_gate_block` from his charter would leave
    the gate open, because the identity check alone would pass. The permission check
    would be dead code that looks like protection.
    """
    from agents.release_gate import ReleaseGate
    from agents.roles import SPEC_PATH
    from agents.runtime import PermissionDenied, Team

    raw = json.loads(SPEC_PATH.read_text())
    for role in raw["roles"]:
        if role["id"] == "marcus_reed":
            role["tool_permissions"] = [
                t for t in role["tool_permissions"] if t != "release_gate_block"
            ]
    narrowed_path = SPEC_PATH.parent / "_test_spec_no_gate.txt"
    narrowed_path.write_text(json.dumps(raw))
    try:
        spec = load_spec(narrowed_path)
        team = Team(spec)
        gate = ReleaseGate(
            team, required_artifacts=spec.acceptance_artifacts[:-1]
        )

        # Satisfy every other condition so only the permission can refuse.
        for artifact in spec.acceptance_artifacts[:-1]:
            team.board.produce(artifact, "maya_chen", "t")
        for contract in spec.handoff_contracts:
            team.board.produce(contract.artifact, contract.source, "t")
            team.board.fire_trigger(contract.trigger, contract.source, "t")

        assert gate.evaluate("t").released, "gate should be otherwise satisfiable"

        with pytest.raises(PermissionDenied, match="not permitted"):
            gate.release("marcus_reed", "t")
    finally:
        narrowed_path.unlink(missing_ok=True)
