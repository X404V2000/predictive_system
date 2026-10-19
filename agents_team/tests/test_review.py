"""
Tests for the performance review.

The review is easy to get subtly wrong in ways that look plausible, so the tests
target the specific traps:

  * a metric that rewards volume instead of quality
  * scoring agents on the system's own deliberate refusals
  * presenting an unmeasurable KPI as a score
  * reporting "no evidence" as a poor result
  * a sample of one being quoted as a percentage
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agents.kpi import Measurability, MetricName, assess_all, assess_kpi
from agents.registry import AgentRegistry
from agents.review import MIN_SAMPLE_FOR_RATE, build_scorecard, run_review
from agents.roles import load_spec


def _event(
    action: str,
    agent_id: str,
    outcome: str = "ok",
    **detail,
) -> dict:
    return {
        "action": action,
        "agent_id": agent_id,
        "outcome": outcome,
        "detail": detail,
        "correlation_id": "test",
        "sequence": 0,
    }


def _handoff(source: str, target: str, outcome: str) -> dict:
    return _event(
        "handoff_attempt", source, outcome, target=target, artifact="a", trigger="t"
    )


@pytest.fixture
def spec():
    return load_spec()


# --- KPI classification ----------------------------------------------------


def test_model_dependent_kpis_are_not_claimed_as_measurable(spec) -> None:
    """
    A KPI about tracking accuracy cannot be scored without a tracker. Reporting it
    as a score would be fabrication dressed as measurement.
    """
    assessments = assess_all(spec.role("maya_chen").kpis)
    by_text = {a.kpi: a for a in assessments}

    idf1 = by_text["Tracking IDF1 and MOTA on validation clips"]
    assert idf1.measurability is Measurability.NOT_MEASURABLE
    assert "tracker" in idf1.reason

    reid = by_text["ReID top-1 retrieval accuracy"]
    assert reid.measurability is Measurability.NOT_MEASURABLE


def test_every_kpi_gets_a_reason() -> None:
    """No KPI may be dismissed or scored without an explanation."""
    spec = load_spec()
    for role in spec.roles.values():
        for assessment in assess_all(role.kpis):
            assert assessment.reason, f"{role.id}: {assessment.kpi} has no reason"


def test_unknown_kpi_defaults_to_not_measurable() -> None:
    assessment = assess_kpi("Some KPI invented later")
    assert assessment.measurability is Measurability.NOT_MEASURABLE
    assert assessment.reason


def test_partially_measurable_kpi_names_the_part_that_is_measured() -> None:
    assessment = assess_kpi("Open cross-team blockers")
    assert assessment.measurability is Measurability.MEASURABLE
    assert assessment.metric is MetricName.HANDOFF_COMPLIANCE
    assert "not the team's full blocker list" in assessment.reason


# --- metric correctness ----------------------------------------------------


def test_deliberate_gate_refusals_are_not_scored_against_the_agent(spec) -> None:
    """
    The gate checks contracts before work exists, on purpose. Those refusals are
    the system working. Counting them would rank the agent with the most contracts
    as the worst performer, which is backwards.
    """
    events = [
        # Deliberate pre-review refusals.
        _handoff("maya_chen", "rafael_ortiz", "blocked"),
        _handoff("maya_chen", "priya_nair", "blocked"),
        # The settled outcomes.
        _handoff("maya_chen", "rafael_ortiz", "allowed"),
        _handoff("maya_chen", "priya_nair", "allowed"),
    ]
    card = build_scorecard("maya_chen", spec, None, events, set())
    metric = next(m for m in card.metrics if m.name is MetricName.HANDOFF_COMPLIANCE)

    assert metric.numerator == 2
    assert metric.denominator == 2, "only settled outcomes should be counted"


def test_handoff_fails_when_the_settled_outcome_is_blocked(spec) -> None:
    """Genuine failure must still be caught."""
    events = [
        _handoff("maya_chen", "rafael_ortiz", "allowed"),
        _handoff("maya_chen", "priya_nair", "blocked"),
    ]
    card = build_scorecard("maya_chen", spec, None, events, set())
    metric = next(m for m in card.metrics if m.name is MetricName.HANDOFF_COMPLIANCE)

    assert metric.numerator == 1
    assert metric.denominator == 2


def test_artifact_completeness_counts_contracted_items_only(spec) -> None:
    """Producing extra documents must not raise the score."""
    contracts = spec.contracts_from("maya_chen")
    required = {c.artifact for c in contracts}

    # Produces exactly what it contracted.
    exact = build_scorecard("maya_chen", spec, None, [], required)
    exact_metric = next(
        m for m in exact.metrics if m.name is MetricName.ARTIFACT_COMPLETENESS
    )
    assert exact_metric.value == 1.0
    assert exact_metric.denominator == len(required)

    # Produces a great many unrelated documents: still the same score.
    bloated = build_scorecard(
        "maya_chen", spec, None, [], required | {f"extra-{i}" for i in range(50)}
    )
    bloated_metric = next(
        m for m in bloated.metrics if m.name is MetricName.ARTIFACT_COMPLETENESS
    )
    assert bloated_metric.value == exact_metric.value
    assert bloated_metric.denominator == exact_metric.denominator


def test_volume_does_not_raise_the_overall_score(spec) -> None:
    """
    An agent that produces many events must not outscore one that produces few,
    when both do their contracted work correctly.
    """
    quiet = [_handoff("maya_chen", "rafael_ortiz", "allowed")] * 3
    busy = quiet + [_event("artifact_claimed", "maya_chen", artifact="x")] * 200

    quiet_card = build_scorecard("maya_chen", spec, None, quiet, set())
    busy_card = build_scorecard("maya_chen", spec, None, busy, set())

    assert busy_card.audit_events > quiet_card.audit_events
    assert busy_card.overall == quiet_card.overall


def test_charter_violation_lowers_adherence(spec) -> None:
    events = [
        _event("permission_denied", "priya_nair", "denied", tool="release_gate_block"),
        _event("decision", "priya_nair"),
        _event("decision", "priya_nair"),
    ]
    card = build_scorecard("priya_nair", spec, None, events, set())
    metric = next(m for m in card.metrics if m.name is MetricName.CHARTER_ADHERENCE)

    assert metric.numerator == 2
    assert metric.denominator == 3
    assert metric.value == pytest.approx(2 / 3)


def test_non_gatekeeper_cannot_earn_gate_discipline(spec) -> None:
    """A release attempt by someone without the permission must not score well."""
    events = [_event("release_decision", "priya_nair", "released")]
    card = build_scorecard("priya_nair", spec, None, events, set())
    metric = next(m for m in card.metrics if m.name is MetricName.GATE_DISCIPLINE)
    assert metric.value == 0.0


# --- sample size and honesty -----------------------------------------------


def test_small_sample_is_not_reported_as_a_percentage(spec) -> None:
    """
    One successful handoff is not 100% compliance, it is one observation.
    """
    events = [_handoff("maya_chen", "rafael_ortiz", "allowed")]
    card = build_scorecard("maya_chen", spec, None, events, set())
    metric = next(m for m in card.metrics if m.name is MetricName.HANDOFF_COMPLIANCE)

    assert metric.denominator < MIN_SAMPLE_FOR_RATE
    assert not metric.has_evidence
    assert "insufficient evidence" in metric.formatted()


def test_no_evidence_gives_no_score_rather_than_zero(spec) -> None:
    """
    An agent with nothing measured has not performed badly; it has not been
    observed. Returning 0.0 would read as an accusation.
    """
    card = build_scorecard("aisha_rahman", spec, None, [], set())
    assert card.overall is None
    assert card.verdict() == "insufficient evidence"


def test_unmeasured_kpis_appear_in_the_report(spec) -> None:
    """Gaps must be visible, not silently omitted."""
    card = build_scorecard("marcus_reed", spec, None, [], set())
    assert card.gaps
    assert all(g.measurability is Measurability.NOT_MEASURABLE for g in card.gaps)


# --- review report ---------------------------------------------------------


def test_review_covers_every_agent_including_silent_ones(spec) -> None:
    report = run_review(spec, None, [])
    assert {c.agent_id for c in report.scorecards} == set(spec.roles)


def test_review_states_its_own_sample_size(spec) -> None:
    """
    One run is a point-in-time check, not a trend. The report must say so rather
    than let a reader assume otherwise.
    """
    events = [_handoff("maya_chen", "rafael_ortiz", "allowed")]
    report = run_review(spec, None, events)
    joined = " ".join(report.team_notes).lower()

    assert "single run" in joined
    assert "not a performance trend" in joined


def test_unregistered_agents_are_flagged(spec, tmp_path: Path) -> None:
    registry = AgentRegistry(tmp_path / "store")
    registry.register(spec.roles["maya_chen"])

    report = run_review(spec, registry, [])
    unregistered = [c.agent_id for c in report.scorecards if c.status == "unregistered"]
    assert len(unregistered) == len(spec.roles) - 1
    assert any("not registered" in note.lower() for note in report.team_notes)


def test_report_serialises_to_json(spec) -> None:
    import json

    report = run_review(spec, None, [])
    payload = json.dumps(report.to_dict())
    restored = json.loads(payload)

    assert restored["totals"]["unmeasured_kpis"] > 0
    assert len(restored["agents"]) == len(spec.roles)
    for agent in restored["agents"]:
        assert "verdict" in agent
        assert isinstance(agent["kpis"], list)


def test_review_never_returns_a_verdict_from_volume_alone(spec) -> None:
    """
    A run full of activity with no contracted work completed must not pass.
    """
    events = [_event("artifact_claimed", "maya_chen", artifact="unrelated")] * 100
    card = build_scorecard("maya_chen", spec, None, events, {"unrelated"})
    # Artifacts produced: none of the contracted ones.
    metric = next(m for m in card.metrics if m.name is MetricName.ARTIFACT_COMPLETENESS)
    assert metric.value == 0.0
