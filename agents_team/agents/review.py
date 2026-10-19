"""
Agent Performance Review.

Produces a per-agent scorecard from the evidence the system actually holds: the
acceptance-run audit trail, the registry roster and the team specification.

Three deliberate design decisions:

  * **Quality over volume.** No metric rewards how much an agent did. An agent that
    produces ten documents scores the same as one that produces two, provided both
    deliver what they contracted. Counting events would put whoever happens to log
    most at the top of the table, which says nothing about their work.

  * **No invented scores.** A KPI that needs a real model, deployment or incident
    history is reported as not measurable, with the reason. A scorecard that omits
    what it cannot see is more useful than one that fills every cell.

  * **Advisory only.** This module reads evidence and writes a report. It cannot
    suspend, retire or re-charter anyone; those remain explicit registry actions
    that a human takes.

The review is honest about its own sample size: a single acceptance run gives very
little evidence, and the report says so rather than implying a trend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agents.kpi import (
    PROCESS_BASELINE,
    KpiAssessment,
    Measurability,
    MetricName,
    assess_all,
)
from agents.registry import AgentRegistry, RegisteredAgent
from agents.roles import TeamSpec


# Minimum number of applicable handoffs before a compliance rate is considered
# meaningful. Below this the metric is reported as insufficient evidence rather
# than as a percentage of one.
MIN_SAMPLE_FOR_RATE: int = 3


@dataclass(frozen=True)
class MetricResult:
    """One measured process metric."""

    name: MetricName
    value: float | None
    numerator: int
    denominator: int
    description: str

    @property
    def has_evidence(self) -> bool:
        return self.denominator >= MIN_SAMPLE_FOR_RATE

    def formatted(self) -> str:
        if self.denominator == 0:
            return "no evidence"
        if not self.has_evidence:
            return f"insufficient evidence ({self.numerator}/{self.denominator})"
        assert self.value is not None
        return f"{self.value:.0%} ({self.numerator}/{self.denominator})"


@dataclass
class AgentScorecard:
    """Evidence and assessment for one agent."""

    agent_id: str
    name: str
    handle: str
    status: str
    version: int
    metrics: tuple[MetricResult, ...]
    kpis: tuple[KpiAssessment, ...]
    audit_events: int

    @property
    def measurable_metrics(self) -> tuple[MetricResult, ...]:
        return tuple(m for m in self.metrics if m.has_evidence)

    @property
    def gaps(self) -> tuple[KpiAssessment, ...]:
        return tuple(k for k in self.kpis if not k.scored)

    @property
    def overall(self) -> float | None:
        """
        Mean of the metrics that have enough evidence to mean anything.

        Returns None when nothing is measurable, rather than 0.0: an agent with no
        evidence has not performed badly, it has simply not been observed, and a
        zero would be a false accusation.
        """
        usable = [m.value for m in self.measurable_metrics if m.value is not None]
        if not usable:
            return None
        return sum(usable) / len(usable)

    def verdict(self) -> str:
        """
        A tempered summary.

        Thresholds are deliberately blunt. With a handful of audit events a finer
        scale would imply precision the evidence does not support.
        """
        score = self.overall
        if score is None:
            return "insufficient evidence"
        if score >= 0.95:
            return "meets expectations"
        if score >= 0.80:
            return "minor follow-up"
        return "review required"


@dataclass
class ReviewReport:
    """The full performance review."""

    generated_at: str
    source_audit: str
    scorecards: tuple[AgentScorecard, ...]
    team_notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_measured_kpis(self) -> int:
        return sum(len([k for k in c.kpis if k.scored]) for c in self.scorecards)

    @property
    def total_unmeasured_kpis(self) -> int:
        return sum(len(c.gaps) for c in self.scorecards)

    def summary(self) -> str:
        lines = [
            "AGENT PERFORMANCE REVIEW",
            f"generated: {self.generated_at}",
            f"evidence : {self.source_audit}",
            "",
            f"{'agent':<16}{'ver':<5}{'status':<11}{'verdict':<22}{'score':<8}events",
        ]
        lines.append("-" * 78)
        for card in sorted(self.scorecards, key=lambda c: c.agent_id):
            score = "n/a" if card.overall is None else f"{card.overall:.0%}"
            lines.append(
                f"{card.agent_id:<16}v{card.version:<4}{card.status:<11}"
                f"{card.verdict():<22}{score:<8}{card.audit_events}"
            )

        lines.append("")
        lines.append("Process metrics (only those with enough evidence):")
        for card in sorted(self.scorecards, key=lambda c: c.agent_id):
            usable = card.measurable_metrics
            if not usable:
                lines.append(f"  {card.agent_id}: none")
                continue
            lines.append(f"  {card.agent_id}:")
            for metric in usable:
                lines.append(f"    {metric.name.value:<24} {metric.formatted()}")

        lines.append("")
        lines.append(
            f"KPIs: {self.total_measured_kpis} measurable, "
            f"{self.total_unmeasured_kpis} not measurable"
        )
        for card in sorted(self.scorecards, key=lambda c: c.agent_id):
            if card.gaps:
                lines.append(f"  {card.agent_id}:")
                for gap in card.gaps:
                    lines.append(f"    - {gap.kpi}")
                    lines.append(f"        {gap.reason}")

        if self.team_notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.team_notes)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "source_audit": self.source_audit,
            "totals": {
                "measured_kpis": self.total_measured_kpis,
                "unmeasured_kpis": self.total_unmeasured_kpis,
            },
            "agents": [
                {
                    "agent_id": c.agent_id,
                    "name": c.name,
                    "handle": c.handle,
                    "status": c.status,
                    "version": c.version,
                    "verdict": c.verdict(),
                    "overall": c.overall,
                    "audit_events": c.audit_events,
                    "metrics": [
                        {
                            "name": m.name.value,
                            "value": m.value,
                            "numerator": m.numerator,
                            "denominator": m.denominator,
                            "has_evidence": m.has_evidence,
                            "description": m.description,
                        }
                        for m in c.metrics
                    ],
                    "kpis": [
                        {
                            "kpi": k.kpi,
                            "measurability": k.measurability.value,
                            "metric": k.metric.value if k.metric else None,
                            "reason": k.reason,
                        }
                        for k in c.kpis
                    ],
                }
                for c in self.scorecards
            ],
            "team_notes": list(self.team_notes),
        }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def build_scorecard(
    agent_id: str,
    spec: TeamSpec,
    registered: RegisteredAgent | None,
    audit_events: list[dict[str, Any]],
    produced_artifacts: set[str],
) -> AgentScorecard:
    """
    Builds one agent's scorecard from the audit trail.

    All metrics are ratios of "did the right thing happen", never counts of how
    much happened.
    """
    role = spec.role(agent_id)
    mine = [e for e in audit_events if e.get("agent_id") == agent_id]

    # Handoffs this agent is contracted to make.
    #
    # A blocked attempt is only counted against the agent when the blockage was
    # genuinely its own. The gate deliberately evaluates contracts BEFORE the
    # required work exists, to prove it refuses; those refusals are the system
    # working, not the agent failing. Scoring them would have marked the agent with
    # the most contracts as the worst performer, which is the exact opposite of the
    # truth.
    contracts = spec.contracts_from(agent_id)
    attempts = [e for e in mine if e.get("action") == "handoff_attempt"]
    final_by_target: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        target = attempt.get("detail", {}).get("target")
        if target:
            # Later attempts supersede earlier ones for the same target.
            final_by_target[target] = attempt

    # Only the settled outcome per contract is scored.
    assessed = list(final_by_target.values())
    allowed = sum(1 for e in assessed if e.get("outcome") == "allowed")
    handoff_metric = MetricResult(
        name=MetricName.HANDOFF_COMPLIANCE,
        value=_rate(allowed, len(assessed)),
        numerator=allowed,
        denominator=len(assessed),
        description=(
            "contracted handoffs whose final outcome was allowed "
            "(earlier refusals from deliberate gate checks are not counted)"
        ),
    )

    # Artifacts this agent is contracted to produce.
    required = {c.artifact for c in contracts}
    delivered = sum(1 for a in required if a in produced_artifacts)
    artifact_metric = MetricResult(
        name=MetricName.ARTIFACT_COMPLETENESS,
        value=_rate(delivered, len(required)),
        numerator=delivered,
        denominator=len(required),
        description="contracted artifacts produced",
    )

    # Charter adherence: a denial means the agent tried something outside its
    # charter. Measured only once the agent has actually acted.
    denials = sum(1 for e in mine if e.get("outcome") == "denied")
    actions = [e for e in mine if e.get("action") not in {"artifact_produced", "artifact_claimed"}]
    adherence_metric = MetricResult(
        name=MetricName.CHARTER_ADHERENCE,
        value=_rate(len(actions) - denials, len(actions)) if actions else None,
        numerator=max(len(actions) - denials, 0),
        denominator=len(actions),
        description="action attempts that stayed inside the agent's tool permissions",
    )

    # Gate discipline: only the gatekeeper may act on the release gate.
    gate_actions = [e for e in mine if e.get("action", "").startswith("release")]
    entitled = 1 if "release_gate_block" in role.tool_permissions else 0
    if gate_actions:
        correct = sum(
            1 for e in gate_actions
            if e.get("outcome") in {"released", "blocked"} and entitled
        )
        gate_value = _rate(correct, len(gate_actions))
    else:
        gate_value = None
    gate_metric = MetricResult(
        name=MetricName.GATE_DISCIPLINE,
        value=gate_value,
        numerator=correct if gate_actions else 0,
        denominator=len(gate_actions),
        description="release-gate actions taken only by an agent entitled to take them",
    )

    # Review participation: did the agent record decisions it holds rights for?
    decisions = [e for e in mine if e.get("action") in {"decision", "release_decision"}]
    review_metric = MetricResult(
        name=MetricName.REVIEW_PARTICIPATION,
        value=_rate(len(decisions), len(decisions)) if decisions else None,
        numerator=len(decisions),
        denominator=len(decisions),
        description="recorded decisions; presence in the trail, not volume",
    )

    metrics = (
        handoff_metric,
        artifact_metric,
        adherence_metric,
        gate_metric,
        review_metric,
    )

    record = registered
    return AgentScorecard(
        agent_id=agent_id,
        name=role.name,
        handle=role.agent_handle,
        status=record.status if record else "unregistered",
        version=record.version if record else 0,
        metrics=metrics,
        kpis=assess_all(role.kpis),
        audit_events=len(mine),
    )


def run_review(
    spec: TeamSpec,
    registry: AgentRegistry | None,
    audit_events: list[dict[str, Any]],
    source: str = "acceptance run",
) -> ReviewReport:
    """
    Produces the review for every agent in the specification.

    Agents that appear in the specification but not the audit trail still receive a
    scorecard, marked as having no evidence, so silence is visible rather than
    absent from the report.
    """
    produced = {
        e.get("detail", {}).get("artifact")
        for e in audit_events
        if e.get("action") == "artifact_claimed"
    }
    produced.discard(None)

    cards = []
    for agent_id in spec.roles:
        record = registry.get(agent_id) if registry else None
        cards.append(
            build_scorecard(agent_id, spec, record, audit_events, produced)
        )

    notes = _team_notes(cards, audit_events)

    from datetime import datetime, timezone

    return ReviewReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_audit=source,
        scorecards=tuple(cards),
        team_notes=notes,
    )


def _team_notes(
    cards: Iterable[AgentScorecard], audit_events: list[dict[str, Any]]
) -> tuple[str, ...]:
    """
    Observations that apply to the review as a whole.

    Kept short and factual. The most important is the sample size: one acceptance
    run is not a trend, and a review that implied otherwise would be misleading.
    """
    notes: list[str] = []

    handoff_attempts = sum(
        1 for e in audit_events if e.get("action") == "handoff_attempt"
    )
    notes.append(
        f"Evidence is drawn from a single run with {handoff_attempts} handoff "
        "attempts. This is a point-in-time check, not a performance trend."
    )

    unregistered = [c.agent_id for c in cards if c.status == "unregistered"]
    if unregistered:
        notes.append(
            f"Not registered: {', '.join(sorted(unregistered))}. Their status and "
            "charter version cannot be confirmed."
        )

    no_evidence = [c.agent_id for c in cards if c.overall is None]
    if no_evidence:
        notes.append(
            f"No measurable process evidence for: {', '.join(sorted(no_evidence))}. "
            "This is an absence of observation, not a poor result."
        )

    notes.append(
        "Model-quality KPIs are not scored. They require a tracker, behaviour "
        "model or deployment that does not exist yet; reporting them would mean "
        "inventing numbers."
    )
    return tuple(notes)


def write_review(report: ReviewReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "performance_review.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


__all__ = [
    "AgentScorecard",
    "MIN_SAMPLE_FOR_RATE",
    "MetricResult",
    "ReviewReport",
    "build_scorecard",
    "run_review",
    "write_review",
]
