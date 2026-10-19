"""
KPI Classification.

Every agent in the specification has KPIs. Almost none of them can be measured from
this system, because they describe the quality of a MODEL that does not exist yet:
tracking IDF1, ReID top-1 accuracy, calibration error, p95 edge latency.

Reporting those as scores would produce a review that looks authoritative and means
nothing, so each is classified explicitly as measurable or not, with the reason.
A reviewer can then see at a glance which parts of the review are evidence and
which are simply unobserved.

MEASURABLE KPIs are process properties the audit trail can actually answer:

    handoff_compliance      did the agent's contracted handoffs succeed?
    artifact_completeness   did it produce what it was contracted to produce?
    charter_adherence       did it stay inside its tool permissions?
    gate_discipline         did it attempt privileged actions it may not perform?
    review_participation    did it record the decisions it holds rights for?

NOT MEASURABLE KPIs need a trained model, a real deployment, or real-world
incidents. The reason is recorded per KPI, not summarised, so a reader can judge
whether the gap is closable soon or structural.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Measurability(str, Enum):
    MEASURABLE = "measurable"
    NOT_MEASURABLE = "not_measurable"


class MetricName(str, Enum):
    """Process metrics computed from the audit trail."""

    HANDOFF_COMPLIANCE = "handoff_compliance"
    ARTIFACT_COMPLETENESS = "artifact_completeness"
    CHARTER_ADHERENCE = "charter_adherence"
    GATE_DISCIPLINE = "gate_discipline"
    REVIEW_PARTICIPATION = "review_participation"


# Why each unmeasurable KPI cannot be scored here. Keyed by the KPI text from the
# specification; a KPI with no entry falls back to the generic reason below.
KPI_BLOCKERS: dict[str, str] = {
    "Tracking IDF1 and MOTA on validation clips": "no multi-object tracker exists in the codebase",
    "ReID top-1 retrieval accuracy": "no ReID embedding model exists in the codebase",
    "End-to-end perception latency within budget": "no production perception pipeline exists",
    "False-alert contribution from perception errors": "requires a real perception stack and alert outcomes",
    "p95 inference latency on target edge device": "no edge deployment exists",
    "FPS per stream at required resolution": "no edge deployment exists",
    "Memory headroom and thermal stability": "requires physical edge hardware",
    "Deployment success rate": "no deployment pipeline has been exercised",
    "Recall at fixed false-positive rate": "no trained behaviour model exists",
    "Calibration error (ECE)": "no trained behaviour model exists to measure calibration on",
    "Alert precision after human review": "reviews here are simulated, not real operator decisions",
    "Drift detection response time": "no model in production to drift",
    "API p95 latency": "the platform API described in the spec is not built",
    "Dashboard uptime": "no operator dashboard exists",
    "Forensic search response time": "no forensic search service exists",
    "Operator review completion rate": "reviews here are simulated",
    "Data quality pass rate": "no production data pipeline exists",
    "Model deployment frequency and rollback time": "no model has been deployed",
    "Drift alert response time": "no production model to monitor",
    "Reproducibility audit pass rate": "no reproducibility audit process exists",
    "Open high-risk findings before release": "measurable in principle; no findings have been raised in a real review yet",
    "Audit log coverage": "partially measurable; this review reports audit events per agent",
    "Privacy incident count": "requires real incidents, not a simulation",
    "Compliance gate pass rate": "no real compliance review has been performed",
    "Milestone predictability": "no milestone history exists",
    "Open cross-team blockers": "partially measured as blocked handoffs in this review",
    "Release gate pass rate": "partially measurable; only one acceptance run has been made",
    "Stakeholder decision latency": "no stakeholder decisions recorded",
}

DEFAULT_BLOCKER = "requires evidence this system does not produce"


@dataclass(frozen=True)
class KpiAssessment:
    """Classification of one KPI from the specification."""

    kpi: str
    measurability: Measurability
    metric: MetricName | None = None
    reason: str = ""

    @property
    def scored(self) -> bool:
        return self.measurability is Measurability.MEASURABLE


# KPIs that map onto a process metric this review actually computes. Matched by the
# exact KPI text in new_team.md, because "partially measurable" must be narrowed to
# the part that really is measured rather than assumed.
PARTIAL_KPI_METRICS: dict[str, tuple[MetricName, str]] = {
    "Audit log coverage": (
        MetricName.REVIEW_PARTICIPATION,
        "measured as the presence of this agent in the audit trail, not as coverage "
        "of every system event",
    ),
    "Open cross-team blockers": (
        MetricName.HANDOFF_COMPLIANCE,
        "measured as the agent's blocked handoff attempts, not the team's full blocker list",
    ),
    "Release gate pass rate": (
        MetricName.GATE_DISCIPLINE,
        "measured on gate actions this agent attempted, not across all releases",
    ),
}

# Every agent is expected to produce its contracted artifacts and keep its handoffs
# clean, so these are assessed for every role regardless of the KPI wording.
PROCESS_BASELINE: tuple[tuple[MetricName, str], ...] = (
    (MetricName.HANDOFF_COMPLIANCE, "contracted handoffs should succeed"),
    (MetricName.ARTIFACT_COMPLETENESS, "contracted artifacts should be produced"),
    (MetricName.CHARTER_ADHERENCE, "the agent should stay inside its tool permissions"),
    (MetricName.GATE_DISCIPLINE, "privileged actions should only be attempted by those entitled to them"),
)


def assess_kpi(kpi: str) -> KpiAssessment:
    """Classifies one KPI from the specification."""
    if kpi in PARTIAL_KPI_METRICS:
        metric, reason = PARTIAL_KPI_METRICS[kpi]
        return KpiAssessment(kpi, Measurability.MEASURABLE, metric, reason)

    blocker = KPI_BLOCKERS.get(kpi)
    if blocker is None:
        return KpiAssessment(
            kpi, Measurability.NOT_MEASURABLE, None, DEFAULT_BLOCKER
        )
    return KpiAssessment(kpi, Measurability.NOT_MEASURABLE, None, blocker)


def assess_all(kpis: tuple[str, ...]) -> tuple[KpiAssessment, ...]:
    return tuple(assess_kpi(k) for k in kpis)


__all__ = [
    "DEFAULT_BLOCKER",
    "KPI_BLOCKERS",
    "KpiAssessment",
    "Measurability",
    "MetricName",
    "PARTIAL_KPI_METRICS",
    "PROCESS_BASELINE",
    "assess_all",
    "assess_kpi",
]
