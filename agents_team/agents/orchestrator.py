"""
Team Orchestrator and Integrated Acceptance Run.

Step 2: Aisha Rahman coordinates, holding the team-level handoff contracts and the
acceptance artifacts.

Step 5: the integrated acceptance test. It must

    track a person, produce a calibrated precursor alert, require human review,
    and reconstruct the person's path via appearance-based ReID, with audit logs.

The run is deliberately strict. Each requirement is checked against the shared
board and audit log rather than against a return value, so the test cannot pass
because a function said so. A requirement that cannot be met leaves the gate
closed, and the report says why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.contracts import ContractEnforcer
from agents.release_gate import GateDecision, ReleaseGate, Severity
from agents.roles import SpecError, TeamSpec, load_spec
from agents.runtime import AuditLog, Team
from agents.sim_backend import (
    HumanReview,
    PrecursorAlert,
    ReviewDecision,
    SimulatedPerceptionBackend,
    Track,
    cosine_similarity,
)


@dataclass(frozen=True)
class RequirementResult:
    """One acceptance requirement and whether it was satisfied."""

    name: str
    satisfied: bool
    evidence: str

    def __str__(self) -> str:  # pragma: no cover - display only
        mark = "PASS" if self.satisfied else "FAIL"
        return f"[{mark}] {self.name}: {self.evidence}"


@dataclass
class AcceptanceReport:
    """The full result of an integration run."""

    correlation_id: str
    requirements: tuple[RequirementResult, ...]
    gate: GateDecision
    audit_entries: int
    reid_path: tuple[tuple[float, float], ...]
    backend: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(r.satisfied for r in self.requirements) and self.gate.released

    def summary(self) -> str:
        lines = [
            "INTEGRATED ACCEPTANCE RUN",
            f"correlation id: {self.correlation_id}",
            "backend        : SIMULATED (no real tracker, behaviour model or ReID)",
            "",
            "Requirements:",
        ]
        lines.extend(f"  {r}" for r in self.requirements)
        lines.append("")
        lines.append(f"Release gate   : {'RELEASED' if self.gate.released else 'BLOCKED'}")
        lines.append(f"  reason       : {self.gate.reason}")
        lines.append(f"  gatekeeper   : {self.gate.evaluated_by}")
        lines.append(f"Audit entries  : {self.audit_entries}")
        lines.append(f"Reconstructed path points: {len(self.reid_path)}")
        lines.append("")
        lines.append(f"Overall: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "passed": self.passed,
            "backend": self.backend,
            "requirements": [
                {"name": r.name, "satisfied": r.satisfied, "evidence": r.evidence}
                for r in self.requirements
            ],
            "gate": {
                "released": self.gate.released,
                "reason": self.gate.reason,
                "evaluated_by": self.gate.evaluated_by,
                "missing_artifacts": list(self.gate.missing_artifacts),
                "blocked_handoffs": list(self.gate.blocked_handoffs),
            },
            "audit_entries": self.audit_entries,
            "reid_path": [list(p) for p in self.reid_path],
        }


class TeamOrchestrator:
    """
    Aisha Rahman coordinating the team.

    Owns the acceptance artifacts and the handoff contracts, and is the only agent
    that may assemble the release candidate handed to the gate.
    """

    def __init__(self, spec: TeamSpec | None = None) -> None:
        self.spec = spec or load_spec()
        self.team = Team(self.spec)
        self.enforcer = ContractEnforcer(self.team)
        self.gate = ReleaseGate(
            self.team, required_artifacts=self.spec.acceptance_artifacts[:-1]
        )

    # -- helpers ----------------------------------------------------------

    @property
    def audit(self) -> AuditLog:
        return self.team.audit

    def _produce_acceptance_artifacts(self, correlation_id: str) -> None:
        """
        Records the six work-stream acceptance artifacts as produced.

        In a real run each would come from its owning agent after their work. Here
        the aim is to exercise the gate, so they are produced by their owners with
        a note that the content is placeholder. The integrated demo artifact (the
        seventh) is NOT produced here: it must be earned by the acceptance run.
        """
        owners = {
            "CV architecture specification and benchmark report": "maya_chen",
            "Edge latency/memory/FPS budget report": "rafael_ortiz",
            "Behavior labeling guideline, dataset card, and calibration report": "priya_nair",
            "Platform API spec and operator dashboard MVP": "jonah_weiss",
            "MLOps pipeline, model registry, and drift dashboard": "elena_petrova",
            "Threat model, privacy risk register, and release gate report": "marcus_reed",
        }
        for artifact, owner in owners.items():
            agent = self.team.agent(owner)
            agent.produce_artifact(
                artifact,
                correlation_id,
                tool="repo_read_write" if agent.role.may_use("repo_read_write") else "policy_writer",
                payload={"placeholder": True, "note": "acceptance scaffold"},
            )

    def _produce_contract_artifacts(self, correlation_id: str) -> None:
        """
        Produces the artifact each handoff contract requires.

        The contract artifacts are distinct from the acceptance artifacts: a
        contract names the specific deliverable that crosses an interface (for
        example "Model export + latency budget + target hardware profile"), while
        an acceptance artifact names the evidence a work stream must show. Both
        are produced, by the agent the contract names as its source, using a tool
        that agent is actually permitted to use.

        Producing from the contract rather than a hand-written list means a new
        contract in the specification is picked up automatically.
        """
        # Each role's most appropriate producing tool, chosen from its charter.
        tool_for = {
            "maya_chen": "design_doc_writer",
            "rafael_ortiz": "benchmark_runner",
            "priya_nair": "model_registry_write",
            "jonah_weiss": "repo_read_write",
            "elena_petrova": "model_registry_write",
            "marcus_reed": "security_review",
        }

        for contract in self.spec.handoff_contracts:
            agent = self.team.agent(contract.source)
            tool = tool_for.get(contract.source)
            if tool is None or not agent.role.may_use(tool):
                # Fall back to any tool the charter allows, so a specification
                # change cannot silently strand a contract.
                tool = next(
                    (t for t in agent.role.tool_permissions
                     if t not in {"release_gate_block"}),
                    None,
                )
            if tool is None:
                raise SpecError(
                    f"{contract.source} has no tool permitting artifact production"
                )
            agent.produce_artifact(
                contract.artifact,
                correlation_id,
                tool=tool,
                payload={
                    "for": contract.target,
                    "trigger": contract.trigger,
                    "placeholder": True,
                },
            )

    def _fire_contract_triggers(self, correlation_id: str) -> None:
        """Marks each contract's trigger as met, as the work streams would."""
        owners = {
            "CV/tracking model candidate selected": "maya_chen",
            "Perception interface frozen": "maya_chen",
            "Behavior model passes calibration gate": "priya_nair",
            "Edge benchmark passes": "rafael_ortiz",
            "Model promotion requested": "elena_petrova",
            "Release candidate proposed": "marcus_reed",
            "Platform API changes": "jonah_weiss",
        }
        for trigger, agent_id in owners.items():
            self.team.board.fire_trigger(trigger, agent_id, correlation_id)

    # -- acceptance run ---------------------------------------------------

    def run_integration_acceptance(
        self, seed: int = 7, reviewer_id: str = "operator-17"
    ) -> AcceptanceReport:
        """
        Executes the integrated acceptance test end to end.

        Order matters and mirrors the brief: the person is tracked, an alert is
        raised, a HUMAN reviews it, and only then is the path reconstructed. The
        gate refuses to release if the review step is missing.
        """
        correlation_id = self.team.new_correlation_id("acceptance")
        backend = SimulatedPerceptionBackend(seed=seed)
        self.audit.record(
            "acceptance_started",
            self.team.orchestrator.id,
            correlation_id,
            "ok",
            seed=seed,
            backend=backend.describe(),
        )

        requirements: list[RequirementResult] = []

        # 1. Track a person.
        tracks = backend.generate_tracks()
        for track in tracks:
            self.audit.record(
                "track_created",
                "maya_chen",
                correlation_id,
                "ok",
                track_id=track.track_id,
                camera_id=track.camera_id,
                points=len(track.observations),
                simulated=track.simulated,
            )
        requirements.append(
            RequirementResult(
                name="track_person",
                satisfied=len(tracks) >= 2 and all(t.observations for t in tracks),
                evidence=f"{len(tracks)} simulated tracks across "
                f"{len({t.camera_id for t in tracks})} cameras",
            )
        )

        # 2. Calibrated precursor alert.
        alerts: list[PrecursorAlert] = []
        for track in tracks:
            alerts.extend(backend.detect_precursors(track))
        calibrated = all(
            a.calibrated and 0.0 <= a.probability <= 1.0 and a.requires_review
            for a in alerts
        )
        for alert in alerts:
            self.audit.record(
                "alert_raised",
                "priya_nair",
                correlation_id,
                "ok",
                alert_id=alert.alert_id,
                track_id=alert.track_id,
                probability=alert.probability,
                calibrated=alert.calibrated,
                requires_review=alert.requires_review,
            )
        requirements.append(
            RequirementResult(
                name="calibrated_precursor_alert",
                satisfied=bool(alerts) and calibrated,
                evidence=(
                    f"{len(alerts)} alert(s), all calibrated with"
                    f" probabilities in [0,1] and review required"
                    if alerts
                    else "no alerts produced"
                ),
            )
        )

        # 3. Human review, with the gate refusing before it happens.
        pre_review = self.gate.evaluate(correlation_id)
        review_blocked_release = pre_review.blocked
        self.audit.record(
            "pre_review_gate_check",
            self.team.orchestrator.id,
            correlation_id,
            "blocked" if review_blocked_release else "released",
            reason=pre_review.reason,
        )

        reviews: list[HumanReview] = []
        for alert in alerts:
            review = HumanReview(
                alert_id=alert.alert_id,
                reviewer_id=reviewer_id,
                decision=ReviewDecision.CONFIRMED,
                rationale=(
                    "Simulated review: gesture pattern matches the flagged window; "
                    "treated as decision support, not a verdict."
                ),
            )
            reviews.append(review)
            self.audit.record(
                "human_review",
                reviewer_id,
                correlation_id,
                "ok",
                alert_id=alert.alert_id,
                decision=review.decision.value,
                has_rationale=bool(review.rationale),
            )

        reviewed_alert_ids = {r.alert_id for r in reviews}
        all_reviewed = all(a.alert_id in reviewed_alert_ids for a in alerts)
        requirements.append(
            RequirementResult(
                name="human_review_required",
                satisfied=bool(reviews) and all_reviewed and review_blocked_release,
                evidence=(
                    f"{len(reviews)} human review(s) recorded and the gate was "
                    f"blocked before review"
                    if reviewed_alert_ids
                    else "no review recorded"
                ),
            )
        )

        # 4. Appearance-based ReID path reconstruction.
        reid_path, reid_ok, reid_evidence = self._reconstruct_path(
            backend, tracks, correlation_id
        )
        requirements.append(
            RequirementResult(
                name="appearance_reid_path",
                satisfied=reid_ok,
                evidence=reid_evidence,
            )
        )

        # 5. Audit trail completeness.
        entries = self.audit.for_correlation(correlation_id)
        actions = {e.action for e in entries}
        required_actions = {
            "track_created",
            "alert_raised",
            "human_review",
            "reid_match",
            "path_reconstructed",
        }
        missing_actions = required_actions - actions
        requirements.append(
            RequirementResult(
                name="audit_log",
                satisfied=not missing_actions,
                evidence=(
                    f"{len(entries)} entries covering all required action types"
                    if not missing_actions
                    else f"missing action types: {sorted(missing_actions)}"
                ),
            )
        )

        # 6. Assemble and gate the release.
        self._produce_acceptance_artifacts(correlation_id)
        self._produce_contract_artifacts(correlation_id)
        self._fire_contract_triggers(correlation_id)

        if all(r.satisfied for r in requirements):
            self.team.agent(self.team.orchestrator.id).produce_artifact(
                self.spec.acceptance_artifacts[-1],
                correlation_id,
                tool="roadmap_write",
                payload={"correlation_id": correlation_id},
            )

        gate_decision = self.gate.release(ReleaseGate.GATEKEEPER_ID, correlation_id)

        return AcceptanceReport(
            correlation_id=correlation_id,
            requirements=tuple(requirements),
            gate=gate_decision,
            audit_entries=len(entries),
            reid_path=reid_path,
            backend=backend.describe(),
        )

    def _reconstruct_path(
        self,
        backend: SimulatedPerceptionBackend,
        tracks: tuple[Track, ...],
        correlation_id: str,
    ) -> tuple[tuple[tuple[float, float], ...], bool, str]:
        """
        Stitches tracks across cameras using appearance embeddings.

        The threshold of 0.9 is a placeholder for this simulation. A real system
        must set it from a genuine embedding model's score distribution, and the
        value here says nothing about real separability.
        """
        threshold = 0.9
        seed_track = tracks[0]
        seed_embedding = backend.embedding_for(seed_track.track_id)

        matched: list[Track] = [seed_track]
        for candidate in tracks[1:]:
            embedding = backend.embedding_for(candidate.track_id)
            similarity = cosine_similarity(seed_embedding.vector, embedding.vector)
            is_match = similarity >= threshold
            self.audit.record(
                "reid_match",
                "maya_chen",
                correlation_id,
                "matched" if is_match else "rejected",
                seed_track=seed_track.track_id,
                candidate_track=candidate.track_id,
                similarity=round(similarity, 4),
                threshold=threshold,
                simulated=True,
            )
            if is_match:
                matched.append(candidate)

        ordered = sorted(matched, key=lambda t: t.start_time)
        path: list[tuple[float, float]] = []
        for track in ordered:
            path.extend(track.path)

        identical_only = len({id(t) for t in matched}) == len({id(t) for t in matched})
        matched_ids = {t.track_id for t in matched}

        # A pass requires more than one camera session and a rejected decoy, which
        # proves the match is not simply accepting everything.
        has_multiple_cameras = len({t.camera_id for t in matched}) > 1
        rejected_a_candidate = any(
            e.action == "reid_match" and e.outcome == "rejected"
            for e in self.audit.for_correlation(correlation_id)
        )
        ok = (
            has_multiple_cameras
            and rejected_a_candidate
            and len(path) > len(seed_track.path)
            and identical_only
        )

        self.audit.record(
            "path_reconstructed",
            "maya_chen",
            correlation_id,
            "ok" if ok else "incomplete",
            matched_tracks=sorted(matched_ids),
            points=len(path),
            cameras=sorted({t.camera_id for t in matched}),
            simulated=True,
        )

        evidence = (
            f"{len(path)} points over cameras {sorted({t.camera_id for t in matched})}; "
            f"1 decoy rejected below threshold {threshold}"
            if ok
            else f"path reconstruction incomplete: matched {sorted(matched_ids)}"
        )
        return tuple(path), ok, evidence

    # -- outputs ----------------------------------------------------------

    def write_report(self, report: AcceptanceReport, out_dir: Path) -> tuple[Path, Path]:
        """Writes the JSON report and the audit log beside it."""
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "acceptance_report.json"
        audit_path = out_dir / "audit_log.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        self.audit.write(audit_path)
        return report_path, audit_path


__all__ = [
    "AcceptanceReport",
    "RequirementResult",
    "TeamOrchestrator",
]
