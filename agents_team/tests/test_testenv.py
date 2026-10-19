"""
Tests for the testing environment itself.

Mutation testing found that the ledger and critics had no tests: their controls
could be deleted and the suite still passed. This file closes that gap, and each
test here corresponds to a mutation that previously SURVIVED.

If these tests are removed, the mutation score drops and `mutation.py` reports the
survivor again, so the gap is visible rather than silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from testenv.critics import (
    CriticPanel,
    CriticRole,
    Finding,
    Override,
    Verdict,
)
from testenv.generators import Generator
from testenv.ledger import (
    MAX_FAILED_APPROACHES,
    Halted,
    ApproachLedger,
    AttemptOutcome,
    LedgerError,
    RetryInDisguise,
    approach_similarity,
)


@pytest.fixture
def ledger(tmp_path: Path) -> ApproachLedger:
    return ApproachLedger(tmp_path / "ledger.jsonl")


# --- ledger: retry-in-disguise ---------------------------------------------


def test_identical_approach_is_refused(ledger: ApproachLedger) -> None:
    """
    Kills the 'ledger-similarity' mutation.

    Repeating the same approach must be refused, not counted as progress.
    """
    problem = "tests fail"
    ledger.record(problem, "add a retry around the parser", AttemptOutcome.FAILED, "still fails")

    with pytest.raises(RetryInDisguise):
        ledger.record(problem, "add a retry around the parser", AttemptOutcome.FAILED, "still fails")

    # The refusal must not consume budget.
    assert ledger.remaining_attempts(problem) == MAX_FAILED_APPROACHES - 1


def test_reworded_approach_is_refused(ledger: ApproachLedger) -> None:
    """Paraphrasing the same idea is still the same idea."""
    problem = "tests fail"
    ledger.record(
        problem,
        "wrap the json parser in try except and return empty",
        AttemptOutcome.FAILED,
        "crashes on nested input",
    )
    with pytest.raises(RetryInDisguise):
        ledger.record(
            problem,
            "wrap json parser in try except returning empty",
            AttemptOutcome.FAILED,
            "same crash",
        )


def test_genuinely_different_approach_is_accepted(ledger: ApproachLedger) -> None:
    """The guard must not block real progress."""
    problem = "tests fail"
    ledger.record(problem, "add a retry around the parser", AttemptOutcome.FAILED, "no change")
    ledger.record(problem, "replace the parser with a tokeniser", AttemptOutcome.FAILED, "partial")

    assert ledger.remaining_attempts(problem) == MAX_FAILED_APPROACHES - 2


def test_similarity_is_symmetric_and_bounded() -> None:
    a = "rewrite the parser with a tokeniser"
    b = "use a tokeniser to rewrite the parser"
    assert approach_similarity(a, b) == pytest.approx(approach_similarity(b, a))
    assert 0.0 <= approach_similarity(a, b) <= 1.0


# --- ledger: escalation budget ---------------------------------------------


def test_three_distinct_failures_escalate(ledger: ApproachLedger) -> None:
    """
    Kills the 'ledger-budget' mutation.

    The third failure must HALT the run, not merely return a value a caller could
    ignore. A decision from the owner is required before further work.
    """
    problem = "tests fail"
    ledger.record(problem, "add retry logic to the loop", AttemptOutcome.FAILED, "no change")
    ledger.record(problem, "rewrite the loop with a state machine", AttemptOutcome.FAILED, "partial")

    with pytest.raises(Halted) as raised:
        ledger.record(
            problem, "replace the loop with an event bus", AttemptOutcome.FAILED, "deadlock"
        )

    assert ledger.is_escalated(problem)
    assert raised.value.escalation.problem == problem
    assert ledger.remaining_attempts(problem) == 0


def test_escalation_halts_further_attempts(ledger: ApproachLedger) -> None:
    """After escalation the problem is closed to further work."""
    problem = "tests fail"
    for i, approach in enumerate(
        ["add retry logic", "rewrite as state machine"]
    ):
        ledger.record(problem, approach, AttemptOutcome.FAILED, f"failure {i}")
    with pytest.raises(Halted):
        ledger.record(problem, "replace with event bus", AttemptOutcome.FAILED, "f3")

    assert ledger.is_escalated(problem)
    with pytest.raises(LedgerError, match="already escalated"):
        ledger.record(problem, "a fourth idea entirely", AttemptOutcome.FAILED, "fails")


def test_success_does_not_escalate(ledger: ApproachLedger) -> None:
    """A solved problem never exhausts a budget."""
    problem = "tests fail"
    ledger.record(problem, "add retry logic", AttemptOutcome.FAILED, "no change")
    ledger.record(problem, "rewrite the loop", AttemptOutcome.SUCCEEDED, "all green")

    assert not ledger.is_escalated(problem)
    assert len(ledger.failed_attempts(problem)) == 1


def test_attempt_without_evidence_is_refused(ledger: ApproachLedger) -> None:
    """
    "It didn't work" is not a data point. An attempt must record what it produced,
    otherwise the escalation report carries no diagnosis.
    """
    with pytest.raises(LedgerError, match="evidence"):
        ledger.record("p", "some approach", AttemptOutcome.FAILED, "   ")


def test_escalation_report_includes_every_approach(ledger: ApproachLedger) -> None:
    """The owner needs the diagnosis, not just the fact of failure."""
    problem = "p"
    for i, approach in enumerate(["alpha strategy", "beta strategy"]):
        ledger.record(problem, approach, AttemptOutcome.FAILED, f"evidence {i}")
    with pytest.raises(Halted):
        ledger.record(problem, "gamma strategy", AttemptOutcome.FAILED, "evidence 2")

    report = ledger.escalation(problem).report()
    for word in ("alpha", "beta", "gamma", "evidence 0", "STOP"):
        assert word in report


def test_ledger_persists_across_restarts(tmp_path: Path) -> None:
    """An escalation must survive a process restart, or it can be forgotten."""
    path = tmp_path / "ledger.jsonl"
    first = ApproachLedger(path)
    problem = "p"
    for approach in ["alpha", "beta"]:
        first.record(problem, approach, AttemptOutcome.FAILED, "evidence")
    with pytest.raises(Halted):
        first.record(problem, "gamma", AttemptOutcome.FAILED, "evidence")

    reopened = ApproachLedger(path)
    assert reopened.is_escalated(problem)
    with pytest.raises(LedgerError):
        reopened.record(problem, "delta", AttemptOutcome.FAILED, "evidence")


# --- critics: blocking -----------------------------------------------------


def test_correctness_critic_blocks_on_failure() -> None:
    panel = CriticPanel()
    reports = panel.review("build", failures=["test_x failed"], total=10)

    correctness = next(r for r in reports if r.critic is CriticRole.CORRECTNESS)
    assert correctness.blocked
    assert panel.blocked(reports)


def test_correctness_critic_blocks_on_zero_tests() -> None:
    """An empty run looks like success and is not."""
    panel = CriticPanel()
    reports = panel.review("build", failures=[], total=0)
    correctness = next(r for r in reports if r.critic is CriticRole.CORRECTNESS)
    assert correctness.blocked


def test_all_critics_must_pass() -> None:
    """One failing critic blocks the whole change."""
    panel = CriticPanel()
    reports = panel.review("build", failures=[], total=5, leaks=["ghp_abc"])
    verdicts = {r.critic: r.blocked for r in reports}

    assert verdicts[CriticRole.CORRECTNESS] is False
    assert verdicts[CriticRole.SECURITY] is True
    assert panel.blocked(reports)


def test_security_critic_blocks_on_escalation() -> None:
    panel = CriticPanel()
    reports = panel.review("build", total=5, escalations=["priya opened the gate"])
    security = next(r for r in reports if r.critic is CriticRole.SECURITY)
    assert security.blocked
    assert security.findings[0].severity == "critical"


# --- critics: honesty (kills 'honesty-claim') ------------------------------


@pytest.mark.parametrize(
    "claim",
    ["100% accurate", "fully validated", "production ready", "no known issues"],
)
def test_honesty_critic_detects_overclaiming(claim: str) -> None:
    """
    Kills the 'honesty-claim' mutation.

    A report asserting capability the evidence does not support must be blocked,
    including when the person writing it is the one running the tests.
    """
    panel = CriticPanel()
    reports = panel.review("report", total=5, reports=[f"The system is {claim}"])
    honesty = next(r for r in reports if r.critic is CriticRole.HONESTY)

    assert honesty.blocked
    assert any(claim in f.summary for f in honesty.findings)


def test_honesty_critic_blocks_unlabelled_simulation() -> None:
    panel = CriticPanel()
    reports = panel.review("report", total=5, unlabelled_simulated=["track 1 path"])
    honesty = next(r for r in reports if r.critic is CriticRole.HONESTY)
    assert honesty.blocked


def test_honesty_critic_passes_a_careful_report() -> None:
    """It must not block a truthful report, or it becomes noise."""
    panel = CriticPanel()
    text = (
        "Accuracy on held-out subjects was 81.6%. Perception is simulated and "
        "the transition model shows no lift over a constant predictor."
    )
    reports = panel.review("report", total=5, reports=[text])
    honesty = next(r for r in reports if r.critic is CriticRole.HONESTY)
    assert not honesty.blocked


# --- overrides (kills 'override-reason') -----------------------------------


def test_override_requires_a_substantive_reason() -> None:
    """
    Kills the 'override-reason' mutation.

    A block must be answered with evidence or reasoning. "I disagree" is neither,
    and allowing it turns the critic into a formality.
    """
    with pytest.raises(ValueError, match="substantive reason"):
        Override(
            critic=CriticRole.CORRECTNESS,
            target="build",
            finding_codes=("COR-001",),
            overridden_by="lead",
            reason="no",
        )


def test_override_is_recorded_with_its_reason() -> None:
    override = Override(
        critic=CriticRole.CORRECTNESS,
        target="build",
        finding_codes=("COR-001",),
        overridden_by="lead",
        reason="the failing test asserts behaviour we deliberately removed",
    )
    assert override.finding_codes == ("COR-001",)
    assert "deliberately removed" in override.reason
    assert override.at


def test_override_does_not_pass_a_blocked_critic() -> None:
    """
    An override is a record of disagreement, not a way to make a block disappear.
    The critic's verdict is a function of evidence only.
    """
    panel = CriticPanel()
    reports = panel.review("build", failures=["bad"], total=1)
    assert panel.blocked(reports)

    Override(
        critic=CriticRole.CORRECTNESS,
        target="build",
        finding_codes=("COR-001",),
        overridden_by="lead",
        reason="we accept this failure for a documented reason",
    )
    # The verdict is unchanged: recording disagreement changes no outcome.
    assert panel.blocked(reports)


# --- generators ------------------------------------------------------------


def test_generators_are_reproducible() -> None:
    """A failing seed must regenerate the exact failing input."""
    first = [Generator(7).spec_dict() for _ in range(3)]
    second = [Generator(7).spec_dict() for _ in range(3)]
    assert first == second


def test_generators_produce_damaged_inputs() -> None:
    """
    A generator that only produced valid specs would never test the validator.
    """
    from agents.roles import REQUIRED_ROLE_FIELDS

    damaged = 0
    for seed in range(60):
        role = Generator(seed).role_dict()
        if any(not role.get(f) for f in REQUIRED_ROLE_FIELDS):
            damaged += 1
    assert damaged > 0, "generator never produced an invalid role"


def test_unicode_generator_includes_non_ascii() -> None:
    text = "".join(Generator(s).unicode_text(10) for s in range(20))
    assert any(ord(ch) > 127 for ch in text)


# --- halt semantics --------------------------------------------------------


def test_escalation_is_written_to_disk_before_the_halt(ledger: ApproachLedger) -> None:
    """
    The escalation record must exist even if the caller dies on the exception.

    Otherwise the one thing the owner needs to read would be lost exactly when the
    run stopped unexpectedly.
    """
    problem = "p"
    for approach in ["alpha technique", "beta technique"]:
        ledger.record(problem, approach, AttemptOutcome.FAILED, "evidence")

    with pytest.raises(Halted):
        ledger.record(problem, "gamma technique", AttemptOutcome.FAILED, "evidence")

    # A fresh ledger over the same file must also see the escalation.
    reopened = ApproachLedger(ledger.path)
    assert reopened.is_escalated(problem)
    assert reopened.escalation(problem) is not None


def test_halt_cannot_be_ignored_as_a_return_value(ledger: ApproachLedger) -> None:
    """
    A caller that does not expect a halt still stops.

    This is the point of raising rather than returning: a gate runner written
    without halt handling cannot accidentally continue past it.
    """
    problem = "p"
    ledger.record(problem, "alpha technique", AttemptOutcome.FAILED, "e")
    ledger.record(problem, "beta technique", AttemptOutcome.FAILED, "e")

    continued = False
    try:
        ledger.record(problem, "gamma technique", AttemptOutcome.FAILED, "e")
        continued = True
    except Halted:
        pass
    assert not continued


def test_runner_refuses_to_start_when_already_escalated(tmp_path: Path) -> None:
    """
    A halted problem must stop the run BEFORE the gates execute.

    Otherwise the runner reports on everything that still works and implies the
    project is healthy while a known problem sits unresolved.
    """
    import subprocess
    import sys

    from agents import config
    from testenv.ledger import ApproachLedger

    root = Path(__file__).resolve().parent.parent
    # The ledger now lives in the private data directory, not beside the code. The
    # test must write to the same place the spawned runner reads from, or it proves
    # nothing about the runner's pre-flight check.
    ledger_path = config.ledger_path()
    ledger = ApproachLedger(ledger_path)
    problem = "_preflight_probe"
    if not ledger.is_escalated(problem):
        ledger.record(problem, "alpha technique", AttemptOutcome.FAILED, "e")
        ledger.record(problem, "beta technique", AttemptOutcome.FAILED, "e")
        try:
            ledger.record(problem, "gamma technique", AttemptOutcome.FAILED, "e")
        except Halted:
            pass

    try:
        completed = subprocess.run(
            [sys.executable, "testenv/run_all.py", "--skip-mutation"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 3, completed.stdout[-400:]
        assert "RUN HALTED BEFORE STARTING" in completed.stdout
        assert problem in completed.stdout
    finally:
        # Remove the probe escalation so later runs are not affected.
        lines = [
            line
            for line in ledger_path.read_text().splitlines()
            if problem not in line
        ]
        ledger_path.write_text("\n".join(lines) + "\n")


# --- ledger resolution -----------------------------------------------------


def test_resolved_problem_leaves_the_open_list(ledger: ApproachLedger) -> None:
    """
    A fixed problem must stop appearing as open.

    A list that keeps showing already-fixed problems trains the reader to ignore
    it, which is how a real escalation gets missed.
    """
    problem = "p"
    ledger.record(problem, "alpha technique", AttemptOutcome.FAILED, "e")
    assert problem in ledger.open_problems()

    ledger.resolve(problem, "root cause identified and fixed at the boundary")

    assert problem not in ledger.open_problems()
    assert ledger.is_resolved(problem)


def test_resolution_requires_a_substantive_reason(ledger: ApproachLedger) -> None:
    """'fixed' is not a record of what was done."""
    problem = "p"
    ledger.record(problem, "alpha technique", AttemptOutcome.FAILED, "e")

    with pytest.raises(LedgerError, match="substantive reason"):
        ledger.resolve(problem, "fixed")


def test_resolution_preserves_attempt_history(ledger: ApproachLedger) -> None:
    """Resolving must not delete what was tried."""
    problem = "p"
    ledger.record(problem, "alpha technique", AttemptOutcome.FAILED, "e")
    ledger.resolve(problem, "root cause identified and fixed at the boundary")

    assert len(ledger.failed_attempts(problem)) == 1
    assert ledger.resolution_of(problem)


def test_resolution_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = ApproachLedger(path)
    problem = "p"
    first.record(problem, "alpha technique", AttemptOutcome.FAILED, "e")
    first.resolve(problem, "root cause identified and fixed at the boundary")

    reopened = ApproachLedger(path)
    assert reopened.is_resolved(problem)
    assert reopened.open_problems() == ()


def test_resolved_problem_does_not_halt_the_runner(tmp_path: Path) -> None:
    """A resolved (not escalated) problem must not block a run."""
    path = tmp_path / "ledger.jsonl"
    ledger = ApproachLedger(path)
    problem = "p"
    ledger.record(problem, "alpha technique", AttemptOutcome.FAILED, "e")
    ledger.resolve(problem, "root cause identified and fixed at the boundary")

    assert ledger.escalations == ()
