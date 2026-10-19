#!/usr/bin/env python3
"""
Mutation Testing.

A test suite that never fails proves nothing. This module checks the tests
themselves by deliberately breaking the code and confirming the suite notices.

Each mutation is a realistic bug: an off-by-one, an inverted comparison, a removed
guard, a swallowed exception. The code is copied, mutated, and the test suite is run
against the copy. If the tests still pass, the mutation SURVIVED, which means that
line is not actually covered by behaviour anyone checked.

Survivors are reported, not hidden. A mutation score of 100% is rare and a suite
claiming it is usually measuring the wrong thing; the useful output is the list of
places where the tests are blind.

The repository is never modified: every mutation runs against an isolated copy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect."""

    identifier: str
    file: str
    original: str
    replacement: str
    description: str


@dataclass
class MutationResult:
    mutation: Mutation
    killed: bool
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        state = "KILLED  " if self.killed else "SURVIVED"
        return f"[{state}] {self.mutation.identifier:<14} {self.mutation.description}"


# Mutations target the checks that matter most. A survivor here means a control the
# project depends on is not actually verified.
MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "handoff-both",
        "agents/contracts.py",
        "if artifact_present and trigger_met:",
        "if artifact_present or trigger_met:",
        "handoff passes with only ONE of artifact or trigger",
    ),
    Mutation(
        "gate-keeper",
        "agents/release_gate.py",
        "if agent_id != self.GATEKEEPER_ID:",
        "if False:",
        "any agent may open the release gate",
    ),
    Mutation(
        "gate-tool",
        "agents/release_gate.py",
        "agent.require_tool(GATE_TOOL, correlation_id)",
        "pass",
        "release does not check the gate permission",
    ),
    Mutation(
        "gate-findings",
        "agents/release_gate.py",
        "BLOCKING_SEVERITIES: frozenset[Severity] = frozenset({Severity.HIGH, Severity.CRITICAL})",
        "BLOCKING_SEVERITIES: frozenset[Severity] = frozenset()",
        "high findings no longer block a release",
    ),
    Mutation(
        "alert-calibrated",
        "agents/sim_backend.py",
        "if not self.calibrated:",
        "if False:",
        "an uncalibrated score is accepted as an alert probability",
    ),
    Mutation(
        "alert-range",
        "agents/sim_backend.py",
        "if not 0.0 <= self.probability <= 1.0:",
        "if False:",
        "an out-of-range probability is accepted",
    ),
    Mutation(
        "registry-version",
        "agents/registry.py",
        "if existing[\"charter_sha256\"] == digest:",
        "if True:",
        "a charter change is never detected",
    ),
    Mutation(
        "ledger-similarity",
        "testenv/ledger.py",
        "if similarity >= SIMILARITY_THRESHOLD:",
        "if False:",
        "retry-in-disguise is never refused",
    ),
    Mutation(
        "ledger-budget",
        "testenv/ledger.py",
        "if len(failed) >= MAX_FAILED_APPROACHES:",
        "if len(failed) >= 999:",
        "escalation never fires",
    ),
    Mutation(
        "review-settled",
        "agents/review.py",
        "assessed = list(final_by_target.values())",
        "assessed = attempts",
        "deliberate gate refusals are scored against the agent again",
    ),
    Mutation(
        "review-evidence",
        "agents/review.py",
        "return self.denominator >= MIN_SAMPLE_FOR_RATE",
        "return self.denominator > 0",
        "a sample of one is reported as a percentage",
    ),
    Mutation(
        "kpi-default",
        "agents/kpi.py",
        "return KpiAssessment(\n            kpi, Measurability.NOT_MEASURABLE, None, DEFAULT_BLOCKER\n        )",
        "return KpiAssessment(kpi, Measurability.MEASURABLE, MetricName.HANDOFF_COMPLIANCE, \"\")",
        "an unknown KPI is claimed as measurable",
    ),
    Mutation(
        "override-reason",
        "testenv/critics.py",
        "if len(self.reason.strip()) < 20:",
        "if False:",
        "an override can be filed without a reason",
    ),
    Mutation(
        "honesty-claim",
        "testenv/critics.py",
        "if phrase in text:",
        "if False:",
        "the honesty critic stops detecting overclaiming",
    ),
)


def _copy_package(destination: Path) -> None:
    """Copies only what a test run needs, without caches or stored state."""
    for name in ("agents", "testenv", "tests"):
        source = PACKAGE_ROOT / name
        if source.is_dir():
            shutil.copytree(
                source,
                destination / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

    # The mutated copy also needs the private data it reads, or every run fails for
    # want of a specification rather than reporting a real mutation result.
    sys.path.insert(0, str(PACKAGE_ROOT))
    from agents import config

    source_spec = config.spec_path()
    if source_spec.exists():
        shutil.copy2(source_spec, destination / "new_team.md")


def _run_tests(root: Path, timeout: int = 120) -> tuple[bool, str]:
    """Runs the suite in a copy. Returns (passed, output)."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--no-header"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    text = (completed.stdout + completed.stderr).strip().splitlines()
    return completed.returncode == 0, " | ".join(text[-2:]) if text else ""


def run_mutation(entry: Mutation) -> MutationResult:
    """
    Applies one mutation to an isolated copy and runs the suite.

    A KILLED result means the tests detected the defect. A SURVIVED result means
    the tests passed with that control removed.
    """
    with tempfile.TemporaryDirectory(prefix="mut_" + entry.identifier + "_") as tmp:
        root = Path(tmp)
        _copy_package(root)

        target = root / entry.file
        if not target.exists():
            return MutationResult(entry, killed=False, detail="target file missing")

        text = target.read_text()
        if entry.original not in text:
            return MutationResult(
                entry,
                killed=False,
                detail="mutation anchor not found; the code changed and this "
                "mutation is stale",
            )

        target.write_text(text.replace(entry.original, entry.replacement, 1))

        passed, detail = _run_tests(root)
        return MutationResult(entry, killed=not passed, detail=detail)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run mutation testing.")
    parser.add_argument("--only", default=None, help="run a single mutation id")
    args = parser.parse_args()

    entries = [m for m in MUTATIONS if args.only is None or m.identifier == args.only]
    if not entries:
        print(f"No mutation with id {args.only!r}")
        return 2

    print("MUTATION TESTING")
    print("Each mutation removes a control. KILLED means the tests noticed.")
    print()

    results = [run_mutation(entry) for entry in entries]
    for result in results:
        print(result)
        if not result.killed:
            print(f"          detail: {result.detail}")

    killed = sum(1 for r in results if r.killed)
    total = len(results)
    survivors = [r.mutation.identifier for r in results if not r.killed]

    print()
    print(f"Mutation score: {killed}/{total} = {killed / total:.0%}")
    if survivors:
        print(f"SURVIVORS (unverified controls): {', '.join(survivors)}")
        print()
        print("A survivor means a control can be removed without any test failing.")
        print("It is a gap in the tests, not proof the code is wrong.")

    report = {
        "killed": killed,
        "total": total,
        "score": killed / total,
        "survivors": survivors,
        "results": [
            {"id": r.mutation.identifier, "killed": r.killed, "detail": r.detail}
            for r in results
        ],
    }
    sys.path.insert(0, str(PACKAGE_ROOT))
    from agents import config

    out = config.artifacts_dir() / "mutation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    return 0 if not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
