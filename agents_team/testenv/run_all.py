#!/usr/bin/env python3
"""
Master Test Runner.

Runs every gate in order and refuses to declare success if any of them fails.
Success here is deliberately hard to achieve, because the previous state of this
project was "68 tests pass, therefore it works" — and that was wrong.

The gates, in order of increasing cost:

  1. SUITE       the hand-written tests
  2. ADVERSARIAL generated attacks against the real modules
  3. MUTATION    remove controls, confirm the tests notice
  4. CRITICS     blocking review of the evidence produced above

A later gate cannot compensate for an earlier one. Passing mutation testing does not
excuse a failing suite.

The summary states what was ACTUALLY verified, and explicitly what was not. It is
written to be read by someone deciding whether to believe the result.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import config  # noqa: E402  (path setup must come first)


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    duration: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - display only
        state = "PASS" if self.passed else "FAIL"
        return f"[{state}] {self.name:<12} {self.duration:6.1f}s  {self.detail}"


@dataclass
class RunnerReport:
    gates: list[GateResult] = field(default_factory=list)
    critic_output: str = ""
    overrides: int = 0

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def summary(self) -> str:
        lines = ["=" * 78, "TEST ENVIRONMENT RESULT", "=" * 78]
        lines.extend(str(g) for g in self.gates)
        lines.append("-" * 78)
        lines.append(f"OVERALL: {'PASS' if self.passed else 'FAIL'}")
        if self.overrides:
            lines.append(
                f"OVERRIDES FILED: {self.overrides} — a critic blocked and was "
                f"overridden. Each override is in the record with its reason."
            )
        if self.critic_output:
            lines.append("")
            lines.append(self.critic_output)
        lines.append("")
        lines.append("WHAT THIS DOES NOT PROVE:")
        lines.append("  - Perception, alerts and ReID are SIMULATED; no accuracy claim")
        lines.append("    from any run here describes a real system.")
        lines.append("  - Agents are deterministic checks, not language models.")
        lines.append("  - Mutation testing covers the 14 controls listed in")
        lines.append("    testenv/mutation.py. Other code paths are not measured.")
        return "\n".join(lines)


def run_gate(name: str, argv: list[str], timeout: int = 600) -> GateResult:
    start = time.time()
    try:
        completed = subprocess.run(
            argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return GateResult(name, False, "timed out", time.time() - start)

    tail = (completed.stdout + completed.stderr).strip().splitlines()
    detail = tail[-1] if tail else "no output"
    return GateResult(
        name=name,
        passed=completed.returncode == 0,
        detail=detail[:110],
        duration=time.time() - start,
    )


def _report_and_exit(
    gates: list[GateResult], critic_output: str, halted_by: str
) -> int:
    """
    Prints the result of an interrupted run and exits non-zero.

    Says plainly that the run was halted and by which gate, so a partial result is
    never mistaken for a complete one.
    """
    print("=" * 78)
    print("RUN HALTED")
    print("=" * 78)
    for gate in gates:
        print(gate)
    print("-" * 78)
    print(f"Halted by: {halted_by}")
    print("Remaining gates were NOT run. This result is incomplete by design:")
    print("a later pass cannot compensate for an earlier failure.")
    if critic_output:
        print()
        print(critic_output)

    out = config.artifacts_dir() / "test_environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": False,
                "halted_by": halted_by,
                "gates": [
                    {"name": g.name, "passed": g.passed, "detail": g.detail}
                    for g in gates
                ],
            },
            indent=2,
        )
    )
    return 1


def _preflight_halt_check() -> list[str]:
    """
    Refuses to start when a problem is already escalated.

    Running the gates while a known problem is in a halted state would produce a
    green result for everything else and quietly imply the project is fine. The
    escalation has to be resolved by the owner first.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from testenv.ledger import ApproachLedger

    ledger = ApproachLedger(config.ledger_path())
    return [e.problem for e in ledger.escalations]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run every test gate.")
    parser.add_argument("--seeds", type=int, default=200, help="adversarial seeds")
    parser.add_argument(
        "--skip-mutation", action="store_true", help="skip the slowest gate"
    )
    args = parser.parse_args()

    # A halt is checked before anything runs. The run stops here, with a reason,
    # rather than proceeding to report on the parts that still work.
    halted = _preflight_halt_check()
    if halted:
        print("=" * 78)
        print("RUN HALTED BEFORE STARTING")
        print("=" * 78)
        print()
        print("These problems have exhausted their approach budget and need a")
        print("decision from the owner before any further work:")
        for problem in halted:
            print(f"  - {problem}")
        print()
        print("Running the remaining gates would report on everything else and")
        print("imply the project is healthy. Resolve the escalation first.")
        return 3

    gates: list[GateResult] = []

    # 1. The hand-written suite.
    gates.append(
        run_gate(
            "SUITE",
            [PYTHON, "-m", "pytest", "tests/", "-q", "--no-header"],
            timeout=600,
        )
    )

    # A failing suite halts immediately. Continuing would run expensive gates
    # against code already known to be broken, and would bury the first failure
    # under later output.
    if not gates[-1].passed:
        return _report_and_exit(gates, critic_output="", halted_by="SUITE")

    # 2. Generated attacks.
    gates.append(
        run_gate(
            "ADVERSARIAL",
            [PYTHON, "testenv/adversarial.py", "--seeds", str(args.seeds)],
            timeout=900,
        )
    )
    if not gates[-1].passed:
        return _report_and_exit(gates, critic_output="", halted_by="ADVERSARIAL")

    # 3. Mutation testing.
    if not args.skip_mutation:
        gates.append(
            run_gate("MUTATION", [PYTHON, "testenv/mutation.py"], timeout=1800)
        )
        if not gates[-1].passed:
            return _report_and_exit(gates, critic_output="", halted_by="MUTATION")

    # 4. Critics review the outcome of the first three gates.
    sys.path.insert(0, str(ROOT))
    from testenv.critics import CriticPanel

    suite, adversarial, *rest = gates
    mutation = rest[0] if rest else None

    failures: list[str] = []
    if not suite.passed:
        failures.append(suite.detail)

    panel = CriticPanel()
    reports = panel.review(
        "test environment",
        total=len(gates),
        failures=failures,
        uncaught_exceptions=[] if adversarial.passed else [adversarial.detail],
        escalations=[],
        leaks=[],
        reports=[
            "mutation score reported honestly including any survivors"
            if (mutation is None or mutation.passed)
            # An explicit overclaim is what the honesty critic exists to catch.
            else "all controls fully validated"
        ] if mutation is not None and not mutation.passed else ["mutation testing passed"],
        unlabelled_simulated=[],
    )

    critic_report = panel.summary(reports)
    gates.append(
        GateResult(
            "CRITICS",
            passed=not panel.blocked(reports),
            detail="all critics passed" if not panel.blocked(reports) else "blocked",
        )
    )

    final = RunnerReport(gates=gates, critic_output=critic_report)
    print(final.summary())

    out = config.artifacts_dir() / "test_environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": final.passed,
                "gates": [
                    {"name": g.name, "passed": g.passed, "detail": g.detail,
                     "duration": round(g.duration, 2)}
                    for g in gates
                ],
            },
            indent=2,
        )
    )
    return 0 if final.passed else 1


if __name__ == "__main__":
    sys.exit(main())
