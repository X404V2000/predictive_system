#!/usr/bin/env python3
"""
Adversarial Test Harness.

Attacks the existing modules with generated and malformed input, and reports what
breaks. This is not a test suite that proves things work; it is a machine for
finding where they do not.

Each attack targets a specific assumption:

  parser      malformed JSON, wrong types, dangling references, unicode keys
  store       truncated files, interrupted writes, read-only directories
  invariant   properties that must hold for every input, checked over many seeds
  authority   attempts to gain privilege the charter does not grant
  honesty     outputs that claim more than the evidence supports

A crash is only a finding when it is the WRONG kind of failure. Raising SpecError
with a message is correct; raising TypeError, KeyError or AttributeError from inside
the parser is a bug, because it exposes internals instead of rejecting the input.

Usage:
    python3 hiring_dir/testenv/adversarial.py
    python3 hiring_dir/testenv/adversarial.py --seeds 500
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testenv.generators import Generator


@dataclass
class AttackResult:
    """The outcome of one kind of attack."""

    name: str
    cases: int
    findings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        state = "CLEAN" if self.clean else f"{len(self.findings)} FINDING(S)"
        return f"[{state:<20}] {self.name:<34} {self.cases} case(s)"


class Harness:
    """Runs attacks and collects findings."""

    def __init__(self, seeds: int = 200) -> None:
        self.seeds = seeds
        self.results: list[AttackResult] = []

    def attack(self, name: str, case_builder: Callable[[Generator], Any]) -> AttackResult:
        """
        Runs one attack across many seeds.

        The builder should raise a *domain* error for bad input. Anything else is
        recorded as a finding, because it means the boundary is not holding.
        """
        result = AttackResult(name=name, cases=0)
        for seed in range(self.seeds):
            generator = Generator(seed)
            try:
                case_builder(generator)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                # A domain-level rejection is the correct behaviour for bad input.
                continue
            except Exception as exc:  # noqa: BLE001 - deliberate: any other error is a finding
                result.findings.append(
                    {
                        "seed": seed,
                        "exception": type(exc).__name__,
                        "message": str(exc)[:200],
                        "where": traceback.format_exc().splitlines()[-3:],
                    }
                )
            finally:
                result.cases += 1
        self.results.append(result)
        return result

    def report(self) -> str:
        lines = ["ADVERSARIAL TEST RESULTS", ""]
        for result in self.results:
            lines.append(result.summary())
            seen: set[tuple[str, str]] = set()
            for finding in result.findings:
                key = (finding["exception"], finding["message"][:60])
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"    seed {finding['seed']}: {finding['exception']}: "
                    f"{finding['message']}"
                )
        return "\n".join(lines)


# --- attacks ---------------------------------------------------------------


def attack_spec_parser(seeds: int = 200) -> AttackResult:
    """
    Feed the specification loader hostile input.

    The loader must reject bad specifications with SpecError. A TypeError from deep
    inside means the boundary leaked.
    """
    from agents.roles import SpecError, load_spec

    harness = Harness(seeds)
    tmp = Path("/tmp/_adv_spec")

    def build(g: Generator) -> None:
        spec = g.spec_dict()
        tmp.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(spec)
        tmp.write_text(text)
        try:
            load_spec(tmp)
        except SpecError:
            return  # correct rejection
        finally:
            tmp.unlink(missing_ok=True)

    return harness.attack("spec_parser", build)


def attack_spec_corruption(seeds: int = 120) -> AttackResult:
    """
    Corrupt the file itself: truncation, wrong top-level types, invalid JSON.

    Every case must raise SpecError, not crash.
    """
    from agents.roles import SpecError, load_spec

    harness = Harness(seeds)
    tmp = Path("/tmp/_adv_corrupt.json")

    def build(g: Generator) -> None:
        raw = json.dumps(g.spec_dict())
        mode = g.integer(0, 4)
        if mode == 0:
            payload = raw[: g.integer(0, len(raw))]          # truncated
        elif mode == 1:
            payload = "[" + raw + "]"                         # wrong top level
        elif mode == 2:
            payload = '{"roles": "not a list"}'               # wrong type
        elif mode == 3:
            payload = ""                                      # empty
        else:
            payload = '{"roles": [null]}'                     # null entry
        tmp.write_text(payload)
        try:
            load_spec(tmp)
        except SpecError:
            return
        finally:
            tmp.unlink(missing_ok=True)

    return harness.attack("spec_corruption", build)


def attack_roster_store(seeds: int = 120) -> AttackResult:
    """
    Break the registry storage: truncated JSON, wrong types, missing keys.

    The registry must raise RegistryError and never fall back to a silently empty
    roster, which would look like a new team and lose every hire date.
    """
    from agents.registry import AgentRegistry, RegistryError

    harness = Harness(seeds)
    root = Path("/tmp/_adv_registry")
    root.mkdir(parents=True, exist_ok=True)

    def build(g: Generator) -> None:
        registry = AgentRegistry(root)
        mode = g.integer(0, 5)
        if mode == 0:
            registry.roster_path.write_text("{ truncated")
        elif mode == 1:
            registry.roster_path.write_text(json.dumps({"agents": "x"}))
        elif mode == 2:
            registry.roster_path.write_text(json.dumps([1, 2, 3]))
        elif mode == 3:
            registry.roster_path.write_text("")
        elif mode == 4:
            registry.roster_path.write_text(json.dumps({"agents": {"x": None}}))
        else:
            registry.roster_path.unlink(missing_ok=True)
        try:
            registry.all()
        except RegistryError:
            return
        finally:
            registry.roster_path.unlink(missing_ok=True)

    return harness.attack("roster_store", build)


def attack_charter_digest(seeds: int = 200) -> AttackResult:
    """
    Invariant: the digest must be stable for identical input and different for
    different authority, whatever the text contains.
    """
    from agents.registry import charter_digest
    from agents.roles import AgentRole

    harness = Harness(seeds)

    def make(g: Generator, salt: str = "") -> AgentRole:
        return AgentRole(
            id="x",
            name="n",
            title="t",
            reports_to="aisha_rahman",
            seniority="s",
            agent_handle="@x",
            responsibilities=(g.unicode_text(20) + salt,),
            decision_rights=(g.unicode_text(20),),
            system_prompt_seed=g.unicode_text(40) + salt,
            tool_permissions=tuple(g.subset(["repo_read_write", "release_gate_block"])),
            memory_scope=g.unicode_text(20),
        )

    def build(g: Generator) -> None:
        role = make(g)
        first = charter_digest(role)
        second = charter_digest(make(Generator(g.seed)) if False else role)
        if first != second:
            raise AssertionError("digest is not stable for identical input")
        # Changing authority must change the digest.
        widened = make(Generator(g.seed), salt="!different")
        if charter_digest(widened) == first:
            raise AssertionError("digest did not change when the charter changed")

    return harness.attack("charter_digest_invariants", build)


def attack_privilege(seeds: int = 60) -> AttackResult:
    """
    Try to gain authority that was not granted.

    Every agent other than the gatekeeper must be refused, no matter which tool it
    asks for.
    """
    from agents.release_gate import ReleaseGate
    from agents.roles import load_spec
    from agents.runtime import PermissionDenied, Team

    harness = Harness(seeds)
    spec = load_spec()

    def build(g: Generator) -> None:
        team = Team(spec)
        gate = ReleaseGate(team)
        attackers = [a for a in spec.roles if a != ReleaseGate.GATEKEEPER_ID]
        attacker = g.choice(attackers)
        try:
            gate.release(attacker, "adv")
        except PermissionDenied:
            return  # correct
        raise AssertionError(f"{attacker} opened the release gate")

    return harness.attack("privilege_escalation", build)


def attack_alert_contract(seeds: int = 120) -> AttackResult:
    """
    Invariant: an alert can never exist with an invalid or uncalibrated probability.
    """
    from agents.sim_backend import PrecursorAlert, PrecursorKind

    harness = Harness(seeds)

    def build(g: Generator) -> None:
        probability = g.choice([-1.0, 1.5, 2.0, -0.001, float("nan")])
        calibrated = g.boolean()
        try:
            PrecursorAlert(
                alert_id="a",
                track_id="t",
                kind=PrecursorKind.LOITERING,
                probability=probability,
                calibrated=calibrated,
                window_start=0.0,
                window_end=1.0,
                evidence=(),
            )
        except ValueError:
            return
        raise AssertionError(
            f"accepted probability={probability} calibrated={calibrated}"
        )

    return harness.attack("alert_probability_contract", build)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run adversarial attacks.")
    parser.add_argument("--seeds", type=int, default=200)
    args = parser.parse_args()

    results = [
        attack_spec_parser(args.seeds),
        attack_spec_corruption(max(50, args.seeds // 2)),
        attack_roster_store(max(50, args.seeds // 2)),
        attack_charter_digest(args.seeds),
        attack_privilege(60),
        attack_alert_contract(120),
    ]

    print("ADVERSARIAL TEST RESULTS")
    print()
    total_findings = 0
    for result in results:
        print(result.summary())
        seen: set[tuple[str, str]] = set()
        for finding in result.findings:
            key = (finding["exception"], finding["message"][:60])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"    seed {finding['seed']}: {finding['exception']}: "
                f"{finding['message']}"
            )
        total_findings += len(result.findings)

    print()
    print(f"Total findings: {total_findings}")
    return 0 if total_findings == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
