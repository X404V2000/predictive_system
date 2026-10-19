#!/usr/bin/env python3
"""
Run the integrated acceptance test for the agent team.

Usage:
    python3 agents_team/run_acceptance.py
    python3 agents_team/run_acceptance.py --seed 7 --show-audit
    python3 agents_team/run_acceptance.py register | roster | check | review

Private data (the team specification, roster, audit log and run output) is read from
$AGENTS_DATA_DIR when set, otherwise from ../hiring_dir alongside this package.

Prints a pass/fail summary, writes the report and audit log, and exits non-zero
when the gate is blocked so it can be used in CI.

The perception backend is SIMULATED. See agents/sim_backend.py for what the run
does and does not demonstrate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.orchestrator import TeamOrchestrator
from agents.roles import SpecError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated acceptance test.")
    parser.add_argument("--seed", type=int, default=7, help="scenario seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the report and audit log (default: <data>/run_artifacts)",
    )
    parser.add_argument(
        "--show-audit", action="store_true", help="print every audit entry"
    )
    parser.add_argument(
        "--reviewer", default="operator-17", help="id of the reviewing operator"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="directory holding roster.json and audit.jsonl (default: <data>/registry_store)",
    )
    parser.add_argument(
        "--actor", default="owner", help="who is performing a registry operation"
    )

    subparsers = parser.add_subparsers(dest="command", help="roster operations")

    subparsers.add_parser("register", help="register every agent in the specification")
    subparsers.add_parser("roster", help="list registered agents")
    subparsers.add_parser("check", help="report drift and shared permissions")
    review = subparsers.add_parser(
        "review", help="performance review from the latest audit trail"
    )
    review.add_argument(
        "--audit", type=Path, default=None, help="audit log to review (JSON array)"
    )

    for name, text in (
        ("retire", "retire an agent (record kept)"),
        ("suspend", "suspend an agent"),
        ("activate", "reactivate an agent"),
    ):
        sub = subparsers.add_parser(name, help=text)
        sub.add_argument("agent_id")
        sub.add_argument("--reason", default="")

    args = parser.parse_args()
    # Defaults resolved here rather than at parser construction, so the environment
    # variable is honoured and tests can redirect the data directory.
    from agents import config

    if args.out is None:
        args.out = config.artifacts_dir()
    if args.registry is None:
        args.registry = config.registry_dir()
    return args


def _registry_command(args: argparse.Namespace) -> int:
    """Handles roster operations, which do not run the acceptance test."""
    from agents.registry import AgentRegistry, AgentStatus, RegistryError
    from agents.roles import load_spec

    registry = AgentRegistry(args.registry)

    try:
        spec = load_spec()
    except SpecError as exc:
        print(f"[FAIL] could not load the team specification: {exc}")
        return 2

    try:
        if args.command == "register":
            records = registry.register_team(spec, actor=args.actor)
            print(f"Registered {len(records)} agent(s) in {registry.roster_path}")
            for record in records:
                print(
                    f"  {record.agent_id:<16} {record.agent_handle:<18} "
                    f"v{record.version} {record.status:<10} "
                    f"charter={record.charter_sha256[:10]} hired={record.registered_at}"
                )
            return 0

        if args.command == "roster":
            records = registry.all()
            if not records:
                print("No agents registered. Run: run_acceptance.py register")
                return 0
            print(f"{'agent':<16}{'handle':<18}{'ver':<5}{'status':<11}hired")
            for record in records:
                print(
                    f"{record.agent_id:<16}{record.agent_handle:<18}"
                    f"v{record.version:<4}{record.status:<11}{record.registered_at}"
                )
            return 0

        if args.command == "check":
            if not registry.all():
                print("No agents registered yet; nothing to compare against.")
                return 1
            drift = registry.detect_drift(spec)
            conflicts = registry.check_privilege_conflicts(spec)
            problems = sum(len(v) for v in drift.values())

            print("Drift against new_team.md:")
            for group, members in drift.items():
                print(f"  {group:<18} {members if members else '-'}")
            print()
            print("Shared sensitive permissions:")
            if conflicts:
                for tool, holders in conflicts.items():
                    print(f"  {tool:<24} {holders}")
            else:
                print("  none")

            if problems:
                print()
                print("Run 'register' to accept the specification changes.")
                return 1
            return 0

        if args.command == "review":
            from agents.review import run_review, write_review

            audit_path = args.audit or (
                args.out / "audit_log.json" if (args.out / "audit_log.json").exists() else None
            )
            if audit_path is None or not Path(audit_path).exists():
                print(
                    "[FAIL] no audit log found. Run the acceptance test first, or pass "
                    "--audit <path>."
                )
                return 2

            audit = json.loads(Path(audit_path).read_text())
            report = run_review(spec, registry, audit, source=str(audit_path))
            print(report.summary())

            path = write_review(report, args.out)
            print()
            print(f"review written to: {path}")
            return 0

        if args.command in {"retire", "suspend", "activate"}:
            status = {
                "retire": AgentStatus.RETIRED,
                "suspend": AgentStatus.SUSPENDED,
                "activate": AgentStatus.ACTIVE,
            }[args.command]
            record = registry.set_status(
                args.agent_id, status, actor=args.actor, reason=args.reason
            )
            print(f"{record.agent_id} is now {record.status}")
            return 0

    except RegistryError as exc:
        print(f"[FAIL] {exc}")
        return 2

    print(f"Unknown command {args.command!r}")
    return 2


def main() -> int:
    args = parse_args()

    if args.command:
        return _registry_command(args)

    try:
        orchestrator = TeamOrchestrator()
    except SpecError as exc:
        print(f"[FAIL] could not load the team specification: {exc}")
        return 2

    report = orchestrator.run_integration_acceptance(
        seed=args.seed, reviewer_id=args.reviewer
    )
    print(report.summary())

    report_path, audit_path = orchestrator.write_report(report, args.out)
    print()
    print(f"report : {report_path}")
    print(f"audit  : {audit_path}")

    if args.show_audit:
        print()
        print("Audit trail:")
        for entry in orchestrator.audit.entries:
            who = entry.agent_id or "-"
            print(
                f"  {entry.sequence:3d} {entry.timestamp} {who:<14} "
                f"{entry.action:<24} {entry.outcome}"
            )

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
