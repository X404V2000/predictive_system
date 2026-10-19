"""
Path Configuration.

The agent system is split into two halves that live in different places:

    agents_team/     CODE. Tracked in git, so it survives losing a laptop.
    hiring_dir/      DATA. Never tracked: the team specification, the registered
                     roster, the audit trail, the approach ledger and run output.

Keeping the two together was convenient and wrong. It meant the whole system lived
in an ignored directory, so a dead laptop took the agents with it, and the only way
to back them up would have been to publish private personnel data.

This module is the single place the split is expressed. Nothing else should
hardcode a path to private data; if it does, moving or backing up the data becomes
a search-and-replace exercise, and something will be missed.

Location is resolved in this order:

  1. an explicit path passed by the caller
  2. the $AGENTS_DATA_DIR environment variable
  3. `../hiring_dir` alongside this package

The environment variable exists so a second machine can point at a copied directory
without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directory containing this package (agents_team/). Code lives here.
CODE_ROOT: Path = Path(__file__).resolve().parent.parent

# Environment variable that relocates private data, for a fresh machine.
DATA_DIR_ENV: str = "AGENTS_DATA_DIR"


def data_dir() -> Path:
    """
    Root of the private data directory.

    Resolved on every call rather than cached at import, so a test can redirect it
    with an environment variable and a long-running process picks up a change.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (CODE_ROOT.parent / "hiring_dir").resolve()


def spec_path() -> Path:
    """The team specification, which lives with the private data."""
    return data_dir() / "new_team.md"


def registry_dir() -> Path:
    """Where roster.json and the registry audit log live."""
    return data_dir() / "registry_store"


def ledger_path() -> Path:
    """The approach ledger."""
    return data_dir() / "ledger.jsonl"


def artifacts_dir() -> Path:
    """Reports written by acceptance, mutation and test runs."""
    return data_dir() / "run_artifacts"


def describe() -> dict[str, str]:
    """
    What the current configuration points at.

    Printed by the runner at startup, because a run against the wrong data directory
    silently evaluates the wrong team, and that is worth catching immediately rather
    than after reading a plausible report.
    """
    return {
        "code_root": str(CODE_ROOT),
        "data_dir": str(data_dir()),
        "spec": str(spec_path()),
        "exists": str(spec_path().exists()),
    }


__all__ = [
    "CODE_ROOT",
    "DATA_DIR_ENV",
    "artifacts_dir",
    "data_dir",
    "describe",
    "ledger_path",
    "registry_dir",
    "spec_path",
]
