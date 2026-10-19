"""
Property-Based Generators.

`hypothesis` is not installed and I am not adding dependencies for this. This module
implements the small subset that matters: seeded random generation of inputs, so the
code is tested against thousands of cases rather than the handful I thought of.

The distinction that makes this worth doing:

    EXAMPLE TESTS   assert that specific hand-written inputs produce specific
                    outputs. They only cover cases the author imagined, and they
                    share the author's blind spots.

    GENERATED TESTS assert INVARIANTS over many random inputs. They do not need to
                    predict the output, only that a property always holds.

Invariants are the lever. "The roster round-trips through JSON" or "a probability is
always in [0,1]" can be checked for inputs nobody wrote down.

Every generator is seeded so a failure is reproducible: the failing seed is part of
the output, and `replay(seed)` regenerates the exact sequence.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence


@dataclass(frozen=True)
class Generated:
    """A generated value and the seed that produced it, so failures replay."""

    value: Any
    seed: int


class Generator:
    """
    Seeded value generator.

    Wraps `random.Random` so a run can be replayed exactly from its seed. Callers
    keep the seed with any failure, which turns a flaky-looking failure into a
    reproducible one.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    # -- primitives -------------------------------------------------------

    def integer(self, low: int = -1000, high: int = 1000) -> int:
        return self._rng.randint(low, high)

    def probability(self) -> float:
        """A value in [0, 1]."""
        return self._rng.random()

    def text(self, max_length: int = 32) -> str:
        """
        Arbitrary printable text, including characters that commonly break parsers.

        The alphabet deliberately includes quotes, backslashes, newlines and
        control characters, because those are what actually break naive handling.
        """
        alphabet = string.ascii_letters + string.digits + ' "\'\\\n\t{}[]:,.-_/'
        return "".join(
            self._rng.choice(alphabet) for _ in range(self._rng.randint(0, max_length))
        )

    def unicode_text(self, max_length: int = 16) -> str:
        """
        Text including non-ASCII, combining marks and emoji.

        Length and comparison bugs surface here that plain ASCII never triggers.
        """
        pools = [
            "abcXYZ0189",
            "\u00e9\u00fc\u00f1\u00e5",          # accented Latin
            "\u0301\u0308",                       # combining marks
            "\u4e2d\u6587\u65e5\u672c",           # CJK
            "\U0001f600\U0001f4a5\u2764",         # emoji
            "\u200b\u200d\ufeff",                 # zero width, joiners, BOM
        ]
        return "".join(
            self._rng.choice(self._rng.choice(pools))
            for _ in range(self._rng.randint(0, max_length))
        )

    def boolean(self) -> bool:
        return self._rng.random() < 0.5

    def choice(self, options: Sequence[Any]) -> Any:
        return self._rng.choice(list(options))

    def subset(self, options: Sequence[Any]) -> list[Any]:
        """A random subset, possibly empty, in random order."""
        items = list(options)
        return [x for x in items if self._rng.random() < 0.5]

    # -- structured values ------------------------------------------------

    def agent_id(self) -> str:
        base = "".join(
            self._rng.choice(string.ascii_lowercase + "_")
            for _ in range(self._rng.randint(1, 12))
        )
        # Sometimes collide on purpose so duplicate handling is exercised.
        if self._rng.random() < 0.2:
            return "maya_chen"
        return base or "x"

    def role_dict(self) -> dict[str, Any]:
        """
        A role specification, valid or deliberately damaged.

        Around one in five are damaged: a missing required field, a wrong type, or
        an empty list. A generator that only produced valid roles would never test
        the validation that exists to reject bad ones.
        """
        role: dict[str, Any] = {
            "id": self.agent_id(),
            "name": self.text(20),
            "title": self.text(30),
            "reports_to": self.choice(["aisha_rahman", "project_sponsor", self.agent_id()]),
            "seniority": self.text(10),
            "agent_handle": "@" + self.text(12),
            "responsibilities": [self.text(20) for _ in range(self._rng.randint(1, 3))],
            "decision_rights": [self.text(20) for _ in range(self._rng.randint(1, 3))],
            "system_prompt_seed": self.text(60),
            "tool_permissions": self.subset(
                ["repo_read_write", "benchmark_runner", "release_gate_block",
                 "model_registry_write", "api_deploy"]
            ),
            "memory_scope": self.text(30),
            "kpis": [self.text(20) for _ in range(self._rng.randint(0, 3))],
        }

        if self._rng.random() < 0.2:
            damage = self._rng.choice(
                ["drop_field", "empty_list", "wrong_type", "null_value"]
            )
            key = self._rng.choice(list(role))
            if damage == "drop_field":
                role.pop(key, None)
            elif damage == "empty_list":
                role[key] = []
            elif damage == "wrong_type":
                role[key] = self.integer()
            else:
                role[key] = None
        return role

    def spec_dict(self) -> dict[str, Any]:
        """A team specification, usually consistent, sometimes dangling."""
        roles = [self.role_dict() for _ in range(self._rng.randint(0, 5))]
        ids = [r.get("id") for r in roles if isinstance(r.get("id"), str)]

        contracts = []
        for _ in range(self._rng.randint(0, 4)):
            source = self.choice(ids) if ids else "ghost"
            target = self.choice(ids) if ids else "ghost"
            # Occasionally reference an agent that does not exist.
            if self._rng.random() < 0.25:
                source = "missing_" + self.text(6)
            contracts.append(
                {
                    "from": source,
                    "to": target,
                    "artifact": self.text(20) or "a",
                    "trigger": self.text(20) or "t",
                }
            )

        return {
            "schema_version": "1.0",
            "team_id": self.text(12),
            "synthetic": True,
            "project_context": {"name": self.text(20), "goal": self.text(40),
                                "constraints": [self.text(20)]},
            "roles": roles,
            "team_level_handoff_contracts": contracts,
            "acceptance_artifacts": [self.text(20) for _ in range(self._rng.randint(0, 4))],
        }

    def audit_event(self) -> dict[str, Any]:
        """A plausible audit entry, sometimes with a malformed field."""
        event = {
            "sequence": self.integer(-5, 10000),
            "timestamp": self.text(25),
            "action": self.choice(
                ["handoff_attempt", "artifact_claimed", "human_review",
                 "release_decision", "permission_denied", self.text(10)]
            ),
            "agent_id": self.choice([self.agent_id(), None]),
            "correlation_id": self.text(10),
            "outcome": self.choice(["ok", "blocked", "allowed", "denied", self.text(8)]),
            "detail": {"artifact": self.text(15), "target": self.text(10)},
        }
        if self._rng.random() < 0.15:
            event.pop(self._rng.choice(list(event)), None)
        return event


def generate(seed: int, make: Callable[[Generator], Any], count: int) -> list[Generated]:
    """Generates `count` values from one seeded stream."""
    generator = Generator(seed)
    return [Generated(make(generator), seed) for _ in range(count)]


def replay(seed: int, make: Callable[[Generator], Any], count: int) -> list[Any]:
    """
    Regenerates a sequence from its seed.

    Used to reproduce a reported failure exactly rather than approximately.
    """
    return [make(g) for g in [Generator(seed)] for _ in range(count)]


def seeds(count: int, start: int = 0) -> Iterator[int]:
    return iter(range(start, start + count))


__all__ = ["Generated", "Generator", "generate", "replay", "seeds"]
