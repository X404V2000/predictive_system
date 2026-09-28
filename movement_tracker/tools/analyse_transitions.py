#!/usr/bin/env python3
"""
Action Transition Analysis.

Builds a first-order Markov model over action sequences and reports what it
learned. Also cross-validates it, because a transition matrix fitted on a
handful of clips can look confident while being meaningless.

Usage:
    python3 tools/analyse_transitions.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from actions import GaussianActionModel, coarse_action, load_labelled_features
from transitions import (
    END_ACTION,
    START_ACTION,
    TransitionModel,
    collect_sequences,
)


def main() -> int:
    samples, subjects = load_labelled_features()
    model = GaussianActionModel.fit(samples)

    print("[..] segmenting sequence clips (this re-reads motion capture) ...")
    sequences = collect_sequences(model)
    print(f"[ OK] {len(sequences)} sequence clips segmented\n")

    if not sequences:
        print("[FAIL] no sequence clips found")
        return 1

    print("segmented sequences (window = 2.0 s):")
    for seq in sequences[:12]:
        joined = " -> ".join(seq.actions)
        mean_conf = (
            float(np.mean(seq.confidences)) if seq.confidences else float("nan")
        )
        print(f"  {seq.clip_id:<12} conf {mean_conf:.2f}  [{joined}]")
    print(f"  ... and {max(0, len(sequences) - 12)} more\n")

    # How often is a window undecided? A high rate means the transitions are
    # weakly supported and the matrix should be read with caution.
    all_windows = [w for s in sequences for w in s.actions]
    uncertain = sum(1 for w in all_windows if w == "uncertain")
    print(
        f"windows classified: {len(all_windows)}   "
        f"undecided: {uncertain} ({100 * uncertain / len(all_windows):.1f}%)\n"
    )

    transition = TransitionModel.fit(sequences)
    states = list(transition.states)

    print("transition counts (rows = from, cols = to):")
    width = max(len(s) for s in states) + 2
    print(" " * width + "".join(f"{s[:9]:>11}" for s in states))
    for i, name in enumerate(states):
        row = "".join(f"{int(v):>11}" for v in transition.counts[i])
        print(f"{name:<{width}}{row}")

    print()
    print("most likely next action, and its probability:")
    for state in states:
        if state == END_ACTION:
            continue
        nxt, probability = transition.most_likely_next(state)
        print(f"  {state:<12} -> {nxt:<12} p={probability:.2f}")

    # --- honest check: is the matrix better than a baseline? -------------
    #
    # All three scores are computed from the SAME per-fold test pairs and then
    # averaged over folds. Mixing pooled and per-fold aggregation would compare
    # unlike quantities and produce a meaningless gap.
    print()
    print("hold-one-subject-out check (predict the next window's action):")

    fold_reports: list[tuple[str, int, float, float, float]] = []

    for held_out in sorted({s.clip_id.split("/")[0] for s in sequences}):
        train = [s for s in sequences if s.clip_id.split("/")[0] != held_out]
        test = [s for s in sequences if s.clip_id.split("/")[0] == held_out]
        if not train or not test:
            continue

        fold_model = TransitionModel.fit(train)

        # Pairs the MODEL is scored on. Only the test subject is scored, and
        # only where both windows carry a decision.
        pairs = [
            (a, b)
            for s in test
            for a, b in zip(s.actions, s.actions[1:])
            if a != "uncertain" and b != "uncertain"
        ]
        if not pairs:
            continue

        model_score = float(
            np.mean([b == fold_model.most_likely_next(a)[0] for a, b in pairs])
        )

        # Baseline 1: always predict the most common next action, learned from
        # the TRAINING folds so the baseline sees no test information.
        train_pairs = [
            (a, b) for s in train for a, b in zip(s.actions, s.actions[1:])
        ]
        common = Counter(b for _, b in train_pairs).most_common(1)[0][0]
        global_score = float(np.mean([b == common for a, b in pairs]))

        # Baseline 2: the majority successor of the current action, again learned
        # from the training folds. This baseline is allowed the same information
        # as the model, so beating it is the meaningful test of learned structure.
        successor: dict[str, str] = {}
        by_current_train: dict[str, list[str]] = {}
        for a, b in train_pairs:
            by_current_train.setdefault(a, []).append(b)
        for state, followings in by_current_train.items():
            successor[state] = Counter(followings).most_common(1)[0][0]
        per_state_score = float(
            np.mean([b == successor.get(a, common) for a, b in pairs])
        )

        fold_reports.append(
            (held_out, len(pairs), model_score, global_score, per_state_score)
        )

    if not fold_reports:
        print("  no evaluable folds")
        return 1

    print(f"  {'held-out':<10}{'n':>6}{'model':>10}{'common':>10}{'per-state':>11}")
    for held_out, n, m, g, ps in fold_reports:
        print(f"  {held_out:<10}{n:>6}{m:>9.1%}{g:>10.1%}{ps:>11.1%}")

    model_score = float(np.mean([r[2] for r in fold_reports]))
    global_score = float(np.mean([r[3] for r in fold_reports]))
    per_state_score = float(np.mean([r[4] for r in fold_reports]))

    print()
    print(f"  mean over {len(fold_reports)} folds:")
    print(f"    transition model            {model_score:>7.1%}")
    print(f"    always predict one class    {global_score:>7.1%}   "
          f"lift {model_score - global_score:+.1f} pts")
    print(f"    per-state majority          {per_state_score:>7.1%}   "
          f"lift {model_score - per_state_score:+.1f} pts")
    print()
    print("  Beating the per-state baseline means real transition structure was")
    print("  learned. Beating the constant guess is what makes it actionable; the")
    print("  model currently does not, because one class dominates the data.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
