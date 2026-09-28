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

    # --- honest check: is the matrix better than chance? -----------------
    print()
    print("hold-one-subject-out check (predict the next window's action):")
    correct = 0
    total = 0
    for seq in sequences:
        subject = seq.clip_id.split("/")[0]
        others = [
            s for s in sequences if s.clip_id.split("/")[0] != subject
        ]
        if not others:
            continue
        fold_model = TransitionModel.fit(others)
        for current, following in zip(seq.actions, seq.actions[1:]):
            if current == "uncertain" or following == "uncertain":
                continue
            predicted, _ = fold_model.most_likely_next(current)
            total += 1
            correct += int(predicted == following)

    if total == 0:
        print("  no evaluable adjacent pairs")
    else:
        print(f"  next-action accuracy: {correct}/{total} = {correct / total:.1%}")
        print(
            "  NOTE: with this few sequence clips the estimate is noisy, and the "
            "action\n        vocabulary is small, so treat it as indicative only."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
