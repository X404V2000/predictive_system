#!/usr/bin/env python3
"""
Diagnose why the action transition model shows no lift over a constant predictor.

Tests each candidate cause independently rather than assuming the window duration
is to blame. Answers, in order:

  1. How skewed is the window classification stream, really?
  2. Does window duration change the skew?
  3. How often does a clip's window sequence match its own label?
  4. Is the transition matrix actually informative, or is every row the majority?
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from actions import DEFAULT_LABELS_PATH, GaussianActionModel, load_labelled_features
from features import extract_features
from mocap import iter_clip_paths, load_clip
import transitions as T


def segment_all(model, window_seconds, min_conf=0.0):
    """Segments every sequence clip at a given window duration."""
    descriptions = {}
    for line in DEFAULT_LABELS_PATH.read_text().splitlines():
        if line.strip():
            code, desc = line.split("\t", 1)
            descriptions[code.strip()] = desc.strip()

    out = []
    for path in iter_clip_paths():
        desc = descriptions.get(path.stem)
        if desc is None or "," not in desc:
            continue
        clip = load_clip(path)
        dt = clip.sample_dt
        frames = max(1, int(round(window_seconds / dt)))
        actions, confs = [], []
        for start in range(0, clip.frame_count, frames):
            end = min(start + frames, clip.frame_count)
            if end - start < frames // 2:
                break
            sub = T._subclip(clip, start, end)
            feats = extract_features(sub, label=desc)
            predicted, conf = model.predict(feats.values)
            actions.append(predicted)
            confs.append(conf)
        out.append((path.stem, desc, actions, confs))
    return out


def balanced_score(sequences, transition):
    by_current = {}
    for actions in sequences:
        for a, b in zip(actions, actions[1:]):
            by_current.setdefault(a, []).append(b)

    model_hits = baseline_hits = 0.0
    considered = 0
    for current, followings in by_current.items():
        if len(followings) < 3:
            continue
        pred, _ = transition.most_likely_next(current)
        model_hits += sum(1 for f in followings if f == pred) / len(followings)
        majority = Counter(followings).most_common(1)[0]
        baseline_hits += majority[1] / len(followings)
        considered += 1
    if considered == 0:
        return None, None, 0
    return model_hits / considered, baseline_hits / considered, considered


def main() -> int:
    samples, _ = load_labelled_features()
    model = GaussianActionModel.fit(samples)

    print("=" * 70)
    print("1. WINDOW DURATION SWEEP")
    print("=" * 70)
    print(f"{'window':>8} {'windows':>8} {'top class':>12} {'top share':>10} {'balanced':>10} {'baseline':>10} {'lift':>8}")

    for window in (0.5, 1.0, 2.0, 3.0, 4.0):
        segs = segment_all(model, window)
        all_actions = [a for _, _, acts, _ in segs for a in acts]
        top, count = Counter(all_actions).most_common(1)[0]
        share = count / len(all_actions)

        seq_objs = [
            T.SegmentedClip(cid, desc, tuple(acts), tuple(confs))
            for cid, desc, acts, confs in segs
        ]
        transition = T.TransitionModel.fit(seq_objs)
        sequences = [list(s.actions) for s in seq_objs]
        bal, base, n = balanced_score(sequences, transition)

        if bal is None:
            print(f"{window:>8.1f} {len(all_actions):>8} {top:>12} {share:>10.1%}    (too few states)")
            continue
        print(
            f"{window:>8.1f} {len(all_actions):>8} {top:>12} {share:>10.1%} "
            f"{bal:>10.1%} {base:>10.1%} {bal - base:>+8.1%}"
        )

    print()
    print("=" * 70)
    print("2. DOES A CLIP'S WINDOW SEQUENCE MATCH ITS OWN LABEL?")
    print("=" * 70)
    segs = segment_all(model, 1.0)
    for cid, desc, acts, confs in segs[:15]:
        counts = Counter(acts)
        summary = " ".join(f"{k}:{v}" for k, v in counts.most_common())
        print(f"  {cid:<10} {desc[:38]:<38} -> {summary}")

    print()
    print("=" * 70)
    print("3. IS THE TRANSITION MATRIX INFORMATIVE?")
    print("=" * 70)
    print("For each state, how concentrated is the outgoing distribution?")
    print("A row identical to the global prior carries no transition information.")
    seq_objs = [
        T.SegmentedClip(cid, desc, tuple(acts), tuple(confs))
        for cid, desc, acts, confs in segs
    ]
    transition = T.TransitionModel.fit(seq_objs)
    global_prior = Counter(b for s in seq_objs for b in s.actions)
    total = sum(global_prior.values())
    prior = {k: v / total for k, v in global_prior.items()}

    for state in transition.states:
        dist = transition.next_action_distribution(state)
        # Total variation distance from the global prior: 0 means no information.
        keys = set(dist) | set(prior)
        tv = 0.5 * sum(abs(dist.get(k, 0) - prior.get(k, 0)) for k in keys)
        print(f"  {state:<12} TV distance from prior = {tv:.3f}")
    print()
    print("  (TV near 0 means the row tells you nothing beyond the base rate)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
