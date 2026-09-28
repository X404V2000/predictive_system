#!/usr/bin/env python3
"""
Trajectory Prediction Benchmark.

Replays real motion-capture clips through the prediction pipeline and reports
ADE / FDE against the static and naive baselines.

Usage:
    python3 tools/benchmark.py                          # all clips, 0.5 s horizon
    python3 tools/benchmark.py --horizon 0.25
    python3 tools/benchmark.py --subjects 007 009
    python3 tools/benchmark.py --noise 0.05             # heavier sensor noise
"""

import argparse
import sys
from pathlib import Path

# Allow running as a script from anywhere in the tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluate_dataset
from mocap import DEFAULT_DATA_ROOT, iter_clip_paths, load_clip

# Human-readable labels for the CMU subject directories used here.
SUBJECT_ACTIONS = {
    "007": "walk",
    "008": "walk/turn",
    "009": "run",
    "013": "sit",
    "020": "sit/static",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trajectory prediction.")
    parser.add_argument(
        "--horizon", type=float, default=0.5, help="prediction horizon in seconds"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="standard deviation of injected position noise, in metres",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None, help="subject directories to include"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="dataset root"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    paths = list(iter_clip_paths(args.data_root, tuple(args.subjects) if args.subjects else None))
    if not paths:
        print(f"[FAIL] No BVH clips found under {args.data_root}.")
        return 1

    print(f"[..] Loading {len(paths)} clips from {args.data_root}")
    clips = [load_clip(path) for path in paths]
    print(
        f"[ OK] {len(clips)} clips, {sum(c.frame_count for c in clips):,} frames total"
    )
    print(f"[..] Horizon {args.horizon}s, position noise sigma {args.noise} m\n")

    results = evaluate_dataset(
        clips,
        horizon=args.horizon,
        noise_std=args.noise,
        seed=args.seed,
        actions=SUBJECT_ACTIONS,
    )

    print(f"{'predictor':<12} {'ADE (m)':>10} {'FDE (m)':>10}")
    print("-" * 34)
    for name in ("static", "naive", "filtered"):
        result = results.get(name)
        if result is not None:
            print(f"{name:<12} {result.ade:>10.4f} {result.fde:>10.4f}")

    filtered = results.get("filtered")
    static = results.get("static")
    naive = results.get("naive")

    print()
    if filtered and static and naive:
        if filtered.ade < naive.ade and filtered.ade < static.ade:
            improvement = 100.0 * (1.0 - filtered.ade / naive.ade)
            print(
                f"[PASS] Filtered prediction beats both baselines "
                f"({improvement:+.1f}% vs naive)."
            )
        else:
            print(
                "[FAIL] Filtered prediction does not beat both baselines — "
                "the filter is not adding value."
            )
            return 1

    # Per-action breakdown, where the filtered predictor is most informative.
    print("\nPer-group ADE (m):")
    print(f"{'group':<14} {'static':>9} {'naive':>9} {'filtered':>9}")
    print("-" * 44)
    groups = sorted(
        set(static.per_action) if static else set(),
        key=lambda g: SUBJECT_ACTIONS.get(g, g),
    )
    for group in groups:
        label = SUBJECT_ACTIONS.get(group, group)
        values = []
        for name in ("static", "naive", "filtered"):
            result = results.get(name)
            value = result.per_action.get(group, float("nan")) if result else float("nan")
            values.append(f"{value:>9.4f}")
        print(f"{label:<14} " + " ".join(values))

    return 0


if __name__ == "__main__":
    sys.exit(main())
