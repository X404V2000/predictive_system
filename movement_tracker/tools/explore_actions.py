"""Explore feature separability to choose a defensible action vocabulary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from features import FEATURE_NAMES, extract_features
from mocap import iter_clip_paths, load_clip

LABELS = Path("training_data/cmu_mocap/action_labels.tsv")


def load_labelled():
    mapping = {}
    for line in LABELS.read_text().splitlines():
        if not line.strip():
            continue
        code, desc = line.split("\t", 1)
        mapping[code.strip()] = desc.strip()
    out = []
    for path in iter_clip_paths():
        if path.stem in mapping:
            out.append(extract_features(load_clip(path), mapping[path.stem]))
    return out


COARSE = [
    ("locomote", ("walk", "jog", "run")),
    ("seated", ("sit",)),
    ("climb", ("climb",)),
    ("jump", ("jump",)),
    ("gesture", ("wave", "traffic")),
    ("manipulate", ("wash", "sweep", "mop", "drink")),
    ("exercise", ("box", "stretch", "bend", "laugh")),
]


def coarse(desc: str) -> str:
    low = desc.lower()
    for name, keys in COARSE:
        if any(k in low for k in keys):
            return name
    return "other"


clips = load_labelled()
print(f"clips with labels: {len(clips)}")
groups = {}
for c in clips:
    groups.setdefault(coarse(c.label), []).append(c)

print(f"\ncoarse classes and clip counts:")
for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    subs = sorted({i.subject for i in items})
    print(f"  {name:<11} {len(items):>3} clips  subjects={subs}")

# Which features actually separate the classes? Measure spread between class
# means relative to within-class spread (a simple discriminability score).
print(f"\nfeature discriminability (between-class spread / within-class spread):")
names = [g for g in groups if g != "other"]
rows = []
for fi, fname in enumerate(FEATURE_NAMES):
    class_means = []
    within = []
    for name in names:
        vals = np.array([c.values[fi] for c in groups[name]])
        class_means.append(vals.mean())
        within.append(vals.std())
    between = np.std(class_means)
    avg_within = np.mean(within) + 1e-9
    rows.append((between / avg_within, fname, between, avg_within))

for score, fname, b, w in sorted(rows, reverse=True):
    print(f"  {fname:<20} {score:6.2f}   (between {b:8.3f}, within {w:8.3f})")
