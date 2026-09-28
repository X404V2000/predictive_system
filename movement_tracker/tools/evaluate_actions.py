#!/usr/bin/env python3
"""
Leave-One-Subject-Out Evaluation of the Action Classifier.

Class and subject are confounded in this dataset: most action classes appear
only in subjects 013 and 014, while locomotion spans five subjects. A random
clip split would therefore leak subject identity into the test set and report a
flattering number.

Holding out entire SUBJECTS is the honest protocol: the classifier must
generalise to a person it has never seen. It is also the harder test, and the
result should be read with that in mind.

Usage:
    python3 tools/evaluate_actions.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from actions import GaussianActionModel, coarse_action, load_labelled_features


def confusion_matrix(truth: list[str], predicted: list[str], classes: list[str]):
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(truth, predicted):
        matrix[index[t], index[p]] += 1
    return matrix


def main() -> int:
    samples, subjects = load_labelled_features()
    unique_subjects = sorted(set(subjects))

    print(f"clips: {len(samples)}   subjects: {unique_subjects}")
    print()

    label_counts = Counter(coarse_action(s.label) for s in samples)
    print("class distribution:")
    for name, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        held_out = sorted({sub for s, sub in zip(samples, subjects)
                           if coarse_action(s.label) == name})
        print(f"  {name:<11} {count:>3} clips   present in subjects {held_out}")
    print()

    # --- leave-one-subject-out ------------------------------------------
    truths: list[str] = []
    predictions: list[str] = []
    confidences: list[float] = []
    per_subject = {}

    for held_out in unique_subjects:
        train = [s for s, sub in zip(samples, subjects) if sub != held_out]
        test = [(s, sub) for s, sub in zip(samples, subjects) if sub == held_out]
        if not test:
            continue

        train_classes = {coarse_action(s.label) for s in train}
        test = [(s, sub) for s, sub in test if coarse_action(s.label) in train_classes]
        if not test:
            # The held-out subject only exhibits classes absent from training, so
            # there is nothing this fold can fairly evaluate.
            per_subject[held_out] = None
            print(f"  subject {held_out}: skipped (no classes in common with training)")
            continue

        # Split the TRAINING subjects into a parameter set and a calibration set,
        # so temperature is never fitted on the data used to measure it. Fitting
        # on the held-out subject would leak and report a flattering calibration.
        train_subjects = sorted({sub for s, sub in zip(samples, subjects) if sub != held_out})
        model = GaussianActionModel.fit(train)

        if len(train_subjects) >= 2:
            cal_subject = train_subjects[-1]
            cal_set = [
                s
                for s, sub in zip(samples, subjects)
                if sub == cal_subject
                and coarse_action(s.label) in {coarse_action(x.label) for x in train}
            ]
            if cal_set:
                model.fit_temperature(cal_set)

        correct = 0
        for s, _ in test:
            truth = coarse_action(s.label)
            predicted, confidence = model.predict(s.values)
            truths.append(truth)
            predictions.append(predicted)
            confidences.append(confidence)
            correct += int(predicted == truth)
        accuracy = correct / len(test)
        per_subject[held_out] = accuracy
        print(f"  subject {held_out}: {correct}/{len(test)} correct  ({accuracy:.1%})")

    if not truths:
        print("[FAIL] no evaluable folds")
        return 1

    classes = sorted(set(truths) | set(predictions))
    overall = float(np.mean(np.array(truths) == np.array(predictions)))
    chance = max(Counter(truths).values()) / len(truths)

    print()
    print(f"overall accuracy : {overall:.1%}  (n={len(truths)})")
    print(f"majority baseline: {chance:.1%}")
    print(f"mean confidence  : {np.mean(confidences):.3f}")
    print()

    matrix = confusion_matrix(truths, predictions, classes)
    width = max(len(c) for c in classes) + 1
    print("confusion matrix (rows = truth, cols = predicted):")
    print(" " * width + "".join(f"{c[:8]:>10}" for c in classes))
    for i, name in enumerate(classes):
        row = "".join(f"{v:>10}" for v in matrix[i])
        print(f"{name:<{width}}{row}")

    # Per-class recall, computed only over classes actually present in truth.
    print()
    print("per-class recall:")
    for i, name in enumerate(classes):
        total = matrix[i].sum()
        if total:
            print(f"  {name:<11} {matrix[i, i] / total:6.1%}  (n={total})")

    evaluable = [v for v in per_subject.values() if v is not None]
    print()
    print(f"folds evaluated: {len(evaluable)} of {len(unique_subjects)}")
    if evaluable:
        print(f"accuracy per fold: min {min(evaluable):.1%} max {max(evaluable):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
