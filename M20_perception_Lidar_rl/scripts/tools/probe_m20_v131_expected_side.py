#!/usr/bin/env python3
"""Offline expected-side observability probe; never modifies the actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("inputs", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def main() -> None:
    arrays = [np.load(path) for path in args.inputs]
    x = np.concatenate([item["X"] for item in arrays])
    y = np.concatenate([item["y"] for item in arrays])
    groups = np.concatenate([item["groups"] for item in arrays])
    # Conservatively discard the first captured frame of every episode group.
    # This also makes older captures safe against a lazy-reset observation/state
    # boundary; new captures already exclude previous_dones at collection time.
    _, first = np.unique(groups, return_index=True)
    keep = np.ones(len(groups), dtype=bool); keep[first] = False
    x, y, groups = x[keep], y[keep], groups[keep]
    if x.ndim != 2 or x.shape[1] != 63:
        raise RuntimeError(f"Probe input must be actor_obs_63, got {x.shape}")
    classes, counts = np.unique(y, return_counts=True)
    if classes.tolist() != [0, 1] or counts.min() < 20:
        raise RuntimeError(f"Both expected-side classes need >=20 samples, got {dict(zip(classes, counts))}")
    split = None
    for offset in range(100):
        candidate = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                           random_state=args.seed + offset).split(x, y, groups))
        if len(np.unique(y[candidate[0]])) == 2 and len(np.unique(y[candidate[1]])) == 2:
            split = candidate; break
    if split is None:
        raise RuntimeError("Could not form a two-class episode-group held-out split")
    train_ids, test_ids = split
    x_train, x_test, y_train, y_test = x[train_ids], x[test_ids], y[train_ids], y[test_ids]
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed),
        ),
        "small_mlp": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", max_iter=2000,
                                            early_stopping=True, random_state=args.seed),
        ),
    }
    result = {
        "samples": int(len(y)), "train_samples": int(len(y_train)), "test_samples": int(len(y_test)),
        "class_counts": {str(int(k)): int(v) for k, v in zip(classes, counts)}, "actor_obs_dim": 63,
        "split": "episode_group_held_out", "train_groups": int(len(np.unique(groups[train_ids]))),
        "test_groups": int(len(np.unique(groups[test_ids]))),
        "models": {},
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        result["models"][name] = {
            "held_out_accuracy": float(accuracy_score(y_test, prediction)),
            "held_out_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("V131_EXPECTED_SIDE_PROBE=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
