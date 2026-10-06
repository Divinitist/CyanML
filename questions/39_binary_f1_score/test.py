from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import binary_f1_score


def build_cases():
    rng = np.random.default_rng(390039)
    cases = []
    for index in range(100):
        size = 1 + index % 80
        targets = rng.integers(0, 2, size=size, dtype=np.int64)
        mode = index % 6
        if mode == 0:
            predictions = targets.copy()
        elif mode == 1:
            predictions = 1 - targets
        elif mode == 2:
            predictions = np.zeros(size, dtype=np.int64)
        elif mode == 3:
            predictions = np.ones(size, dtype=np.int64)
        else:
            predictions = rng.integers(0, 2, size=size, dtype=np.int64)
        if mode == 5:
            targets.fill(0)
        cases.append({"name": f"n={size}, mode={mode}", "args": (predictions, targets)})
    return cases


def reference(predictions, targets):
    return float(f1_score(targets, predictions, zero_division=0))


def run():
    return run_cases(
        build_cases(), binary_f1_score, reference,
        rtol=0.0, atol=0.0,
        mismatch_message="F1-score 与 scikit-learn 基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
