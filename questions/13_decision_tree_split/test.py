from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import best_decision_tree_split as reference
from submission import best_decision_tree_split


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        samples = 8 + index % 15
        features = 1 + index % 6
        classes = 2 + index % 3
        x = rng.integers(-4, 7, size=(samples, features)).astype(np.float64)
        labels = ((x[:, index % features] > (index % 3 - 1)).astype(int) + rng.integers(0, classes, size=samples) * (index % 4 == 0)) % classes
        criterion = "gini" if index % 2 == 0 else "entropy"
        cases.append({"name": f"{criterion} {classes}类", "args": (x, labels, criterion)})
    return cases


def run():
    return run_cases(
        build_cases(),
        best_decision_tree_split,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
