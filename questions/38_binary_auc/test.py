from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import binary_auc


def build_cases():
    rng = np.random.default_rng(380038)
    cases = []
    for index in range(100):
        size = 2 + index % 63
        labels = rng.integers(0, 2, size=size, dtype=np.int64)
        labels[0], labels[1] = 0, 1
        mode = index % 5
        if mode == 0:
            scores = labels.astype(np.float64)
        elif mode == 1:
            scores = 1.0 - labels.astype(np.float64)
        elif mode == 2:
            scores = rng.integers(-3, 4, size=size).astype(np.float64)
        else:
            scores = rng.normal(size=size)
        if mode == 4:
            scores = scores * (1.0 + index % 7) + 1000.0
        cases.append({"name": f"n={size}, mode={mode}", "args": (scores, labels)})
    return cases


def reference(scores, labels):
    return float(roc_auc_score(labels, scores))


def run():
    return run_cases(
        build_cases(), binary_auc, reference,
        rtol=1e-12, atol=1e-12,
        mismatch_message="AUC 与 scikit-learn 基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
