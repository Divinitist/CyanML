from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import knn_predict as reference
from submission import knn_predict


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        samples = 5 + index % 16
        features = 1 + index % 6
        queries = 1 + index % 7
        train_x = rng.normal(size=(samples, features)).astype(np.float64)
        query_x = rng.normal(size=(queries, features)).astype(np.float64)
        task = "classification" if index % 2 == 0 else "regression"
        train_y = rng.integers(0, 4, size=samples) if task == "classification" else rng.normal(size=samples)
        k = 1 + index % min(7, samples)
        cases.append({"name": f"{task} k={k}", "args": (train_x, train_y, query_x, k, task)})
    return cases


def run():
    return run_cases(
        build_cases(),
        knn_predict,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
