from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import kmeans as reference
from submission import kmeans


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        clusters = 2 + index % 4
        features = 1 + index % 4
        points_per = 4 + index % 6
        centers = rng.normal(scale=4.0, size=(clusters, features))
        x = np.concatenate([center + rng.normal(scale=0.4, size=(points_per, features)) for center in centers]).astype(np.float64)
        cases.append({"name": f"{clusters}簇 {features}维", "args": (x, clusters, 30, 1e-8, 1000 + index)})
    return cases


def run():
    return run_cases(
        build_cases(),
        kmeans,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
