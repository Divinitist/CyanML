from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import gmm_em as reference
from submission import gmm_em


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        components = 2 + index % 3
        features = 1 + index % 3
        points = 12 + index % 8
        anchors = rng.normal(scale=3.0, size=(components, features))
        x = np.vstack([anchors[i % components] + rng.normal(scale=0.5, size=features) for i in range(points)]).astype(np.float64)
        cases.append({"name": f"{components}分量 {features}维", "args": (x, components, 12, 1e-7, 1e-5, 2000 + index)})
    return cases


def run():
    return run_cases(
        build_cases(),
        gmm_em,
        reference,
        rtol=1e-7,
        atol=1e-9,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
