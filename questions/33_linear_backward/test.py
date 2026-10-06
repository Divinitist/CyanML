from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import linear_forward_backward as reference
from submission import linear_forward_backward


def build_cases():
    rng = np.random.default_rng(330033); cases = []
    for index in range(100):
        batch, inputs, outputs = 1 + index % 7, 1 + (index * 2) % 8, 1 + (index * 3) % 9
        scale = (0.01, .1, 1., 3.)[index % 4]
        x = rng.normal(scale=scale, size=(batch, inputs)); weight = rng.normal(scale=scale, size=(inputs, outputs)); bias = rng.normal(scale=scale, size=outputs); grad = rng.normal(size=(batch, outputs))
        cases.append({"name": f"{batch}x{inputs}->{outputs}, scale={scale}", "args": (x, weight, bias, grad)})
    return cases


def run():
    return run_cases(
        build_cases(),
        linear_forward_backward,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
