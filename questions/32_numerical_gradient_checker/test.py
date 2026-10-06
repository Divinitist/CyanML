from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import check_gradient as reference
from submission import check_gradient


def build_cases():
    rng = np.random.default_rng(320032); cases = []
    for index in range(100):
        shape = (1 + index % 7,) if index % 3 else (2 + index % 3, 1 + index % 4)
        x = rng.normal(scale=.5, size=shape); mode = index % 5
        if mode == 0: function = lambda value: np.sum(value ** 2); gradient = 2 * x
        elif mode == 1: function = lambda value: np.sum(value ** 3); gradient = 3 * x ** 2
        elif mode == 2: function = lambda value: np.sum(np.sin(value)); gradient = np.cos(x)
        elif mode == 3: function = lambda value: np.sum(np.exp(value)); gradient = np.exp(x)
        else:
            coefficient = rng.normal(size=shape); function = lambda value, c=coefficient: np.sum(c * value); gradient = coefficient.copy()
        if index % 10 == 0: gradient = gradient + 1e-3
        epsilon, tolerance = (1e-5, 1e-6) if index % 2 else (1e-6, 1e-5)
        cases.append({"name": f"mode={mode}, shape={shape}", "args": (function, x, gradient, epsilon, tolerance), "input": {"mode": mode, "shape": shape, "epsilon": epsilon, "tolerance": tolerance}})
    return cases


def run():
    return run_cases(
        build_cases(),
        check_gradient,
        reference,
        rtol=1e-9,
        atol=1e-11,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
