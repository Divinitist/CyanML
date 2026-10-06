from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import sliding_window_attention as reference
from submission import sliding_window_attention


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length = 2 + index % 7
        heads = 1 + index % 4
        head_dim = 2 + index % 3
        model_dim = heads * head_dim
        x = rng.normal(size=(length, model_dim)).astype(np.float64)
        weights = [rng.normal(scale=0.3, size=(model_dim, model_dim)) for _ in range(4)]
        window = 1 + index % (length + 2)
        cases.append({"name": f"length={length}, window={window}", "args": (x, *weights, heads, window)})
    return cases


def run():
    return run_cases(
        build_cases(),
        sliding_window_attention,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
