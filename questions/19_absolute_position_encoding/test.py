from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import add_sinusoidal_position_encoding as reference
from submission import add_sinusoidal_position_encoding


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length = 1 + index % 12
        model_dim = 1 + index % 13
        x = rng.normal(size=(length, model_dim)).astype(np.float64)
        start = (index * 7) % 101
        base = (100.0, 1000.0, 10000.0)[index % 3]
        cases.append({"name": f"shape={x.shape}, start={start}", "args": (x, start, base)})
    return cases


def run():
    return run_cases(
        build_cases(),
        add_sinusoidal_position_encoding,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
