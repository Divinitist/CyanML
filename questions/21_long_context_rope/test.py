from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import apply_scaled_rope as reference
from submission import apply_scaled_rope


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length, heads = 1 + index % 8, 1 + index % 3
        head_dim = (4, 6, 8, 10)[index % 4]
        q = rng.normal(size=(length, heads, head_dim))
        k = rng.normal(size=(length, heads, head_dim))
        positions = np.arange(length, dtype=np.float64) + index * 13
        scale = (1.25, 2.0, 4.0, 8.0)[index % 4]
        method = "linear" if index % 2 == 0 else "ntk"
        cases.append({"name": f"{method}, scale={scale}", "args": (q, k, positions, scale, method, 10000.0)})
    return cases


def run():
    return run_cases(
        build_cases(),
        apply_scaled_rope,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
