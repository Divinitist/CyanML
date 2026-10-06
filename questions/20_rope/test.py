from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import apply_rope as reference
from submission import apply_rope


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length, heads = 1 + index % 9, 1 + index % 4
        head_dim = (2, 4, 6, 8)[index % 4]
        rotary_dim = (2 * (1 + index % (head_dim // 2)))
        q = rng.normal(size=(length, heads, head_dim))
        k = rng.normal(size=(length, heads, head_dim))
        positions = np.arange(length, dtype=np.float64) + index % 11
        cases.append({"name": f"dim={head_dim}, rotary={rotary_dim}", "args": (q, k, positions, 10000.0, rotary_dim)})
    return cases


def run():
    return run_cases(
        build_cases(),
        apply_rope,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
