from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import moe_route as reference
from submission import moe_route


def build_cases():
    rng = np.random.default_rng(300030); cases = []
    for index in range(100):
        tokens, dim, experts = 2 + index % 9, 2 + index % 6, 2 + index % 5
        hidden, top_k = 3 + index % 7, 1 + index % min(2, experts)
        x = rng.normal(size=(tokens, dim)); gate = rng.normal(scale=.3, size=(dim, experts)); w1 = rng.normal(scale=.3, size=(experts, dim, hidden)); w2 = rng.normal(scale=.3, size=(experts, hidden, dim)); capacity = (.5, 1., 1.5, 2.)[index % 4]
        cases.append({"name": f"experts={experts}, k={top_k}, capacity={capacity}", "args": (x, gate, w1, w2, top_k, capacity)})
    return cases


def run():
    return run_cases(
        build_cases(),
        moe_route,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
