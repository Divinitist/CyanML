from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import cross_attention as reference
from submission import cross_attention


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        q_len, kv_len = 1 + index % 6, 2 + index % 8
        heads, head_dim = 1 + index % 4, 2 + index % 3
        model_dim = heads * head_dim
        query = rng.normal(size=(q_len, model_dim))
        context = rng.normal(size=(kv_len, model_dim))
        weights = [rng.normal(scale=0.3, size=(model_dim, model_dim)) for _ in range(4)]
        mask = rng.random((q_len, kv_len)) > 0.25
        mask[:, 0] = True
        if index % 2 == 0:
            mask = None
        cases.append({"name": f"{q_len}x{kv_len}, heads={heads}", "args": (query, context, *weights, heads, mask)})
    return cases


def run():
    return run_cases(
        build_cases(),
        cross_attention,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
