from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import multi_query_attention as reference
from submission import multi_query_attention


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length = 2 + index % 7
        heads = 1 + index % 4
        head_dim = 2 + index % 3
        model_dim = heads * head_dim
        x = rng.normal(size=(length, model_dim)).astype(np.float64)
        w_q = rng.normal(scale=0.3, size=(model_dim, model_dim))
        w_k = rng.normal(scale=0.3, size=(model_dim, head_dim))
        w_v = rng.normal(scale=0.3, size=(model_dim, head_dim))
        w_o = rng.normal(scale=0.3, size=(model_dim, model_dim))
        mask = np.tril(np.ones((length, length), dtype=bool)) if index % 3 else None
        cases.append({"name": f"query_heads={heads}", "args": (x, w_q, w_k, w_v, w_o, heads, mask)})
    return cases


def run():
    return run_cases(
        build_cases(),
        multi_query_attention,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
