from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import grouped_query_attention as reference
from submission import grouped_query_attention


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length = 2 + index % 7
        query_heads = (1, 2, 4, 8)[index % 4]
        kv_heads = (1, query_heads)[index % 2]
        head_dim = 2 + index % 3
        model_dim = query_heads * head_dim
        x = rng.normal(size=(length, model_dim))
        w_q = rng.normal(scale=0.3, size=(model_dim, model_dim))
        w_k = rng.normal(scale=0.3, size=(model_dim, kv_heads * head_dim))
        w_v = rng.normal(scale=0.3, size=(model_dim, kv_heads * head_dim))
        w_o = rng.normal(scale=0.3, size=(model_dim, model_dim))
        mask = np.tril(np.ones((length, length), dtype=bool)) if index % 2 else None
        cases.append({"name": f"q={query_heads}, kv={kv_heads}", "args": (x, w_q, w_k, w_v, w_o, query_heads, kv_heads, mask)})
    return cases


def run():
    return run_cases(
        build_cases(),
        grouped_query_attention,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
