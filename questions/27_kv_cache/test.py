from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import kv_cache_attention as reference
from submission import kv_cache_attention


def build_cases():
    rng = np.random.default_rng(270027); cases = []
    for index in range(100):
        q_heads = (1, 2, 4, 8)[index % 4]; kv_heads = (1, q_heads)[index % 2]
        head_dim, chunk, past = 2 + index % 3, 1 + index % 4, index % 6
        dim = q_heads * head_dim; x = rng.normal(size=(chunk, dim))
        w_q = rng.normal(scale=.25, size=(dim, dim)); w_k = rng.normal(scale=.25, size=(dim, kv_heads * head_dim)); w_v = rng.normal(scale=.25, size=(dim, kv_heads * head_dim)); w_o = rng.normal(scale=.25, size=(dim, dim))
        cache_k = None if past == 0 else rng.normal(size=(kv_heads, past, head_dim)); cache_v = None if past == 0 else rng.normal(size=(kv_heads, past, head_dim))
        cases.append({"name": f"q={q_heads}, kv={kv_heads}, past={past}, chunk={chunk}", "args": (x, w_q, w_k, w_v, w_o, q_heads, kv_heads, cache_k, cache_v)})
    return cases


def run():
    return run_cases(
        build_cases(),
        kv_cache_attention,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
