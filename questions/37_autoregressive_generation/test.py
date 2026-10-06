from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import autoregressive_generate as reference
from submission import autoregressive_generate


def build_cases():
    rng = np.random.default_rng(370037); cases = []
    shared = []
    for pair in range(50):
        heads, head_dim = 1 + pair % 3, 2 + pair % 3; dim = heads * head_dim; vocab = max(5, dim + 3 + pair % 5); prompt_length = 1 + pair % 5
        embedding = rng.normal(scale=.3, size=(vocab, dim)); params = {key: rng.normal(scale=.2, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}; prompt = rng.integers(0, vocab, size=prompt_length); steps, end = 1 + pair % 5, pair % vocab
        shared.append((prompt, embedding, params, heads, steps, end))
    for index in range(100):
        common = shared[index // 2]; cached = bool(index % 2)
        cases.append({"name": f"pair={index // 2}, cache={cached}", "args": (*common, cached)})
    return cases


def run():
    return run_cases(
        build_cases(),
        autoregressive_generate,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
