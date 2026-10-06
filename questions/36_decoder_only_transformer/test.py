from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import decoder_only_transformer as reference
from submission import decoder_only_transformer


def build_cases():
    rng = np.random.default_rng(360036); cases = []
    for index in range(100):
        heads, head_dim = 1 + index % 3, (2, 4, 6)[index % 3]; dim = heads * head_dim
        vocab, length, layers = max(5, dim + 2 + index % 7), 1 + index % 6, 1 + index % 2
        embedding = rng.normal(scale=.25, size=(vocab, dim)); blocks = []
        for _ in range(layers):
            hidden = 2 * dim + index % 4; params = {key: rng.normal(scale=.2, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}; params.update(w_gate=rng.normal(scale=.2, size=(dim, hidden)), w_up=rng.normal(scale=.2, size=(dim, hidden)), w_down=rng.normal(scale=.2, size=(hidden, dim)), norm1_scale=rng.uniform(.8, 1.2, size=dim), norm2_scale=rng.uniform(.8, 1.2, size=dim)); blocks.append(params)
        ids = rng.integers(0, vocab, size=length); final_scale = rng.uniform(.8, 1.2, size=dim)
        cases.append({"name": f"length={length}, layers={layers}, heads={heads}", "args": (ids, embedding, blocks, final_scale, heads)})
    return cases


def run():
    return run_cases(
        build_cases(),
        decoder_only_transformer,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
