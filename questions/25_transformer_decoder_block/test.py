from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import transformer_decoder_block as reference
from submission import transformer_decoder_block


def build_cases():
    rng = np.random.default_rng(250025); cases = []
    for index in range(100):
        heads, head_dim, length = 1 + index % 3, (2, 4, 6)[index % 3], 1 + index % 7
        dim, hidden = heads * head_dim, 2 * heads * head_dim + index % 5
        params = {key: rng.normal(scale=.25, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}
        params.update(w_gate=rng.normal(scale=.25, size=(dim, hidden)), w_up=rng.normal(scale=.25, size=(dim, hidden)), w_down=rng.normal(scale=.25, size=(hidden, dim)), norm1_scale=rng.uniform(.8, 1.2, size=dim), norm2_scale=rng.uniform(.8, 1.2, size=dim))
        x = rng.normal(size=(length, dim)); positions = np.arange(length) + index % 17
        cases.append({"name": f"heads={heads}, start={positions[0]}", "args": (x, params, heads, positions)})
    return cases


def run():
    return run_cases(
        build_cases(),
        transformer_decoder_block,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
