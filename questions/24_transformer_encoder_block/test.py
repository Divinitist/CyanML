from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import transformer_encoder_block as reference
from submission import transformer_encoder_block


def build_cases():
    rng = np.random.default_rng(240024)
    cases = []
    for index in range(100):
        heads, head_dim, length = 1 + index % 3, 2 + index % 3, 2 + index % 6
        dim, hidden = heads * head_dim, 2 * heads * head_dim + index % 4
        params = {key: rng.normal(scale=0.25, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}
        params.update(w1=rng.normal(scale=.25, size=(dim, hidden)), b1=rng.normal(scale=.1, size=hidden), w2=rng.normal(scale=.25, size=(hidden, dim)), b2=rng.normal(scale=.1, size=dim), ln1_scale=rng.uniform(.8, 1.2, size=dim), ln1_bias=rng.normal(scale=.1, size=dim), ln2_scale=rng.uniform(.8, 1.2, size=dim), ln2_bias=rng.normal(scale=.1, size=dim))
        x = rng.normal(size=(length, dim)); mask = None if index % 2 else (rng.random((length, length)) > .25); 
        if mask is not None: np.fill_diagonal(mask, True)
        cases.append({"name": f"pre={bool(index % 2)}, heads={heads}", "args": (x, params, heads, bool(index % 2), mask)})
    return cases


def run():
    return run_cases(
        build_cases(),
        transformer_encoder_block,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
