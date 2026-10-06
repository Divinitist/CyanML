from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import encoder_decoder_block as reference
from submission import encoder_decoder_block


def build_cases():
    rng = np.random.default_rng(260026); cases = []
    for index in range(100):
        heads, head_dim = 1 + index % 3, 2 + index % 3; dim = heads * head_dim
        target_len, source_len, hidden = 1 + index % 5, 2 + index % 7, 2 * dim + index % 3
        params = {}
        for prefix in ("self", "cross"):
            for suffix in ("q", "k", "v", "o"): params[prefix + "_" + suffix] = rng.normal(scale=.2, size=(dim, dim))
        params.update(ff1=rng.normal(scale=.2, size=(dim, hidden)), ff1_bias=rng.normal(scale=.1, size=hidden), ff2=rng.normal(scale=.2, size=(hidden, dim)), ff2_bias=rng.normal(scale=.1, size=dim))
        for prefix in ("ln1", "ln2", "ln3"):
            params[prefix + "_scale"] = rng.uniform(.8, 1.2, size=dim); params[prefix + "_bias"] = rng.normal(scale=.1, size=dim)
        decoder_x, encoder_x = rng.normal(size=(target_len, dim)), rng.normal(size=(source_len, dim))
        self_mask = np.tril(np.ones((target_len, target_len), dtype=bool)); cross_mask = rng.random((target_len, source_len)) > .2; cross_mask[:, 0] = True
        cases.append({"name": f"target={target_len}, source={source_len}", "args": (decoder_x, encoder_x, params, heads, self_mask, cross_mask)})
    return cases


def run():
    return run_cases(
        build_cases(),
        encoder_decoder_block,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
