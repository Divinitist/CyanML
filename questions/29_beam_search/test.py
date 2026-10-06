from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import beam_search as reference
from submission import beam_search


def build_cases():
    rng = np.random.default_rng(290029); cases = []
    for index in range(100):
        vocab = 3 + index % 8; raw = rng.normal(size=(vocab, vocab)); raw -= np.log(np.exp(raw).sum(axis=1, keepdims=True))
        start, end = index % vocab, (index * 3 + 1) % vocab
        width, steps, penalty = 1 + index % min(5, vocab), 1 + index % 7, (0., .6, 1., 1.4)[index % 4]
        cases.append({"name": f"vocab={vocab}, width={width}, alpha={penalty}", "args": (raw, start, end, steps, width, penalty)})
    return cases


def run():
    return run_cases(
        build_cases(),
        beam_search,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
