from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import embedding_lm_head as reference
from submission import embedding_lm_head


def build_cases():
    rng = np.random.default_rng(340034); cases = []
    for index in range(100):
        vocab, dim = 3 + index % 17, 2 + index % 8
        shape = (1 + index % 8,) if index % 3 else (2, 1 + index % 5)
        ids = rng.integers(0, vocab, size=shape); embedding = rng.normal(scale=.3, size=(vocab, dim)); projection = rng.normal(scale=.3, size=(dim, dim)); head = rng.normal(scale=.3, size=(vocab, dim)); bias = None if index % 4 == 0 else rng.normal(scale=.1, size=vocab); tied = bool(index % 2)
        cases.append({"name": f"shape={shape}, tied={tied}, bias={bias is not None}", "args": (ids, embedding, projection, head, bias, tied)})
    return cases


def run():
    return run_cases(
        build_cases(),
        embedding_lm_head,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
