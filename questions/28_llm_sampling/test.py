from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import sample_next_token as reference
from submission import sample_next_token


def build_cases():
    rng = np.random.default_rng(280028); cases = []
    for index in range(100):
        vocab = 2 + index % 19; logits = rng.normal(scale=1 + index % 4, size=vocab)
        temperature = (.5, .8, 1., 1.5, 2.)[index % 5]; top_k = (0, 1, min(3, vocab), min(7, vocab))[index % 4]; top_p = (.35, .7, .9, 1.)[index % 4]
        cases.append({"name": f"vocab={vocab}, k={top_k}, p={top_p}", "args": (logits, temperature, top_k, top_p, 9000 + index)})
    return cases


def run():
    return run_cases(
        build_cases(),
        sample_next_token,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
