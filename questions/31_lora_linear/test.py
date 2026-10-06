from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import lora_linear as reference
from submission import lora_linear


def build_cases():
    rng = np.random.default_rng(310031); cases = []
    for index in range(100):
        inputs, outputs = 2 + index % 7, 2 + (index * 3) % 8; rank = 1 + index % min(4, inputs, outputs)
        shape = (1 + index % 5, inputs) if index % 3 else (2, 1 + index % 4, inputs)
        x = rng.normal(size=shape); weight = rng.normal(scale=.3, size=(inputs, outputs)); bias = None if index % 4 == 0 else rng.normal(scale=.1, size=outputs); a = rng.normal(scale=.2, size=(inputs, rank)); b = rng.normal(scale=.2, size=(rank, outputs)); alpha = (.5, 1., 4., 8.)[index % 4]; dropout = (0., .1, .25, .5)[index % 4]; training = bool(index % 2)
        cases.append({"name": f"rank={rank}, dropout={dropout}, training={training}", "args": (x, weight, bias, a, b, alpha, dropout, training, 12000 + index)})
    return cases


def run():
    return run_cases(
        build_cases(),
        lora_linear,
        reference,
        rtol=1e-9,
        atol=1e-11,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
