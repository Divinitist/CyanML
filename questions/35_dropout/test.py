from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import dropout as reference
from submission import dropout


def build_cases():
    rng = np.random.default_rng(350035); cases = []
    for index in range(100):
        shape = ((1 + index % 11,), (2 + index % 4, 1 + index % 7), (2, 3, 1 + index % 5))[index % 3]
        x = rng.normal(size=shape); probability = (0., .1, .25, .5, .8)[index % 5]; training = bool(index % 2)
        if index % 13 == 0: x.fill(0.)
        cases.append({"name": f"shape={shape}, p={probability}, training={training}", "args": (x, probability, training, 15000 + index)})
    return cases


def run():
    return run_cases(
        build_cases(),
        dropout,
        reference,
        rtol=1e-12,
        atol=1e-12,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
