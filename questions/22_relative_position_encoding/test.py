from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import relative_position_bias as reference
from submission import relative_position_bias


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        query_length = 1 + index % 9
        key_length = 1 + (index * 3) % 11
        heads = 1 + index % 6
        maximum = 1 + index % 5
        mode = "alibi" if index % 2 == 0 else "table"
        table = None if mode == "alibi" else rng.normal(size=(heads, 2 * maximum + 1))
        cases.append({"name": f"{mode} {query_length}x{key_length}", "args": (query_length, key_length, heads, mode, maximum, table)})
    return cases


def run():
    return run_cases(
        build_cases(),
        relative_position_bias,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
