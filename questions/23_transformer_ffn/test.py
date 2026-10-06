from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import transformer_ffn as reference
from submission import transformer_ffn


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    modes = ("relu", "gelu", "glu", "geglu", "swiglu")
    for index in range(100):
        model_dim, hidden_dim = 2 + index % 6, 3 + index % 9
        shape = (1 + index % 5, model_dim) if index % 3 else (2, 1 + index % 4, model_dim)
        x = rng.normal(size=shape)
        w_up = rng.normal(scale=0.3, size=(model_dim, hidden_dim)); b_up = rng.normal(scale=0.1, size=hidden_dim)
        w_down = rng.normal(scale=0.3, size=(hidden_dim, model_dim)); b_down = rng.normal(scale=0.1, size=model_dim)
        mode = modes[index % len(modes)]
        if mode in ("relu", "gelu"):
            w_gate = b_gate = None
        else:
            w_gate = rng.normal(scale=0.3, size=(model_dim, hidden_dim)); b_gate = rng.normal(scale=0.1, size=hidden_dim)
        cases.append({"name": f"{mode}, shape={shape}", "args": (x, w_up, b_up, w_down, b_down, mode, w_gate, b_gate)})
    return cases


def run():
    return run_cases(
        build_cases(),
        transformer_ffn,
        reference,
        rtol=1e-9,
        atol=1e-11,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
