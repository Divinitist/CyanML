from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from answer import pca as reference
from submission import pca


def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        samples = 5 + index % 15
        features = 2 + index % 7
        latent = rng.normal(size=(samples, features))
        mixing = rng.normal(size=(features, features))
        x = (latent @ mixing + rng.normal(size=features) * (index % 5)).astype(np.float64)
        n_components = 1 + index % min(samples - 1, features)
        cases.append({"name": f"{features}维降至{n_components}维", "args": (x, n_components)})
    return cases


def run():
    return run_cases(
        build_cases(),
        pca,
        reference,
        rtol=1e-8,
        atol=1e-10,
        mismatch_message="输出与参考实现不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
