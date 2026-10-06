from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import dpo_loss


def build_cases():
    rng = np.random.default_rng(420042)
    cases = []
    betas = (0.01, 0.1, 0.5, 1.0, 5.0)
    for index in range(100):
        shape = ((1,), (2 + index % 31,), (2 + index % 4, 3 + index % 7))[index % 3]
        scale = (1.0, 10.0, 100.0)[index % 3]
        policy_chosen = -np.abs(rng.normal(size=shape) * scale)
        policy_rejected = -np.abs(rng.normal(size=shape) * scale)
        reference_chosen = -np.abs(rng.normal(size=shape) * scale)
        reference_rejected = -np.abs(rng.normal(size=shape) * scale)
        if index % 10 == 0:
            policy_chosen = reference_chosen.copy()
            policy_rejected = reference_rejected.copy()
        beta = betas[index % len(betas)]
        cases.append({
            "name": f"shape={shape}, beta={beta}, scale={scale}",
            "args": (
                policy_chosen, policy_rejected,
                reference_chosen, reference_rejected, beta,
            ),
        })
    return cases


def reference(chosen, rejected, reference_chosen, reference_rejected, beta):
    margins = beta * ((chosen - rejected) - (reference_chosen - reference_rejected))
    losses = np.maximum(-margins, 0.0) + np.log1p(np.exp(-np.abs(margins)))
    return float(losses.sum() / losses.size)


def run():
    return run_cases(
        build_cases(), dpo_loss, reference,
        rtol=1e-12, atol=1e-12,
        mismatch_message="DPO loss 与稳定公式基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
