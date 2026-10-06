from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import ppo_clipped_loss


def build_cases():
    rng = np.random.default_rng(410041)
    cases = []
    clips = (0.05, 0.1, 0.2, 0.3, 0.5)
    for index in range(100):
        shape = ((1,), (2 + index % 13,), (2 + index % 5, 3 + index % 7))[index % 3]
        old = rng.normal(-1.0, 1.5, size=shape).astype(np.float64)
        shift = rng.uniform(-2.0, 2.0, size=shape)
        if index % 11 == 0:
            shift.fill(0.0)
        new = old + shift
        advantages = rng.normal(size=shape).astype(np.float64)
        if index % 7 == 0:
            advantages.flat[0] = 0.0
        clip = clips[index % len(clips)]
        cases.append({"name": f"shape={shape}, clip={clip}", "args": (new, old, advantages, clip)})
    return cases


def reference(new_log_probs, old_log_probs, advantages, clip_epsilon):
    ratios = np.exp(new_log_probs - old_log_probs)
    lower, upper = 1.0 - clip_epsilon, 1.0 + clip_epsilon
    bounded = np.maximum(lower, np.minimum(upper, ratios))
    objectives = np.where(
        ratios * advantages < bounded * advantages,
        ratios * advantages,
        bounded * advantages,
    )
    return float(-objectives.sum() / objectives.size)


def run():
    return run_cases(
        build_cases(), ppo_clipped_loss, reference,
        rtol=1e-12, atol=1e-12,
        mismatch_message="PPO clipped policy loss 与公式基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
