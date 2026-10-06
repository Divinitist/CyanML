from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import generalized_advantage_estimation


def build_cases():
    rng = np.random.default_rng(400040)
    cases = []
    gammas = (0.0, 0.5, 0.9, 0.99, 1.0)
    lambdas = (0.0, 0.5, 0.8, 0.95, 1.0)
    for index in range(100):
        steps = 1 + index % 64
        scale = (0.01, 1.0, 20.0)[index % 3]
        rewards = rng.normal(size=steps).astype(np.float64) * scale
        values = rng.normal(size=steps + 1).astype(np.float64) * scale
        dones = rng.random(steps) < (0.05 + 0.1 * (index % 4))
        if index % 4 == 0:
            dones[-1] = True
        if index % 13 == 0:
            rewards.fill(0.0)
        gamma = gammas[index % len(gammas)]
        gae_lambda = lambdas[(index // len(gammas)) % len(lambdas)]
        cases.append({
            "name": f"steps={steps}, gamma={gamma}, lambda={gae_lambda}",
            "args": (rewards, values, dones, gamma, gae_lambda),
        })
    return cases


def reference(rewards, values, dones, gamma, gae_lambda):
    output = np.zeros_like(rewards, dtype=np.float64)
    accumulator = 0.0
    for index in reversed(range(rewards.size)):
        mask = 0.0 if dones[index] else 1.0
        temporal_difference = rewards[index] + gamma * mask * values[index + 1] - values[index]
        accumulator = temporal_difference + gamma * gae_lambda * mask * accumulator
        output[index] = accumulator
    return output, output + values[:-1]


def run():
    return run_cases(
        build_cases(), generalized_advantage_estimation, reference,
        rtol=1e-11, atol=1e-12,
        mismatch_message="GAE 优势或回报与递推基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
