from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import grpo_loss


def build_cases():
    rng = np.random.default_rng(430043)
    cases = []
    clips = (0.05, 0.1, 0.2, 0.3)
    betas = (0.0, 0.01, 0.05, 0.2)
    for index in range(100):
        groups = 1 + index % 4
        samples = 2 + index % 6
        tokens = 1 + index % 12
        shape = (groups, samples, tokens)
        old = rng.normal(-2.0, 1.0, size=shape)
        new = old + rng.uniform(-1.0, 1.0, size=shape)
        reference = old + rng.uniform(-0.8, 0.8, size=shape)
        rewards = rng.normal(size=(groups, samples))
        if index % 10 == 0:
            rewards[:] = rewards[:, :1]
        lengths = rng.integers(1, tokens + 1, size=(groups, samples))
        positions = np.arange(tokens)[None, None, :]
        mask = positions < lengths[..., None]
        clip = clips[index % len(clips)]
        beta = betas[(index // len(clips)) % len(betas)]
        epsilon = (1e-8, 1e-6)[index % 2]
        cases.append({
            "name": f"shape={shape}, clip={clip}, beta={beta}",
            "args": (new, old, reference, rewards, mask, clip, beta, epsilon),
        })
    return cases


def reference(new, old, reference_log_probs, rewards, mask, clip, beta, epsilon):
    group_means = rewards.sum(axis=1, keepdims=True) / rewards.shape[1]
    deviations = rewards - group_means
    group_variances = (deviations * deviations).sum(axis=1, keepdims=True) / rewards.shape[1]
    advantages = deviations / np.sqrt(group_variances + epsilon)
    ratios = np.exp(new - old)
    clipped = np.maximum(1.0 - clip, np.minimum(1.0 + clip, ratios))
    first = ratios * advantages[:, :, None]
    second = clipped * advantages[:, :, None]
    policy_objective = np.where(first < second, first, second)
    log_reference_ratio = reference_log_probs - new
    kl = np.exp(log_reference_ratio) - log_reference_ratio - 1.0
    token_objective = policy_objective - beta * kl
    response_values = []
    for group in range(rewards.shape[0]):
        for sample in range(rewards.shape[1]):
            selected = token_objective[group, sample][mask[group, sample]]
            response_values.append(float(selected.sum() / selected.size))
    return float(-sum(response_values) / len(response_values))


def run():
    return run_cases(
        build_cases(), grpo_loss, reference,
        rtol=1e-11, atol=1e-12,
        mismatch_message="GRPO loss 与组内标准化和裁剪公式基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
