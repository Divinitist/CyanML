from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import logistic_regression_step


def build_cases() -> list[dict]:
    cases = [
        {"name": "零 logit", "args": (np.array([[0.0]]), np.array([1.0]), np.array([0.0]), 0.0, 0.1, 0.0)},
        {"name": "极大正 logit", "args": (np.array([[1000.0]]), np.array([0.0]), np.array([1.0]), 0.0, 0.01, 0.0)},
        {"name": "极大负 logit", "args": (np.array([[1000.0]]), np.array([1.0]), np.array([-1.0]), 0.0, 0.01, 0.0)},
        {"name": "仅正则梯度", "args": (np.zeros((3, 2)), np.full(3, 0.5), np.array([2.0, -3.0]), 0.0, 0.1, 0.4)},
    ]
    rng = np.random.default_rng(20260929)
    scales = (0.01, 0.1, 1.0, 5.0, 20.0)
    while len(cases) < 100:
        index = len(cases)
        batch = 1 + index % 11
        features = 1 + index % 8
        scale = scales[index % len(scales)]
        x = rng.normal(size=(batch, features)).astype(np.float64) * scale
        weights = rng.normal(size=features).astype(np.float64)
        targets = rng.integers(0, 2, size=batch).astype(np.float64)
        bias = float(rng.normal() * scale)
        learning_rate = float(10 ** (-(1 + index % 4)))
        l2 = float((index % 5) * 0.025)
        cases.append({"name": f"随机覆盖 {index + 1}", "args": (x, targets, weights, bias, learning_rate, l2)})
    return cases


def reference(x, targets, weights, bias, learning_rate, l2):
    tensor_x = torch.from_numpy(x)
    tensor_targets = torch.from_numpy(targets)
    tensor_weights = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tensor_bias = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    logits = tensor_x @ tensor_weights + tensor_bias
    probabilities = torch.sigmoid(logits)
    loss = functional.binary_cross_entropy_with_logits(logits, tensor_targets)
    loss = loss + 0.5 * l2 * torch.sum(tensor_weights**2)
    loss.backward()
    grad_weights = tensor_weights.grad.detach().numpy()
    grad_bias = float(tensor_bias.grad.item())
    return (
        probabilities.detach().numpy(),
        float(loss.item()),
        (probabilities >= 0.5).to(torch.int64).numpy(),
        grad_weights,
        grad_bias,
        weights - learning_rate * grad_weights,
        float(bias - learning_rate * grad_bias),
    )


def run():
    return run_cases(
        build_cases(),
        logistic_regression_step,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与 PyTorch autograd 基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
