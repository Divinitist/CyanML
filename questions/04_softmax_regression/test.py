from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import softmax_regression_step


def build_cases() -> list[dict]:
    cases = [
        {"name": "单样本二分类", "args": (np.array([[1.0]]), np.array([1]), np.array([[0.0, 1.0]]), np.zeros(2), 0.1, 0.0)},
        {"name": "等概率三分类", "args": (np.ones((2, 2)), np.array([0, 2]), np.zeros((2, 3)), np.zeros(3), 0.01, 0.0)},
        {"name": "大幅值参数", "args": (np.array([[10.0, -10.0]]), np.array([1]), np.array([[100.0, -100.0], [-100.0, 100.0]]), np.zeros(2), 1e-4, 0.0)},
        {"name": "含 L2 正则", "args": (np.zeros((3, 2)), np.array([0, 1, 2]), np.arange(6, dtype=np.float64).reshape(2, 3), np.zeros(3), 0.1, 0.2)},
    ]
    rng = np.random.default_rng(20260929)
    scales = (0.01, 0.1, 1.0, 5.0, 20.0)
    while len(cases) < 100:
        index = len(cases)
        batch = 1 + index % 10
        features = 1 + index % 7
        classes = 2 + index % 8
        scale = scales[index % len(scales)]
        x = rng.normal(size=(batch, features)).astype(np.float64) * scale
        weights = rng.normal(size=(features, classes)).astype(np.float64)
        bias = rng.normal(size=classes).astype(np.float64) * scale
        targets = rng.integers(0, classes, size=batch, dtype=np.int64)
        learning_rate = float(10 ** (-(1 + index % 4)))
        l2 = float((index % 5) * 0.02)
        cases.append({"name": f"随机覆盖 {index + 1}", "args": (x, targets, weights, bias, learning_rate, l2)})
    return cases


def reference(x, targets, weights, bias, learning_rate, l2):
    tensor_x = torch.from_numpy(x)
    tensor_targets = torch.from_numpy(targets)
    tensor_weights = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tensor_bias = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    logits = tensor_x @ tensor_weights + tensor_bias
    probabilities = torch.softmax(logits, dim=1)
    loss = functional.cross_entropy(logits, tensor_targets)
    loss = loss + 0.5 * l2 * torch.sum(tensor_weights**2)
    loss.backward()
    grad_weights = tensor_weights.grad.detach().numpy()
    grad_bias = tensor_bias.grad.detach().numpy()
    return (
        probabilities.detach().numpy(),
        float(loss.item()),
        torch.argmax(probabilities, dim=1).numpy(),
        grad_weights,
        grad_bias,
        weights - learning_rate * grad_weights,
        bias - learning_rate * grad_bias,
    )


def run():
    return run_cases(
        build_cases(),
        softmax_regression_step,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与 PyTorch autograd 基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
