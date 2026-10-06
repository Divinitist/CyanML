from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import two_layer_mlp


def build_cases() -> list[dict]:
    cases = [
        {"name": "单神经元 ReLU", "args": (np.array([[2.0]]), np.array([[3.0]]), np.array([-1.0]), np.array([[4.0]]), np.array([0.5]), "relu")},
        {"name": "ReLU 截断负值", "args": (np.array([[-2.0]]), np.array([[3.0]]), np.array([1.0]), np.array([[4.0]]), np.array([0.5]), "relu")},
        {"name": "tanh 零输入", "args": (np.zeros((2, 3)), np.ones((3, 2)), np.zeros(2), np.ones((2, 1)), np.zeros(1), "tanh")},
        {"name": "仅输出偏置", "args": (np.ones((3, 2)), np.zeros((2, 4)), np.zeros(4), np.zeros((4, 2)), np.array([2.0, -1.0]), "relu")},
    ]
    rng = np.random.default_rng(20260929)
    scales = (0.01, 0.1, 1.0, 3.0, 10.0)
    while len(cases) < 100:
        index = len(cases)
        batch = 1 + index % 8
        input_size = 1 + index % 7
        hidden_size = 1 + index % 9
        output_size = 1 + index % 6
        scale = scales[index % len(scales)]
        x = rng.normal(size=(batch, input_size)).astype(np.float64) * scale
        weight1 = rng.normal(size=(input_size, hidden_size)).astype(np.float64)
        bias1 = rng.normal(size=hidden_size).astype(np.float64)
        weight2 = rng.normal(size=(hidden_size, output_size)).astype(np.float64)
        bias2 = rng.normal(size=output_size).astype(np.float64)
        activation = "relu" if index % 2 == 0 else "tanh"
        cases.append({"name": f"{activation} 随机覆盖 {index + 1}", "args": (x, weight1, bias1, weight2, bias2, activation)})
    return cases


def reference(x, weight1, bias1, weight2, bias2, activation):
    hidden = torch.from_numpy(x) @ torch.from_numpy(weight1) + torch.from_numpy(bias1)
    hidden = functional.relu(hidden) if activation == "relu" else torch.tanh(hidden)
    return (hidden @ torch.from_numpy(weight2) + torch.from_numpy(bias2)).numpy()


def run():
    return run_cases(
        build_cases(),
        two_layer_mlp,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="输出与 PyTorch 双精度基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
