from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import adamw_update


def make_case(name, parameter, gradient, first, second, step, lr, beta1, beta2, eps, decay):
    return {"name": name, "args": (parameter, gradient, first, second, step, lr, beta1, beta2, eps, decay)}


def build_cases() -> list[dict]:
    cases = [
        make_case("首次标量更新", np.array([1.0]), np.array([0.5]), np.zeros(1), np.zeros(1), 1, 1e-3, 0.9, 0.999, 1e-8, 0.01),
        make_case("零梯度仅衰减", np.array([2.0, -3.0]), np.zeros(2), np.zeros(2), np.zeros(2), 1, 0.1, 0.9, 0.999, 1e-8, 0.2),
        make_case("已有矩状态", np.array([1.0, 2.0]), np.array([0.2, -0.4]), np.array([0.1, 0.3]), np.array([0.2, 0.5]), 10, 0.01, 0.8, 0.95, 1e-6, 0.0),
        make_case("矩阵参数", np.eye(2), np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), 2, 0.001, 0.9, 0.99, 1e-8, 0.1),
    ]
    rng = np.random.default_rng(20260929)
    shapes = ((1,), (3,), (2, 4), (3, 3), (2, 2, 2))
    while len(cases) < 100:
        index = len(cases)
        shape = shapes[index % len(shapes)]
        parameter = rng.normal(size=shape).astype(np.float64)
        gradient = rng.normal(size=shape).astype(np.float64)
        first = rng.normal(scale=0.1, size=shape).astype(np.float64)
        second = rng.uniform(0.0, 1.0, size=shape).astype(np.float64)
        step = (1, 2, 10, 100, 1000)[index % 5]
        lr = float((1, 3, 10)[index % 3] * 1e-4)
        beta1 = (0.8, 0.9, 0.95)[index % 3]
        beta2 = (0.9, 0.99, 0.999)[index % 3]
        eps = (1e-8, 1e-6)[index % 2]
        decay = (0.0, 0.01, 0.1)[index % 3]
        cases.append(make_case(f"随机覆盖 {index + 1}", parameter, gradient, first, second, step, lr, beta1, beta2, eps, decay))
    return cases


def reference(parameter, gradient, first, second, step, learning_rate, beta1, beta2, epsilon, weight_decay):
    tensor_parameter = torch.tensor(parameter, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [tensor_parameter],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )
    state = optimizer.state[tensor_parameter]
    state["step"] = torch.tensor(float(step - 1))
    state["exp_avg"] = torch.from_numpy(first.copy())
    state["exp_avg_sq"] = torch.from_numpy(second.copy())
    tensor_parameter.grad = torch.from_numpy(gradient.copy())
    optimizer.step()
    return (
        tensor_parameter.detach().numpy(),
        state["exp_avg"].numpy(),
        state["exp_avg_sq"].numpy(),
    )


def run():
    return run_cases(
        build_cases(),
        adamw_update,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="更新结果与 PyTorch AdamW 基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
