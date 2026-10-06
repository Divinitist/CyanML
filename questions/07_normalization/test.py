from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import normalization


def build_cases() -> list[dict]:
    cases = [
        {"name": "BatchNorm 两样本", "args": (np.array([[1.0, 2.0], [3.0, 4.0]]), np.ones(2), np.zeros(2), "batch", 1e-5)},
        {"name": "LayerNorm 单样本", "args": (np.array([[1.0, 2.0, 3.0]]), np.ones(3), np.zeros(3), "layer", 1e-5)},
        {"name": "RMSNorm 单样本", "args": (np.array([[1.0, -2.0, 3.0]]), np.ones(3), np.zeros(3), "rms", 1e-6)},
        {"name": "常量输入与仿射", "args": (np.full((4, 3), 7.0), np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 1.0]), "batch", 1e-5)},
    ]
    rng = np.random.default_rng(20260929)
    modes = ("batch", "layer", "rms")
    scales = (0.01, 0.1, 1.0, 10.0, 100.0)
    while len(cases) < 100:
        index = len(cases)
        batch = 1 + index % 10
        features = 1 + index % 12
        scale = scales[index % len(scales)]
        x = rng.normal(size=(batch, features)).astype(np.float64) * scale
        if index % 11 == 0:
            x.fill(float(index % 4))
        gamma = rng.normal(loc=1.0, scale=0.5, size=features).astype(np.float64)
        beta = rng.normal(scale=0.5, size=features).astype(np.float64)
        mode = modes[index % len(modes)]
        epsilon = (1e-5, 1e-6, 1e-8)[index % 3]
        cases.append({"name": f"{mode} 随机覆盖 {index + 1}", "args": (x, gamma, beta, mode, epsilon)})
    return cases


def reference(x, gamma, beta, mode, epsilon):
    values = torch.from_numpy(x)
    tensor_gamma = torch.from_numpy(gamma)
    tensor_beta = torch.from_numpy(beta)
    if mode == "batch":
        mean = torch.mean(values, dim=0, keepdim=True)
        variance = torch.mean((values - mean) ** 2, dim=0, keepdim=True)
        normalized = (values - mean) * torch.rsqrt(variance + epsilon)
    elif mode == "layer":
        mean = torch.mean(values, dim=1, keepdim=True)
        variance = torch.mean((values - mean) ** 2, dim=1, keepdim=True)
        normalized = (values - mean) * torch.rsqrt(variance + epsilon)
    else:
        mean_square = torch.mean(values**2, dim=1, keepdim=True)
        normalized = values * torch.rsqrt(mean_square + epsilon)
    return (normalized * tensor_gamma + tensor_beta).numpy()


def run():
    return run_cases(
        build_cases(),
        normalization,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="归一化输出与 PyTorch 双精度基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
