from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import run_cases

from submission import conv2d


def build_cases() -> list[dict]:
    cases = [
        {"name": "1×1 单通道", "args": (np.array([[[[2.0]]]]), np.array([[[[3.0]]]]), np.array([1.0]), 1, 0)},
        {"name": "3×3 单核", "args": (np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3), np.ones((1, 1, 2, 2)), np.zeros(1), 1, 0)},
        {"name": "stride 2", "args": (np.arange(25, dtype=np.float64).reshape(1, 1, 5, 5), np.ones((1, 1, 3, 3)), np.zeros(1), 2, 0)},
        {"name": "padding 1 多输出通道", "args": (np.ones((1, 2, 3, 4)), np.ones((3, 2, 2, 3)), np.arange(3, dtype=np.float64), 1, 1)},
    ]
    rng = np.random.default_rng(20260929)
    while len(cases) < 100:
        index = len(cases)
        batch = 1 + index % 3
        in_channels = 1 + index % 4
        out_channels = 1 + index % 5
        kernel_height = 1 + index % 3
        kernel_width = 1 + (index // 2) % 3
        stride = 1 + index % 2
        padding = index % 3
        height = kernel_height + 2 + index % 4
        width = kernel_width + 2 + (index // 3) % 4
        scale = (0.1, 1.0, 5.0)[index % 3]
        x = rng.normal(scale=scale, size=(batch, in_channels, height, width)).astype(np.float64)
        weight = rng.normal(size=(out_channels, in_channels, kernel_height, kernel_width)).astype(np.float64)
        bias = rng.normal(size=out_channels).astype(np.float64)
        cases.append({"name": f"随机覆盖 {index + 1}", "args": (x, weight, bias, stride, padding)})
    return cases


def reference(x, weight, bias, stride, padding):
    return functional.conv2d(
        torch.from_numpy(x),
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        stride=stride,
        padding=padding,
    ).numpy()


def run():
    return run_cases(
        build_cases(),
        conv2d,
        reference,
        rtol=1e-10,
        atol=1e-12,
        mismatch_message="卷积输出与 PyTorch conv2d 基准不一致。",
    )


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
