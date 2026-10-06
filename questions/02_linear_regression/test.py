from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from submission import linear_regression_step

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import capture_standard_output


RTOL = 1e-10
ATOL = 1e-12


def build_cases() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, str]]:
    cases = [
        (np.array([[1.0]]), np.array([2.0]), np.array([0.5]), 0.0, 0.1, "单样本单特征"),
        (np.eye(3), np.array([1.0, 2.0, 3.0]), np.ones(3), 0.0, 0.01, "单位特征"),
        (np.zeros((4, 2)), np.full(4, 3.0), np.array([8.0, -2.0]), 3.0, 0.2, "零残差"),
        (np.ones((2, 2)), np.zeros(2), np.ones(2), -2.0, 1e-8, "极小学习率"),
    ]
    rng = np.random.default_rng(20260929)
    scales = (0.01, 0.1, 1.0, 10.0, 100.0)
    while len(cases) < 100:
        index = len(cases)
        batch_size = 1 + index % 12
        num_features = 1 + index % 9
        scale = scales[index % len(scales)]
        x = rng.normal(size=(batch_size, num_features)).astype(np.float64) * scale
        weights = rng.normal(size=num_features).astype(np.float64)
        bias = float(rng.normal())
        noise = rng.normal(size=batch_size) * (0.01 if index % 3 else 1.0)
        targets = (x @ rng.normal(size=num_features) + rng.normal() + noise).astype(np.float64)
        learning_rate = float((1 + index % 5) * 10 ** (-(2 + index % 3)))
        cases.append((x, targets, weights, bias, learning_rate, f"随机覆盖 {index + 1}"))
    return cases


def compact(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return {"shape": list(value.shape), "values": flat[:12].tolist(), "truncated": flat.size > 12}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    return value


def reference(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float]:
    tensor_x = torch.from_numpy(x)
    tensor_targets = torch.from_numpy(targets)
    tensor_weights = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    tensor_bias = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    predictions = tensor_x @ tensor_weights + tensor_bias
    loss = torch.mean((predictions - tensor_targets) ** 2)
    loss.backward()
    grad_weights = tensor_weights.grad.detach().numpy()
    grad_bias = float(tensor_bias.grad.item())
    return (
        predictions.detach().numpy(),
        float(loss.item()),
        grad_weights,
        grad_bias,
        weights - learning_rate * grad_weights,
        float(bias - learning_rate * grad_bias),
    )


def outputs_match(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, tuple) or len(actual) != 6:
        return False
    return all(
        np.allclose(np.asarray(actual_item), np.asarray(expected_item), rtol=RTOL, atol=ATOL)
        for actual_item, expected_item in zip(actual, expected)
    )


def run() -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for case_id, (x, targets, weights, bias, learning_rate, name) in enumerate(build_cases(), start=1):
        expected = reference(x, targets, weights, bias, learning_rate)
        item: dict[str, Any] = {
            "id": case_id,
            "name": name,
            "input": compact({"x": compact(x), "targets": compact(targets), "weights": compact(weights), "bias": bias, "learning_rate": learning_rate}),
            "expected": compact(expected),
        }
        try:
            with capture_standard_output() as case_stdout:
                actual = linear_regression_step(
                    x.copy(),
                    targets.copy(),
                    weights.copy(),
                    bias,
                    learning_rate,
                )
            passed = outputs_match(actual, expected)
            item.update({"passed": passed, "actual": compact(actual), "error": None if passed else "输出与 PyTorch autograd 基准不一致。"})
        except Exception as error:
            item.update({"passed": False, "actual": None, "error": f"{type(error).__name__}: {error}"})
        standard_output = case_stdout.getvalue().rstrip()
        if standard_output:
            item["stdout"] = standard_output
        results.append(item)
    passed_count = sum(item["passed"] for item in results)
    return {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "cases": results,
    }


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
