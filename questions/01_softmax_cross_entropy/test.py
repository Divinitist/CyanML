from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from submission import softmax_cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_support import capture_standard_output


RTOL = 1e-10
ATOL = 1e-12


def build_cases() -> list[tuple[np.ndarray, np.ndarray, str]]:
    cases: list[tuple[np.ndarray, np.ndarray, str]] = [
        (np.array([[0.0, 0.0]], dtype=np.float64), np.array([0]), "两类等值"),
        (np.array([[1.0, 2.0, 3.0]], dtype=np.float64), np.array([2]), "单样本三分类"),
        (
            np.array([[1000.0, 999.0], [-1000.0, -999.0]], dtype=np.float64),
            np.array([0, 1]),
            "极大绝对值",
        ),
        (
            np.array([[1e6, 1e6 - 1, 1e6 - 2]], dtype=np.float64),
            np.array([1]),
            "巨大整体平移",
        ),
    ]
    rng = np.random.default_rng(20260929)
    scales = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    while len(cases) < 100:
        index = len(cases)
        batch_size = 1 + index % 8
        num_classes = 2 + index % 11
        scale = scales[index % len(scales)]
        logits = rng.normal(0.0, scale, size=(batch_size, num_classes))
        if index % 5 == 0:
            logits += rng.choice((-1e6, 1e6))
        targets = rng.integers(0, num_classes, size=batch_size, dtype=np.int64)
        cases.append((logits.astype(np.float64), targets, f"随机覆盖 {index + 1}"))
    return cases


def summarize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        flattened = value.reshape(-1)
        return {
            "shape": list(value.shape),
            "values": flattened[:12].tolist(),
            "truncated": flattened.size > 12,
        }
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    return value


def reference(logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    tensor_logits = torch.from_numpy(logits)
    tensor_targets = torch.from_numpy(targets)
    probabilities = torch.softmax(tensor_logits, dim=1).numpy()
    loss = functional.cross_entropy(tensor_logits, tensor_targets).item()
    return probabilities, loss


def run() -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for case_id, (logits, targets, name) in enumerate(build_cases(), start=1):
        expected_probabilities, expected_loss = reference(logits, targets)
        case_result: dict[str, Any] = {
            "id": case_id,
            "name": name,
            "input": {
                "logits": summarize(logits),
                "targets": summarize(targets),
            },
            "expected": {
                "probabilities": summarize(expected_probabilities),
                "loss": expected_loss,
            },
        }
        try:
            with capture_standard_output() as case_stdout:
                actual_probabilities, actual_loss = softmax_cross_entropy(
                    logits.copy(), targets.copy()
                )
            actual_probabilities = np.asarray(actual_probabilities)
            probabilities_match = np.allclose(
                actual_probabilities,
                expected_probabilities,
                rtol=RTOL,
                atol=ATOL,
            )
            loss_matches = np.isclose(actual_loss, expected_loss, rtol=RTOL, atol=ATOL)
            passed = bool(probabilities_match and loss_matches)
            case_result.update(
                {
                    "passed": passed,
                    "actual": {
                        "probabilities": summarize(actual_probabilities),
                        "loss": summarize(actual_loss),
                    },
                    "error": None if passed else "概率或交叉熵与 PyTorch 基准不一致。",
                }
            )
        except Exception as error:
            case_result.update(
                {
                    "passed": False,
                    "actual": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        standard_output = case_stdout.getvalue().rstrip()
        if standard_output:
            case_result["stdout"] = standard_output
        results.append(case_result)

    passed = sum(result["passed"] for result in results)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "cases": results,
    }


if __name__ == "__main__":
    print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
