from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import time
from typing import Any, Callable

import numpy as np


@contextmanager
def capture_standard_output() -> Iterator[StringIO]:
    output = StringIO()
    with redirect_stdout(output):
        yield output


def compact(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return {
            "shape": list(value.shape),
            "values": flat[:12].tolist(),
            "truncated": flat.size > 12,
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    return value


def clone(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    return value


def all_close(actual: Any, expected: Any, rtol: float, atol: float) -> bool:
    if isinstance(expected, (tuple, list)):
        return (
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected)
            and all(
                all_close(actual_item, expected_item, rtol, atol)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    try:
        return bool(
            np.allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=rtol,
                atol=atol,
            )
        )
    except (TypeError, ValueError):
        return actual == expected


def run_cases(
    cases: list[dict[str, Any]],
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    *,
    rtol: float,
    atol: float,
    mismatch_message: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for case_id, case in enumerate(cases, start=1):
        expected = reference(*clone(case["args"]))
        item = {
            "id": case_id,
            "name": case["name"],
            "input": compact(case.get("input", case["args"])),
            "expected": compact(expected),
        }
        try:
            with capture_standard_output() as case_stdout:
                actual = candidate(*clone(case["args"]))
            passed = all_close(actual, expected, rtol, atol)
            item.update(
                {
                    "passed": passed,
                    "actual": compact(actual),
                    "error": None if passed else mismatch_message,
                }
            )
        except Exception as error:
            item.update(
                {
                    "passed": False,
                    "actual": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        standard_output = case_stdout.getvalue().rstrip()
        if standard_output:
            item["stdout"] = standard_output
        results.append(item)

    passed = sum(item["passed"] for item in results)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "cases": results,
    }
