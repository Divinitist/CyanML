from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RESULT_PREFIX = "CYANML_RESULT="


class QuestionRunner:
    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, question_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        child_environment = os.environ.copy()
        child_environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        try:
            process = subprocess.run(
                [sys.executable, "test.py"],
                cwd=question_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_environment,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return self._error_result(
                f"测试超过 {self.timeout_seconds} 秒，已终止。",
                started,
            )

        if process.returncode != 0:
            details = process.stderr.strip() or process.stdout.strip()
            return self._error_result(
                details or "测试进程异常退出。",
                started,
                process.stdout.strip(),
            )

        output_lines = process.stdout.splitlines()
        for index in range(len(output_lines) - 1, -1, -1):
            line = output_lines[index]
            if line.startswith(RESULT_PREFIX):
                try:
                    result = json.loads(line.removeprefix(RESULT_PREFIX))
                except json.JSONDecodeError as error:
                    return self._error_result(
                        f"测试结果不是有效 JSON：{error}",
                        started,
                        process.stdout.strip(),
                    )
                standard_output = "\n".join(
                    output_lines[:index] + output_lines[index + 1 :]
                ).strip()
                if standard_output:
                    result["stdout"] = standard_output
                return result

        return self._error_result(
            "测试脚本没有输出结构化结果。",
            started,
            process.stdout.strip(),
        )

    @staticmethod
    def _error_result(
        message: str,
        started: float,
        standard_output: str = "",
    ) -> dict[str, Any]:
        result = {
            "total": 0,
            "passed": 0,
            "failed": 1,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "cases": [],
            "error": message,
        }
        if standard_output:
            result["stdout"] = standard_output
        return result
