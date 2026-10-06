import json
from pathlib import Path

import pytest

from backend.cyanml_api.runner import QuestionRunner


def test_runner_returns_structured_case_results(tmp_path: Path) -> None:
    cases = [
        {
            "id": index + 1,
            "name": f"case {index + 1}",
            "passed": index != 4,
            "input": {"value": index},
            "expected": index,
            "actual": index if index != 4 else -1,
            "error": None if index != 4 else "values differ",
        }
        for index in range(100)
    ]
    payload = {
        "total": 100,
        "passed": 99,
        "failed": 1,
        "duration_ms": 12.5,
        "cases": cases,
    }
    script = "import json\nprint('CYANML_RESULT=' + json.dumps(" + repr(payload) + "))\n"
    (tmp_path / "test.py").write_text(script, encoding="utf-8")

    result = QuestionRunner(timeout_seconds=2).run(tmp_path)

    assert result == payload


def test_runner_returns_standard_output_without_the_result_protocol_line(
    tmp_path: Path,
) -> None:
    payload = {"total": 1, "passed": 1, "failed": 0, "cases": []}
    (tmp_path / "test.py").write_text(
        "import json\n"
        "print('debug value=3')\n"
        "print('second line')\n"
        "print('CYANML_RESULT=' + json.dumps(" + repr(payload) + "))\n",
        encoding="utf-8",
    )

    result = QuestionRunner(timeout_seconds=2).run(tmp_path)

    assert result["stdout"] == "debug value=3\nsecond line"


def test_runner_keeps_candidate_output_attached_to_each_case(tmp_path: Path) -> None:
    support_path = Path(__file__).resolve().parents[1] / "questions"
    (tmp_path / "test.py").write_text(
        "import json\n"
        "import sys\n"
        f"sys.path.insert(0, {str(support_path)!r})\n"
        "from test_support import run_cases\n"
        "def candidate(value):\n"
        "    print(f'case output {value}')\n"
        "    return value\n"
        "result = run_cases(\n"
        "    [{'name': 'one', 'args': (1,)}, {'name': 'two', 'args': (2,)}],\n"
        "    candidate, lambda value: value, rtol=0.0, atol=0.0,\n"
        "    mismatch_message='mismatch',\n"
        ")\n"
        "print('CYANML_RESULT=' + json.dumps(result))\n",
        encoding="utf-8",
    )

    result = QuestionRunner(timeout_seconds=2).run(tmp_path)

    assert "stdout" not in result
    assert result["cases"][0]["stdout"] == "case output 1"
    assert result["cases"][1]["stdout"] == "case output 2"


def test_runner_reports_test_process_errors(tmp_path: Path) -> None:
    (tmp_path / "test.py").write_text(
        "print('before crash')\nraise RuntimeError('broken test module')\n",
        encoding="utf-8",
    )

    result = QuestionRunner(timeout_seconds=2).run(tmp_path)

    assert result["total"] == 0
    assert result["failed"] == 1
    assert "broken test module" in result["error"]
    assert result["stdout"] == "before crash"


def test_runner_preserves_chinese_output_when_windows_locale_is_gbk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "total": 1,
        "passed": 0,
        "failed": 1,
        "cases": [{"id": 1, "name": "中文用例", "error": "结果不一致"}],
    }
    (tmp_path / "test.py").write_text(
        "import json\nprint('CYANML_RESULT=' + json.dumps("
        + repr(payload)
        + ", ensure_ascii=False))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONIOENCODING", "gbk")
    monkeypatch.setenv("PYTHONUTF8", "0")

    result = QuestionRunner(timeout_seconds=2).run(tmp_path)

    assert result["cases"][0]["name"] == "中文用例"
    assert result["cases"][0]["error"] == "结果不一致"
