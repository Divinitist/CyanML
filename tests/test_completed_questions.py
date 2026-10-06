from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


QUESTIONS_ROOT = Path(__file__).resolve().parents[1] / "questions"
QUESTION_PATHS = sorted(
    path for path in QUESTIONS_ROOT.iterdir() if (path / "manifest.json").is_file()
)

VERIFY_SCRIPT = r"""
import importlib.util
import json
import sys
from pathlib import Path

question_path = Path(sys.argv[1]).resolve()
entrypoint = sys.argv[2]
sys.path.insert(0, str(question_path))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

tests = load("question_tests", question_path / "test.py")
answer = load("question_answer", question_path / "answer.py")
setattr(tests, entrypoint, getattr(answer, entrypoint))
result = tests.run()
print(json.dumps({key: result[key] for key in ("total", "passed", "failed")}))
"""


@pytest.mark.parametrize("question_path", QUESTION_PATHS, ids=lambda path: path.name)
def test_reference_answer_passes_all_100_cases(question_path: Path) -> None:
    manifest = json.loads((question_path / "manifest.json").read_text(encoding="utf-8"))
    process = subprocess.run(
        [sys.executable, "-c", VERIFY_SCRIPT, str(question_path), manifest["entrypoint"]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result == {"total": 100, "passed": 100, "failed": 0}
