from __future__ import annotations

import ast
from pathlib import Path

import pytest


QUESTIONS_ROOT = Path(__file__).resolve().parents[1] / "questions"
QUESTION_PATHS = sorted(
    path for path in QUESTIONS_ROOT.glob("[0-9][0-9]_*") if (path / "manifest.json").is_file()
)
REQUIRED_SECTIONS = ("## 背景", "## 需求", "## 输入", "## 输出", "## 测试约定")


@pytest.mark.parametrize("question_path", QUESTION_PATHS, ids=lambda path: path.name)
def test_problem_statement_has_complete_structure(question_path: Path) -> None:
    statement = (question_path / "problem.md").read_text(encoding="utf-8")
    assert len(statement) >= 450, "题面过短，通常意味着接口、约束或测试约定缺失"
    for section in REQUIRED_SECTIONS:
        assert section in statement
    assert "100" in statement
    assert "rtol=" in statement and "atol=" in statement


@pytest.mark.parametrize("question_path", QUESTION_PATHS, ids=lambda path: path.name)
def test_template_documents_typed_public_interface(question_path: Path) -> None:
    source = (question_path / "template.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef))
    docstring = ast.get_docstring(function) or ""
    assert "Args:" in docstring
    assert "Returns:" in docstring
    assert function.returns is not None
    assert all(argument.annotation is not None for argument in function.args.args)
