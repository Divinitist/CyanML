import json
import shutil
from pathlib import Path

import pytest

from backend.cyanml_api.repository import QuestionNotFoundError, QuestionRepository


def create_question(root: Path) -> Path:
    question = root / "01_example"
    question.mkdir()
    (question / "manifest.json").write_text(
        '{"id":"01","slug":"example","title":"Example","library":"numpy"}',
        encoding="utf-8",
    )
    (question / "problem.md").write_text("# Example\n", encoding="utf-8")
    (question / "template.py").write_text("def solve():\n    pass\n", encoding="utf-8")
    (question / "submission.py").write_text("# old\n", encoding="utf-8")
    (question / "answer.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
    (question / "test.py").write_text("print('ok')\n", encoding="utf-8")
    return question


def test_user_can_list_and_read_questions(tmp_path: Path) -> None:
    create_question(tmp_path)
    repository = QuestionRepository(tmp_path)

    assert repository.list_questions() == [
        {
            "id": "01",
            "slug": "example",
            "title": "Example",
            "library": "numpy",
            "last_score": None,
        }
    ]
    question = repository.get_question("01")
    assert question["description"] == "# Example\n"
    assert question["submission"] == "# old\n"
    assert question["last_score"] is None
    assert question["last_result"] is None


def test_questions_are_listed_by_learning_order_without_changing_stable_ids(
    tmp_path: Path,
) -> None:
    first = create_question(tmp_path)
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    first_manifest["order"] = 2
    (first / "manifest.json").write_text(json.dumps(first_manifest), encoding="utf-8")

    second = tmp_path / "99_second"
    shutil.copytree(first, second)
    second_manifest = {
        **first_manifest,
        "id": "99",
        "slug": "second",
        "title": "Second",
        "order": 1,
    }
    (second / "manifest.json").write_text(json.dumps(second_manifest), encoding="utf-8")

    repository = QuestionRepository(tmp_path)

    assert [question["id"] for question in repository.list_questions()] == [
        "99",
        "01",
    ]


def test_latest_score_is_persisted_between_repository_instances(tmp_path: Path) -> None:
    create_question(tmp_path)
    repository = QuestionRepository(tmp_path)

    repository.save_last_score("01", 73)

    reloaded = QuestionRepository(tmp_path)
    assert reloaded.list_questions()[0]["last_score"] == 73
    assert reloaded.get_question("01")["last_score"] == 73


def test_latest_test_result_is_persisted_between_repository_instances(tmp_path: Path) -> None:
    create_question(tmp_path)
    repository = QuestionRepository(tmp_path)
    result = {
        "total": 100,
        "passed": 73,
        "failed": 27,
        "duration_ms": 12.5,
        "cases": [
            {
                "id": 1,
                "name": "example",
                "passed": False,
                "input": {"x": [1.0, 2.0]},
                "expected": 3.0,
                "actual": 2.0,
                "error": None,
            }
        ],
        "score": 73,
    }

    repository.save_last_result("01", result)

    reloaded = QuestionRepository(tmp_path)
    assert reloaded.get_question("01")["last_result"] == result
    assert reloaded.get_question("01")["last_score"] == 73


def test_user_can_save_and_reset_submission(tmp_path: Path) -> None:
    question_path = create_question(tmp_path)
    repository = QuestionRepository(tmp_path)

    repository.save_submission("01", "def solve():\n    return 2\n")
    assert (question_path / "submission.py").read_text(encoding="utf-8") == (
        "def solve():\n    return 2\n"
    )

    reset_code = repository.reset_submission("01")
    assert reset_code == "def solve():\n    pass\n"
    assert (question_path / "submission.py").read_text(encoding="utf-8") == reset_code


def test_unknown_question_is_rejected(tmp_path: Path) -> None:
    repository = QuestionRepository(tmp_path)

    with pytest.raises(QuestionNotFoundError):
        repository.get_question("404")
