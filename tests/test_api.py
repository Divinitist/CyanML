import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.cyanml_api.app import create_app


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
    payload = {"total": 100, "passed": 100, "failed": 0, "cases": []}
    (question / "test.py").write_text(
        "import json\nprint('debug from submission')\n"
        "print('CYANML_RESULT=' + json.dumps("
        + repr(payload)
        + "))\n",
        encoding="utf-8",
    )
    return question


def test_question_workflow_through_http_api(tmp_path: Path) -> None:
    create_question(tmp_path)
    client = TestClient(create_app(tmp_path))

    initial_summary = client.get("/api/questions").json()[0]
    assert initial_summary["id"] == "01"
    assert initial_summary["last_score"] is None
    detail = client.get("/api/questions/01").json()
    assert detail["submission"] == "# old\n"
    assert detail["answer"] == "def solve():\n    return 1\n"
    assert detail["last_score"] is None
    assert detail["last_result"] is None

    save = client.put(
        "/api/questions/01/submission",
        json={"code": "def solve():\n    return 2\n"},
    )
    assert save.status_code == 204

    test_result = client.post("/api/questions/01/test").json()
    assert test_result["passed"] == 100
    assert test_result["score"] == 100
    assert test_result["stdout"] == "debug from submission"
    assert client.get("/api/questions").json()[0]["last_score"] == 100
    assert client.get("/api/questions/01").json()["last_result"] == test_result

    reloaded_client = TestClient(create_app(tmp_path))
    assert reloaded_client.get("/api/questions").json()[0]["last_score"] == 100
    assert reloaded_client.get("/api/questions/01").json()["last_result"] == test_result

    reset = client.post("/api/questions/01/reset").json()
    assert reset["code"] == "def solve():\n    pass\n"


def test_unknown_question_returns_404(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    assert client.get("/api/questions/404").status_code == 404


def test_failed_attempt_caches_percentage_score(tmp_path: Path) -> None:
    question = create_question(tmp_path)
    payload = {"total": 100, "passed": 73, "failed": 27, "cases": []}
    (question / "test.py").write_text(
        "import json\nprint('CYANML_RESULT=' + json.dumps("
        + repr(payload)
        + "))\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))

    assert client.post("/api/questions/01/test").json()["score"] == 73
    assert client.get("/api/questions").json()[0]["last_score"] == 73
