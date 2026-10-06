from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class QuestionNotFoundError(KeyError):
    """Raised when a question id does not exist in the bank."""


class QuestionRepository:
    def __init__(self, questions_root: Path) -> None:
        self.questions_root = questions_root
        self.scores_path = questions_root / ".scores.json"
        self.results_path = questions_root / ".results"

    def list_questions(self) -> list[dict[str, Any]]:
        scores = self._read_scores()
        questions = []
        for path in self.questions_root.iterdir():
            if not path.is_dir() or not (path / "manifest.json").is_file():
                continue
            manifest = self._read_manifest(path)
            questions.append({**manifest, "last_score": scores.get(manifest["id"])})
        return sorted(
            questions,
            key=lambda question: (
                int(question.get("order", question["id"])),
                question["id"],
            ),
        )

    def get_question(self, question_id: str) -> dict[str, Any]:
        path = self._question_path(question_id)
        return {
            **self._read_manifest(path),
            "last_score": self._read_scores().get(question_id),
            "last_result": self._read_last_result(question_id),
            "description": (path / "problem.md").read_text(encoding="utf-8"),
            "template": (path / "template.py").read_text(encoding="utf-8"),
            "submission": (path / "submission.py").read_text(encoding="utf-8"),
            "answer": (path / "answer.py").read_text(encoding="utf-8"),
        }

    def save_submission(self, question_id: str, code: str) -> None:
        path = self._question_path(question_id) / "submission.py"
        path.write_text(code, encoding="utf-8")

    def reset_submission(self, question_id: str) -> str:
        question_path = self._question_path(question_id)
        code = (question_path / "template.py").read_text(encoding="utf-8")
        (question_path / "submission.py").write_text(code, encoding="utf-8")
        return code

    def question_path(self, question_id: str) -> Path:
        return self._question_path(question_id)

    def save_last_score(self, question_id: str, score: int) -> None:
        self._question_path(question_id)
        scores = self._read_scores()
        scores[question_id] = int(score)
        temporary_path = self.scores_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(scores, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.scores_path)

    def save_last_result(self, question_id: str, result: dict[str, Any]) -> None:
        self._question_path(question_id)
        self.results_path.mkdir(exist_ok=True)
        result_path = self.results_path / f"{question_id}.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(result_path)
        self.save_last_score(question_id, int(result["score"]))

    def _question_path(self, question_id: str) -> Path:
        for path in self.questions_root.iterdir():
            if not path.is_dir() or not (path / "manifest.json").is_file():
                continue
            if self._read_manifest(path)["id"] == question_id:
                return path
        raise QuestionNotFoundError(question_id)

    @staticmethod
    def _read_manifest(question_path: Path) -> dict[str, Any]:
        return json.loads((question_path / "manifest.json").read_text(encoding="utf-8"))

    def _read_scores(self) -> dict[str, int]:
        if not self.scores_path.is_file():
            return {}
        return json.loads(self.scores_path.read_text(encoding="utf-8"))

    def _read_last_result(self, question_id: str) -> dict[str, Any] | None:
        result_path = self.results_path / f"{question_id}.json"
        if not result_path.is_file():
            return None
        return json.loads(result_path.read_text(encoding="utf-8"))
