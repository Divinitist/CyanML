from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .repository import QuestionNotFoundError, QuestionRepository
from .runner import QuestionRunner


class SubmissionPayload(BaseModel):
    code: str


def create_app(questions_root: Path | None = None) -> FastAPI:
    root = questions_root or Path(__file__).resolve().parents[2] / "questions"
    repository = QuestionRepository(root)
    runner = QuestionRunner()
    application = FastAPI(title="CyanML API", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def question_path(question_id: str) -> Path:
        try:
            return repository.question_path(question_id)
        except QuestionNotFoundError as error:
            raise HTTPException(status_code=404, detail="题目不存在。") from error

    @application.get("/api/questions")
    def list_questions() -> list[dict[str, object]]:
        return repository.list_questions()

    @application.get("/api/questions/{question_id}")
    def get_question(question_id: str) -> dict[str, object]:
        try:
            return repository.get_question(question_id)
        except QuestionNotFoundError as error:
            raise HTTPException(status_code=404, detail="题目不存在。") from error

    @application.put(
        "/api/questions/{question_id}/submission",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def save_submission(question_id: str, payload: SubmissionPayload) -> Response:
        question_path(question_id)
        repository.save_submission(question_id, payload.code)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @application.post("/api/questions/{question_id}/reset")
    def reset_submission(question_id: str) -> dict[str, str]:
        question_path(question_id)
        return {"code": repository.reset_submission(question_id)}

    @application.post("/api/questions/{question_id}/test")
    def run_tests(question_id: str) -> dict[str, object]:
        result = runner.run(question_path(question_id))
        total = int(result["total"])
        score = round(int(result["passed"]) * 100 / total) if total else 0
        scored_result = {**result, "score": score}
        repository.save_last_result(question_id, scored_result)
        return scored_result

    return application


app = create_app()
