import pytest

from questions.test_support import run_cases


def test_run_cases_captures_standard_output_per_case(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {"name": "first", "args": (1,)},
        {"name": "second", "args": (2,)},
    ]

    def candidate(value: int) -> int:
        print(f"candidate case {value}")
        return value

    result = run_cases(
        cases,
        candidate,
        lambda value: value,
        rtol=0.0,
        atol=0.0,
        mismatch_message="mismatch",
    )

    assert result["cases"][0]["stdout"] == "candidate case 1"
    assert result["cases"][1]["stdout"] == "candidate case 2"
    assert capsys.readouterr().out == ""


def test_run_cases_keeps_case_output_when_candidate_raises() -> None:
    def candidate() -> None:
        print("printed before failure")
        raise ValueError("broken")

    result = run_cases(
        [{"name": "failure", "args": ()}],
        candidate,
        lambda: None,
        rtol=0.0,
        atol=0.0,
        mismatch_message="mismatch",
    )

    assert result["cases"][0]["stdout"] == "printed before failure"
    assert result["cases"][0]["error"] == "ValueError: broken"
