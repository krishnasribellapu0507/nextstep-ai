import re
from dataclasses import dataclass

from sympy import sympify


@dataclass
class EvaluationResult:
    correct: bool
    normalized_answer: object
    expected: object
    reason: str


def parse_numbers(text):
    return [
        float(value)
        for value in re.findall(
            r"-?\d+(?:\.\d+)?",
            text,
        )
    ]


def evaluate_answer(
    question,
    student_answer,
):
    answer = student_answer.strip()

    if question.answer_type == "number":

        try:
            student_value = float(
                sympify(answer)
            )

            expected = float(
                question.expected
            )

            correct = (
                abs(
                    student_value
                    - expected
                )
                < 1e-6
            )

            return EvaluationResult(
                correct=correct,
                normalized_answer=student_value,
                expected=expected,
                reason=(
                    "correct"
                    if correct
                    else question.misconception
                ),
            )

        except Exception:

            return EvaluationResult(
                correct=False,
                normalized_answer=answer,
                expected=question.expected,
                reason="could_not_parse_answer",
            )

    if question.answer_type == "number_set":

        student_values = sorted(
            parse_numbers(answer)
        )

        expected_values = sorted(
            float(x)
            for x in question.expected
        )

        correct = (
            len(student_values)
            == len(expected_values)
            and all(
                abs(a - b) < 1e-6
                for a, b
                in zip(
                    student_values,
                    expected_values,
                )
            )
        )

        return EvaluationResult(
            correct=correct,
            normalized_answer=student_values,
            expected=expected_values,
            reason=(
                "correct"
                if correct
                else question.misconception
            ),
        )

    return EvaluationResult(
        correct=False,
        normalized_answer=answer,
        expected=question.expected,
        reason="unsupported_answer_type",
    )
