import random

from knowledge.questions import QUESTIONS
from evaluation.answer_checker import evaluate_answer
from memory.student_model import StudentModel


student = StudentModel()


def questions_for(topic):
    return [
        q for q in QUESTIONS
        if q.topic.lower() == topic.lower()
    ]


def run():

    print()
    print("NEXTSTEP AI V4")
    print("==============")
    print("Automatic answer evaluation")
    print()

    topic = input(
        "Topic: "
    ).strip()

    available = questions_for(topic)

    if not available:
        print(
            "I do not yet have diagnostic "
            f"questions for {topic}."
        )
        return

    progress = student.get_topic(topic)

    print()
    print(
        f"Current mastery: "
        f"{progress.mastery:.0%}"
    )

    print(
        f"Previous attempts: "
        f"{progress.attempts}"
    )

    while True:

        question = random.choice(
            available
        )

        print()
        print("QUESTION")
        print("========")
        print(question.prompt)

        answer = input(
            "\nYour answer: "
        )

        result = evaluate_answer(
            question,
            answer,
        )

        print()

        if result.correct:

            print("✓ Correct")

        else:

            print("✗ Not quite")

            print(
                "Detected issue:",
                result.reason,
            )

            print(
                "Expected:",
                result.expected,
            )


        progress = student.record_attempt(
            topic,
            correct=result.correct,
            hint_used=False,
        )

        print()
        print("STUDENT MODEL UPDATED")
        print("=====================")

        print(
            f"Mastery:      "
            f"{progress.mastery:.0%}"
        )

        print(
            f"Mistake rate: "
            f"{progress.mistake_rate():.0%}"
        )

        print(
            f"Attempts:     "
            f"{progress.attempts}"
        )


        again = input(
            "\nAnother question? (y/n): "
        ).strip().lower()

        if again != "y":
            break


if __name__ == "__main__":
    run()
