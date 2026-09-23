import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


DATA_FILE = Path("memory/student_progress.json")


@dataclass
class TopicProgress:
    topic: str

    mastery: float = 0.25
    attempts: int = 0
    correct_attempts: int = 0
    mistakes: int = 0
    hints_used: int = 0

    last_reviewed: str | None = None

    def mistake_rate(self):
        if self.attempts == 0:
            return 0.5

        return self.mistakes / self.attempts


    def forgetting_risk(self):
        if self.last_reviewed is None:
            return 0.5

        last = datetime.fromisoformat(
            self.last_reviewed
        )

        days = (
            datetime.now() - last
        ).total_seconds() / 86400

        # Risk rises gradually as time passes.
        risk = 1 - math.exp(-days / 5)

        return min(
            max(risk, 0.0),
            1.0
        )


    def record_attempt(
        self,
        correct: bool,
        hint_used: bool = False,
    ):
        self.attempts += 1

        if correct:
            self.correct_attempts += 1

            gain = 0.10

            if hint_used:
                gain *= 0.5

            self.mastery += gain

        else:
            self.mistakes += 1

            self.mastery -= 0.06

        if hint_used:
            self.hints_used += 1

        self.mastery = min(
            max(self.mastery, 0.0),
            1.0
        )

        self.last_reviewed = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )


class StudentModel:

    def __init__(self):
        self.topics = {}

        self.load()


    def get_topic(self, topic):

        key = topic.lower().strip()

        if key not in self.topics:

            self.topics[key] = TopicProgress(
                topic=topic
            )

        return self.topics[key]


    def record_attempt(
        self,
        topic,
        correct,
        hint_used=False,
    ):

        progress = self.get_topic(topic)

        progress.record_attempt(
            correct,
            hint_used,
        )

        self.save()

        return progress


    def save(self):

        DATA_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            key: asdict(value)
            for key, value
            in self.topics.items()
        }

        DATA_FILE.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )


    def load(self):

        if not DATA_FILE.exists():
            return

        try:
            raw = json.loads(
                DATA_FILE.read_text(
                    encoding="utf-8"
                )
            )

            for key, value in raw.items():

                self.topics[key] = TopicProgress(
                    **value
                )

        except Exception:
            self.topics = {}


if __name__ == "__main__":

    student = StudentModel()

    topic = "eigenvalues"

    progress = student.record_attempt(
        topic,
        correct=False,
    )

    print()
    print("NEXTSTEP STUDENT MODEL")
    print("======================")

    print("Topic:", progress.topic)
    print(
        "Mastery:",
        f"{progress.mastery:.0%}"
    )

    print(
        "Mistake rate:",
        f"{progress.mistake_rate():.0%}"
    )

    print(
        "Forgetting risk:",
        f"{progress.forgetting_risk():.0%}"
    )
