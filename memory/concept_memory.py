import json
import math

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from knowledge.normalization import normalize_topic


DATA_FILE = Path(
    "memory/concept_progress.json"
)


@dataclass
class ConceptProgress:
    topic: str
    concept_id: str
    concept_name: str

    mastery: float = 0.25
    attempts: int = 0
    correct_attempts: int = 0
    mistakes: int = 0
    hints_used: int = 0

    last_reviewed: str | None = None

    misconceptions: dict = field(
        default_factory=dict
    )

    def mistake_rate(self):
        if self.attempts == 0:
            return 0.50

        return self.mistakes / self.attempts

    def forgetting_risk(self):
        if self.last_reviewed is None:
            return 0.50

        try:
            last = datetime.fromisoformat(
                self.last_reviewed
            )
        except ValueError:
            return 0.50

        days = (
            datetime.now() - last
        ).total_seconds() / 86400

        risk = 1 - math.exp(
            -days / 5
        )

        return min(
            max(risk, 0.0),
            1.0
        )

    def record_attempt(
        self,
        correct,
        hint_used=False,
        misconception=None,
    ):
        self.attempts += 1

        if correct:
            self.correct_attempts += 1

            gain = 0.10

            if hint_used:
                gain *= 0.50

            self.mastery += gain

        else:
            self.mistakes += 1
            self.mastery -= 0.06

            if misconception:
                self.misconceptions[
                    misconception
                ] = (
                    self.misconceptions.get(
                        misconception,
                        0,
                    )
                    + 1
                )

        if hint_used:
            self.hints_used += 1

        self.mastery = min(
            max(self.mastery, 0.0),
            1.0,
        )

        self.last_reviewed = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )


class ConceptMemory:

    def __init__(
        self,
        student_id="local",
        data_file=DATA_FILE,
    ):
        self.student_id = (
            str(student_id).strip()
            or "local"
        )

        self.data_file = Path(
            data_file
        )

        self.progress = {}

        self.load()

    def make_key(
        self,
        topic,
        concept_id,
    ):
        topic = normalize_topic(topic)

        return (
            f"{self.student_id}"
            f"::{topic}"
            f"::{concept_id}"
        )

    def ensure_topic(
        self,
        topic,
        concepts,
    ):
        topic = normalize_topic(topic)

        changed = False

        for concept in concepts:
            key = self.make_key(
                topic,
                concept["id"],
            )

            if key not in self.progress:
                self.progress[key] = (
                    ConceptProgress(
                        topic=topic,
                        concept_id=concept["id"],
                        concept_name=concept["name"],
                    )
                )

                changed = True

        if changed:
            self.save()

    def get(
        self,
        topic,
        concept_id,
    ):
        return self.progress.get(
            self.make_key(
                topic,
                concept_id,
            )
        )

    def record_attempt(
        self,
        topic,
        concept_id,
        correct,
        hint_used=False,
        misconception=None,
    ):
        state = self.get(
            topic,
            concept_id,
        )

        if state is None:
            raise ValueError(
                f"Unknown concept: {concept_id}"
            )

        state.record_attempt(
            correct=correct,
            hint_used=hint_used,
            misconception=misconception,
        )

        self.save()

        return state

    def topic_progress(
        self,
        topic,
    ):
        topic = normalize_topic(topic)

        prefix = (
            f"{self.student_id}"
            f"::{topic}::"
        )

        return [
            value
            for key, value
            in self.progress.items()
            if key.startswith(prefix)
        ]

    def topic_summary(
        self,
        topic,
    ):
        states = self.topic_progress(
            topic
        )

        if not states:
            return (
                0.25,
                0.50,
                0.50,
            )

        mastery = sum(
            x.mastery
            for x in states
        ) / len(states)

        mistakes = sum(
            x.mistake_rate()
            for x in states
        ) / len(states)

        forgetting = sum(
            x.forgetting_risk()
            for x in states
        ) / len(states)

        return (
            mastery,
            mistakes,
            forgetting,
        )

    def save(self):
        self.data_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            key: asdict(value)
            for key, value
            in self.progress.items()
        }

        self.data_file.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self):
        if not self.data_file.exists():
            return

        try:
            raw = json.loads(
                self.data_file.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return

        for key, value in raw.items():

            # Migrate old local records:
            # topic::concept
            if key.count("::") == 1:
                key = "local::" + key

            try:
                self.progress[key] = (
                    ConceptProgress(
                        **value
                    )
                )
            except TypeError:
                continue
