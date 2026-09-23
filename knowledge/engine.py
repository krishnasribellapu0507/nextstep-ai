import json
from pathlib import Path

from knowledge.generator import QuestionGenerator
from knowledge.normalization import normalize_topic


KNOWLEDGE_DIR = Path("data/knowledge")


class KnowledgeEngine:

    def __init__(self):
        self.generator = QuestionGenerator()
        self.topics = {}
        self.load_knowledge()

    def load_knowledge(self):
        self.topics = {}

        if not KNOWLEDGE_DIR.exists():
            return

        for file in KNOWLEDGE_DIR.glob("*.json"):
            data = json.loads(
                file.read_text(encoding="utf-8")
            )

            topic = normalize_topic(
                data["topic"]
            )

            self.topics[topic] = data

    def concepts_for(self, topic):
        topic = normalize_topic(topic)

        if topic not in self.topics:
            return []

        return self.topics[
            topic
        ].get("concepts", [])

    def get_concept(
        self,
        topic,
        concept_id,
    ):
        for concept in self.concepts_for(topic):
            if concept["id"] == concept_id:
                return concept

        return None

    def prerequisites_for(
        self,
        topic,
        concept_id,
    ):
        concept = self.get_concept(
            topic,
            concept_id,
        )

        if concept is None:
            return []

        return concept.get(
            "prerequisites",
            []
        )

    def choose_concept(
        self,
        topic,
        mastery,
        action,
    ):
        concepts = self.concepts_for(topic)

        if not concepts:
            return None

        concepts = sorted(
            concepts,
            key=lambda c: c["difficulty"],
        )

        if mastery < 0.30:
            return concepts[0]

        if action == "teach_foundation":
            return concepts[0]

        if action == "correct_misconception":
            if mastery < 0.50:
                return concepts[0]

            return concepts[
                min(1, len(concepts) - 1)
            ]

        if action == "guided_practice":
            if mastery < 0.50:
                index = 1
            else:
                index = 2

            return concepts[
                min(index, len(concepts) - 1)
            ]

        if action == "retrieval_practice":
            return concepts[
                len(concepts) // 2
            ]

        if action == "exam_priority":
            if mastery < 0.40:
                return concepts[0]

            return concepts[
                min(2, len(concepts) - 1)
            ]

        if action == "increase_difficulty":
            return concepts[-1]

        return concepts[0]

    def generate_question(
        self,
        topic,
        mastery,
        action,
    ):
        concept = self.choose_concept(
            topic,
            mastery,
            action,
        )

        if concept is None:
            return None, None

        try:
            question = self.generator.generate(
                concept["id"]
            )
        except ValueError:
            return concept, None

        return concept, question
