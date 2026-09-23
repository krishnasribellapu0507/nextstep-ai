from knowledge.engine import KnowledgeEngine
from knowledge.normalization import normalize_topic


class AdaptiveKnowledgeEngine:

    def __init__(
        self,
        concept_memory,
    ):
        self.knowledge = KnowledgeEngine()
        self.memory = concept_memory

    def prerequisite_ready(
        self,
        topic,
        concept,
    ):
        topic = normalize_topic(topic)

        prerequisites = concept.get(
            "prerequisites",
            []
        )

        for prerequisite in prerequisites:
            state = self.memory.get(
                topic,
                prerequisite,
            )

            if state is None:
                return False

            if state.mastery < 0.60:
                return False

        return True

    def choose_concept(
        self,
        topic,
        action,
    ):
        topic = normalize_topic(topic)

        concepts = (
            self.knowledge.concepts_for(
                topic
            )
        )

        if not concepts:
            return None

        states = []

        for concept in concepts:
            state = self.memory.get(
                topic,
                concept["id"],
            )

            if state is not None:
                states.append(
                    (concept, state)
                )

        if not states:
            return None

        ready = [
            pair
            for pair in states
            if self.prerequisite_ready(
                topic,
                pair[0],
            )
        ]

        if not ready:
            ready = states

        if action == "correct_misconception":
            return max(
                ready,
                key=lambda x:
                    x[1].mistake_rate(),
            )[0]

        if action == "retrieval_practice":
            return max(
                ready,
                key=lambda x:
                    x[1].forgetting_risk(),
            )[0]

        if action == "increase_difficulty":
            return max(
                ready,
                key=lambda x:
                    x[0]["difficulty"],
            )[0]

        if action == "teach_foundation":
            return min(
                ready,
                key=lambda x: (
                    x[0]["difficulty"],
                    x[1].mastery,
                ),
            )[0]

        return min(
            ready,
            key=lambda x:
                x[1].mastery,
        )[0]

    def generate_question(
        self,
        topic,
        action,
    ):
        topic = normalize_topic(topic)

        concept = self.choose_concept(
            topic,
            action,
        )

        if concept is None:
            return None, None

        try:
            question = (
                self.knowledge.generator.generate(
                    concept["id"]
                )
            )
        except ValueError:
            return concept, None

        return concept, question
