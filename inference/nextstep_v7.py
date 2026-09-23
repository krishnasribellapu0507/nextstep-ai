from pathlib import Path

import torch

from evaluation.answer_checker import (
    evaluate_answer,
)

from knowledge.engine import (
    KnowledgeEngine,
)

from knowledge.adaptive_engine import (
    AdaptiveKnowledgeEngine,
)

from learning_engine.action_model import (
    ACTIONS,
    NextActionModel,
    state_tensor,
)

from memory.concept_memory import (
    ConceptMemory,
)


POLICY_CHECKPOINT = Path(
    "checkpoints/nextstep-action-v1.pt"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================
# LOAD POLICY NETWORK
# =====================================================

checkpoint = torch.load(
    POLICY_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

policy = NextActionModel().to(
    device
)

policy.load_state_dict(
    checkpoint["model_state_dict"]
)

policy.eval()


# =====================================================
# SYSTEM COMPONENTS
# =====================================================

memory = ConceptMemory()
knowledge = KnowledgeEngine()

adaptive = AdaptiveKnowledgeEngine(
    memory
)


@torch.no_grad()
def choose_action(
    mastery,
    mistake_rate,
    forgetting_risk,
    exam_days,
):

    state = state_tensor(
        mastery,
        mistake_rate,
        forgetting_risk,
        exam_days,
    ).unsqueeze(0).to(device)

    logits = policy(state)

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    index = torch.argmax(
        probabilities
    ).item()

    return (
        ACTIONS[index],
        probabilities[index].item(),
    )


def show_concepts(topic):

    print()
    print("CONCEPT MASTERY MAP")
    print("===================")

    states = memory.topic_progress(
        topic
    )

    for state in sorted(
        states,
        key=lambda x:
            x.mastery
    ):

        print(
            f"{state.concept_name:30} "
            f"{state.mastery:>6.0%} "
            f"| attempts "
            f"{state.attempts}"
        )


def run():

    print()
    print("NEXTSTEP AI V7")
    print("==============")
    print(
        "Concept-level adaptive learning"
    )

    print()

    topic = input(
        "Topic: "
    ).strip().lower()

    exam_days = int(
        input(
            "Days until exam: "
        )
    )

    concepts = (
        knowledge.concepts_for(
            topic
        )
    )

    if not concepts:

        print(
            f"No knowledge map exists "
            f"for '{topic}' yet."
        )
        return

    memory.ensure_topic(
        topic,
        concepts,
    )


    while True:

        show_concepts(topic)

        (
            mastery,
            mistake_rate,
            forgetting_risk,
        ) = memory.topic_summary(
            topic
        )


        action, confidence = (
            choose_action(
                mastery,
                mistake_rate,
                forgetting_risk,
                exam_days,
            )
        )


        print()
        print("NEXTSTEP DECISION")
        print("=================")

        print(
            f"Overall mastery: "
            f"{mastery:.0%}"
        )

        print(
            f"Action:          "
            f"{action}"
        )

        print(
            f"Confidence:      "
            f"{confidence:.1%}"
        )


        concept, question = (
            adaptive.generate_question(
                topic,
                action,
            )
        )


        if (
            concept is None
            or question is None
        ):

            print(
                "No question generator "
                "exists for the selected concept."
            )

            break


        print()
        print("NEXT CONCEPT")
        print("============")

        print(
            concept["name"]
        )

        prerequisites = concept.get(
            "prerequisites",
            []
        )

        if prerequisites:

            print(
                "Prerequisites:",
                ", ".join(
                    prerequisites
                )
            )


        print()
        print("QUESTION")
        print("========")

        print(
            question.prompt
        )


        answer = input(
            "\nYour answer: "
        ).strip()


        if answer.lower() in {
            "quit",
            "exit",
            "stop",
        }:

            print(
                "Progress saved."
            )

            break


        result = evaluate_answer(
            question,
            answer,
        )


        print()
        print("ANSWER ANALYSIS")
        print("===============")

        if result.correct:

            print("✓ Correct")

        else:

            print("✗ Incorrect")

            print(
                "Misconception:",
                result.reason,
            )

            print(
                "Expected:",
                result.expected,
            )


        state = memory.record_attempt(
            topic,
            question.concept,
            correct=result.correct,
            hint_used=False,
        )


        print()
        print("CONCEPT UPDATED")
        print("===============")

        print(
            state.concept_name
        )

        print(
            "Mastery:",
            f"{state.mastery:.0%}"
        )

        print(
            "Mistake rate:",
            f"{state.mistake_rate():.0%}"
        )


        again = input(
            "\nContinue? (y/n): "
        ).strip().lower()

        if again != "y":

            print()
            print(
                "Learning state saved."
            )

            break


if __name__ == "__main__":
    run()
