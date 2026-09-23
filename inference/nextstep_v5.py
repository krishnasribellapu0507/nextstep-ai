from pathlib import Path
import random

import torch

from evaluation.answer_checker import evaluate_answer
from knowledge.questions import QUESTIONS

from learning_engine.action_model import (
    ACTIONS,
    NextActionModel,
    state_tensor,
)

from memory.student_model import StudentModel

from model.transformer import (
    NextStepLM,
    ModelConfig,
)

from tokenizer.byte_tokenizer import ByteTokenizer


# =========================================================
# CONFIG
# =========================================================

POLICY_CHECKPOINT = Path(
    "checkpoints/nextstep-action-v1.pt"
)

LANGUAGE_CHECKPOINT = Path(
    "checkpoints/nextstep-v1-best.pt"
)

CONFIDENCE_THRESHOLD = 0.55


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = ByteTokenizer()
student_model = StudentModel()


# =========================================================
# LOAD LEARNING POLICY BRAIN
# =========================================================

policy_data = torch.load(
    POLICY_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

policy_model = NextActionModel().to(device)

policy_model.load_state_dict(
    policy_data["model_state_dict"]
)

policy_model.eval()


# =========================================================
# LOAD LANGUAGE BRAIN
# =========================================================

language_data = torch.load(
    LANGUAGE_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

config = ModelConfig(
    **language_data["config"]
)

language_model = NextStepLM(
    config
).to(device)

language_model.load_state_dict(
    language_data["model_state_dict"]
)

language_model.eval()


# =========================================================
# POLICY DECISION
# =========================================================

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
    )

    state = state.unsqueeze(0).to(device)

    logits = policy_model(state)

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
        probabilities,
    )


# =========================================================
# LANGUAGE RESPONSE
# =========================================================

@torch.no_grad()
def generate_guidance(
    topic,
    mastery,
    mistake_rate,
    forgetting_risk,
    exam_days,
    action,
):

    prompt = f"""<STUDENT>
I am studying {topic}. Tell me my next learning step.

<STATE>
topic={topic}
mastery={mastery:.2f}
mistake_rate={mistake_rate:.2f}
forgetting_risk={forgetting_risk:.2f}
exam_days={exam_days}

<ACTION>
{action}

<TUTOR>
"""

    encoded = tokenizer.encode(prompt)

    x = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device,
    )

    output = language_model.generate(
        x,
        max_new_tokens=100,
        temperature=0.35,
        top_k=5,
    )

    generated = output[
        0,
        len(encoded):
    ].tolist()

    response = tokenizer.decode(
        generated
    )

    for marker in [
        "<END>",
        "<STUDENT>",
        "<STATE>",
        "<ACTION>",
    ]:
        if marker in response:
            response = response.split(
                marker
            )[0]

    return response.strip()


# =========================================================
# QUESTION SELECTION
# =========================================================

def questions_for(topic):

    return [
        q for q in QUESTIONS
        if q.topic.lower()
        == topic.lower()
    ]


def choose_question(
    topic,
    action,
    last_misconception=None,
):

    available = questions_for(topic)

    if not available:
        return None


    # If we identified a specific misconception,
    # prefer another question targeting it.
    if last_misconception:

        targeted = [
            q for q in available
            if q.misconception
            == last_misconception
        ]

        if targeted:
            return random.choice(
                targeted
            )


    if action == "increase_difficulty":

        hardest = max(
            q.difficulty
            for q in available
        )

        choices = [
            q for q in available
            if q.difficulty == hardest
        ]

        return random.choice(
            choices
        )


    if action in {
        "teach_foundation",
        "exam_priority",
    }:

        easiest = min(
            q.difficulty
            for q in available
        )

        choices = [
            q for q in available
            if q.difficulty == easiest
        ]

        return random.choice(
            choices
        )


    if action == "guided_practice":

        choices = sorted(
            available,
            key=lambda q: q.difficulty,
        )

        return choices[
            min(
                len(choices) // 2,
                len(choices) - 1,
            )
        ]


    if action == "retrieval_practice":

        return random.choice(
            available
        )


    if action == "correct_misconception":

        return random.choice(
            available
        )


    return random.choice(
        available
    )


# =========================================================
# DISPLAY STATE
# =========================================================

def display_state(progress):

    print()
    print("STUDENT MODEL")
    print("=============")

    print(
        f"Topic:           "
        f"{progress.topic}"
    )

    print(
        f"Mastery:         "
        f"{progress.mastery:.0%}"
    )

    print(
        f"Mistake rate:    "
        f"{progress.mistake_rate():.0%}"
    )

    print(
        f"Forgetting risk: "
        f"{progress.forgetting_risk():.0%}"
    )

    print(
        f"Attempts:        "
        f"{progress.attempts}"
    )


# =========================================================
# MAIN ADAPTIVE LEARNING LOOP
# =========================================================

def run():

    print()
    print("NEXTSTEP AI V5")
    print("==============")
    print(
        "Adaptive learning intelligence"
    )
    print()

    topic = input(
        "What topic are you studying? "
    ).strip()

    exam_days = int(
        input(
            "Days until exam: "
        )
    )

    progress = student_model.get_topic(
        topic
    )

    last_misconception = None


    while True:

        display_state(progress)


        mastery = progress.mastery

        mistake_rate = (
            progress.mistake_rate()
        )

        forgetting_risk = (
            progress.forgetting_risk()
        )


        (
            action,
            confidence,
            probabilities,
        ) = choose_action(
            mastery,
            mistake_rate,
            forgetting_risk,
            exam_days,
        )


        print()
        print("NEXTSTEP DECISION")
        print("=================")

        print(
            f"Action:     {action}"
        )

        print(
            f"Confidence: "
            f"{confidence:.1%}"
        )


        # -------------------------------------------------
        # UNCERTAINTY HANDLING
        # -------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            print()
            print(
                "I'm not confident enough "
                "to choose a strong intervention yet."
            )

            print(
                "I'll use a diagnostic question "
                "to collect more evidence."
            )

            action_for_question = (
                "guided_practice"
            )

        else:

            action_for_question = action

            guidance = generate_guidance(
                topic,
                mastery,
                mistake_rate,
                forgetting_risk,
                exam_days,
                action,
            )

            print()
            print("NEXTSTEP")
            print("========")

            print(guidance)


        # -------------------------------------------------
        # QUESTION SELECTION
        # -------------------------------------------------

        question = choose_question(
            topic,
            action_for_question,
            last_misconception,
        )


        if question is None:

            print()
            print(
                "NextStep does not yet have "
                f"practice questions for '{topic}'."
            )

            print(
                "We'll add this topic to the "
                "knowledge system later."
            )

            break


        print()
        print("QUESTION")
        print("========")

        print(question.prompt)


        answer = input(
            "\nYour answer: "
        ).strip()


        if answer.lower() in {
            "quit",
            "exit",
            "stop",
        }:

            print()
            print(
                "Progress saved."
            )

            break


        # -------------------------------------------------
        # AUTOMATIC ANSWER EVALUATION
        # -------------------------------------------------

        result = evaluate_answer(
            question,
            answer,
        )


        print()
        print("ANSWER ANALYSIS")
        print("===============")


        if result.correct:

            print("✓ Correct")

            last_misconception = None

        else:

            print("✗ Incorrect")

            print(
                "Detected issue:",
                result.reason,
            )

            print(
                "Expected:",
                result.expected,
            )

            last_misconception = (
                result.reason
            )


        # -------------------------------------------------
        # AUTOMATIC STUDENT MODEL UPDATE
        # -------------------------------------------------

        progress = student_model.record_attempt(
            topic,
            correct=result.correct,
            hint_used=False,
        )


        print()
        print("LEARNING UPDATE")
        print("===============")

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


        print()
        again = input(
            "Continue learning? (y/n): "
        ).strip().lower()

        if again != "y":

            print()
            print(
                "Your progress has been saved."
            )

            break


if __name__ == "__main__":
    run()
